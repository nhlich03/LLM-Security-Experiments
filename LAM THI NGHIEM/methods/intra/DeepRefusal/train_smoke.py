"""
DeepRefusal - train lai tren Llama-3 (smoke-size).

CANH BAO TRUOC KHI CHAY: repo upstream chi co DUNG 3 FILE PYTHON
(`src/args.py`, `src/main.py`, `src/train_dataset.py`) - khong co lenh train, khong co
data, khong co script. README noi thang la phai sang 2 repo khac. Script nay lam cong
viec lap rap do, va tung mieng lap rap deu la XAP XI, phai khai bao trong bao cao.

Ba thu `main.py` doi ma repo KHONG co:

  1. `direction_path` -> file .pt chua REFUSAL DIRECTION.
     Goc: repo andyrdt/refusal_direction, mot pipeline rieng.
     O day tu tinh theo dung dinh nghia cua Arditi et al.: lay trung binh hidden state
     o vi tri token cuoi tren tap harmful tru trung binh tren tap harmless, roi chuan
     hoa. Tinh o layer giua (mac dinh 14/32). => DUNG CONG THUC, nhung tap prompt va
     layer khac ban goc -> huong khong trung khit.

  2. `dataset/ultrachat_200k-test_sft.arrow` -> retain set (benign).
     Tai tu HuggingFaceH4/ultrachat_200k (split test_sft) roi ghi ra .arrow.

  3. `./data/train/circuit_breakers_train_processed_2k.json` -> harmful set.
     Goc la ban da xu ly cua CircuitBreaker. Repo CircuitBreakers minh da clone san co
     `data/circuit_breakers_train.json` -> dung lai file do, cat 2000 dong dau.

Ngoai ra `train_dataset.py` DA co nhanh Llama-3 san (dong 38) -> khong phai va model.

Vi vay model tu train ra day chi la "DeepRefusal-style", KHONG tai lap bit-exact.
Bang ket qua chinh phai dung checkpoint tac gia:
  skysys00/Meta-Llama-3-8B-Instruct-DeepRefusal

Cau hinh upstream (paper): 1 epoch, LoRA r=16 alpha=16, batch 16, PAA p=0.5, ~45 phut
tren 1xA100-80GB.

Run (needs GPU):
  python train_smoke.py                 # smoke: ~10 step
  python train_smoke.py --full          # 1 epoch nhu paper
  python train_smoke.py --stage prep    # chi lap rap data, khong train
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REPO = os.path.join(HERE, "repo")
WORK = os.path.join(HERE, "work")          # noi chay - repo/ giu nguyen

CB_TRAIN = os.path.join(ROOT, "methods", "intra", "CircuitBreakers", "repo",
                        "data", "circuit_breakers_train.json")

DEFAULT_BASE = os.environ.get("DR_BASE", "NousResearch/Meta-Llama-3-8B-Instruct")


def log(m):
    print(f"[deeprefusal] {m}", flush=True)


# ----- Lap rap 3 thu upstream thieu -----
def prep(base, n_harmful, direction_layer):
    os.makedirs(WORK, exist_ok=True)
    shutil.copytree(os.path.join(REPO, "src"), os.path.join(WORK, "src"), dirs_exist_ok=True)
    os.makedirs(os.path.join(WORK, "data", "train"), exist_ok=True)
    os.makedirs(os.path.join(WORK, "dataset"), exist_ok=True)

    # --- (3) harmful set: muon tu repo CircuitBreakers da clone ---
    out_cb = os.path.join(WORK, "data", "train", "circuit_breakers_train_processed_2k.json")
    if not os.path.exists(out_cb):
        if not os.path.exists(CB_TRAIN):
            sys.exit(f"THIEU {CB_TRAIN} - clone repo CircuitBreakers truoc")
        data = json.load(open(CB_TRAIN, encoding="utf-8"))[:n_harmful]
        json.dump(data, open(out_cb, "w", encoding="utf-8"))
        log(f"harmful set: {len(data)} muc <- circuit_breakers_train.json")

    # --- (2) retain set: ultrachat_200k test_sft -> .arrow ---
    out_uc = os.path.join(WORK, "dataset", "ultrachat_200k-test_sft.arrow")
    if not os.path.exists(out_uc):
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
        ds = ds.select(range(min(len(ds), max(n_harmful * 2, 200))))
        ds.data.table  # noqa - cham de chac chan da materialize
        from datasets import Dataset
        Dataset.from_dict(ds.to_dict()).save_to_disk(out_uc + ".hf")
        # loader cua ho goi load_dataset("arrow", data_files=...) -> can file .arrow that
        import pyarrow as pa
        with pa.OSFile(out_uc, "wb") as sink:
            tbl = ds.data.table
            with pa.RecordBatchFileWriter(sink, tbl.schema) as w:
                w.write_table(tbl)
        log(f"retain set: {len(ds)} muc -> {out_uc}")

    # --- (1) refusal direction: TU TINH (xap xi) ---
    out_dir = os.path.join(WORK, "refusal_direction.pt")
    if not os.path.exists(out_dir):
        compute_refusal_direction(base, out_dir, direction_layer, n_harmful)
    return out_cb, out_uc, out_dir


def compute_refusal_direction(base, out_path, layer, n):
    """Arditi et al.: d = normalize(mean_harmful(h_last) - mean_harmless(h_last)) o 1 layer.

    Dung dung cong thuc nhung tap prompt lay tu chinh 2 file train o tren, khong phai
    tap cua repo refusal_direction -> huong khac ban goc.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"tinh refusal direction tai layer {layer} (xap xi Arditi et al.)")
    tok = AutoTokenizer.from_pretrained(base)
    kw = {"device_map": "auto"}
    import transformers
    kw["dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(base, **kw)
    model.eval()

    cb = json.load(open(os.path.join(WORK, "data", "train",
                                     "circuit_breakers_train_processed_2k.json"), encoding="utf-8"))
    harmful = [d.get("prompt") or d.get("instruction") or "" for d in cb][:n]
    harmful = [h for h in harmful if h][:n]

    from datasets import load_dataset
    uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft").select(range(len(harmful)))
    harmless = [m["messages"][0]["content"] for m in uc]

    def mean_hidden(prompts):
        acc = None
        with torch.no_grad():
            for p in prompts:
                ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                              add_generation_prompt=True, return_tensors="pt")
                h = model(ids.to(model.device), output_hidden_states=True
                          ).hidden_states[layer][0, -1].float().cpu()
                acc = h if acc is None else acc + h
        return acc / len(prompts)

    d = mean_hidden(harmful) - mean_hidden(harmless)
    d = d / d.norm()
    torch.save(d, out_path)
    log(f"refusal direction {tuple(d.shape)} -> {out_path}")
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--full", action="store_true", help="1 epoch nhu paper")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--n-harmful", type=int, default=None)
    ap.add_argument("--direction-layer", type=int, default=14)
    ap.add_argument("--stage", default="all", choices=["all", "prep", "train"])
    args = ap.parse_args()

    n_harmful = args.n_harmful or (2000 if args.full else 40)
    cb, uc, direction = prep(args.base, n_harmful, args.direction_layer)
    if args.stage == "prep":
        return

    out = os.path.join(WORK, "train_out")
    cmd = [sys.executable, "src/main.py",
           "--model_name_or_path", args.base,
           "--output_dir", out,
           "--direction_path", direction,
           "--transform_layers", "10,12,14,16,18,20",
           "--lora_r", "16", "--lora_alpha", "16",
           "--ablation_prob", "0.5",                    # PAA p=0.5 nhu paper
           "--per_device_train_batch_size", "2" if not args.full else "4",
           "--gradient_accumulation_steps", "8" if args.full else "1",
           "--num_train_epochs", "1",
           "--model_max_length", "512" if not args.full else "1024",
           "--logging_steps", "1", "--save_steps", "10000",
           "--bf16", "True", "--report_to", "none"]
    if not args.full:
        cmd += ["--max_steps", str(args.steps)]

    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
    log("train: " + " ".join(cmd[1:]))
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=WORK, env=env)
    log(f"exit={r.returncode} trong {time.perf_counter()-t0:.1f}s")
    if r.returncode == 0:
        log(f"adapter -> {out}")
        log(f"chay thu:  DR_MODEL={args.base} DR_LORA={out} python method.py response --task harmbench --limit 3")
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
