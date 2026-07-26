"""
DeRTa - retrain on our own base model (smoke-sized by default).

Wraps upstream's own trainer `repo/run_files/run_clm_lora_derta_llama.py`. Upstream's
`train.sh` is written for 8 GPUs + DeepSpeed and targets Llama-3-70B; this launches the same
script on ONE GPU with the 8B model and smoke-sized steps.

Upstream LoRA settings (from train.sh, the 70B branch - the 8B branch there is full-param):
  batch 8 x grad-accum 2, 2 epochs, lr 1e-4, block_size 512, bf16, gradient checkpointing
Upstream never reports a wall-clock time.

Data is generated locally, nothing to download:
  cd repo/data/train && python generate_training_data.py
produces three files, and which one you pick IS the ablation:
  llama_derta.json    - DeRTa: harmful-prefix samples + RTO transition targets  <- the method
  llama_vanilla.json  - plain safety SFT                                        (baseline)
  llama_recaug.json   - MLE with harmful response prefix only                   (baseline)

Usage:
  python train_smoke.py                       # generates data if missing, ~10 steps
  python train_smoke.py --dataset llama_vanilla
  python train_smoke.py --full                # 2 epochs like upstream
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "repo")
TRAIN_PY = os.path.join("run_files", "run_clm_lora_derta_llama.py")
DATA_DIR = os.path.join(REPO, "data", "train")


ENTRY_COMPAT = os.path.join("run_files", "run_clm_lora_derta_llama_compat.py")

# Upstream calls this unconditionally at module level:
#     deepspeed.ops.op_builder.CPUAdamBuilder().load()
# which JIT-compiles DeepSpeed's cpu_adam kernel. On a box without ninja + a full CUDA build
# toolchain that dies with:
#     RuntimeError: Error building extension 'cpu_adam' ... Command '['ninja','-v']' failed
# It is only needed for DeepSpeed CPU-offload, which a single-GPU LoRA run never touches.
COMPAT_PATCHES = [
    ("deepspeed.ops.op_builder.CPUAdamBuilder().load()",
     "(deepspeed.ops.op_builder.CPUAdamBuilder().load()\n"
     "     if os.environ.get('DERTA_BUILD_CPU_ADAM') == '1' else None)",
     "RuntimeError: Error building extension 'cpu_adam' (needs ninja + CUDA toolchain); "
     "only used for DeepSpeed CPU offload"),
]


def make_compat_entry():
    """Launch a patched COPY of upstream's trainer so repo/ stays a pristine clone."""
    src = os.path.join(REPO, TRAIN_PY)
    with open(src, encoding="utf-8") as f:
        code = f.read()
    applied = []
    for old, new, why in COMPAT_PATCHES:
        if old in code and new not in code:
            code = code.replace(old, new, 1)
            applied.append(why)
    if not applied:
        print(f"[compat] no patch needed -> {TRAIN_PY}")
        return TRAIN_PY
    if "\nimport os" not in code and "^import os" not in code:
        code = "import os\n" + code
    with open(os.path.join(REPO, ENTRY_COMPAT), "w", encoding="utf-8") as f:
        f.write(code)
    print(f"[compat] {len(applied)} patch -> {ENTRY_COMPAT}")
    for w in applied:
        print(f"[compat]   skips: {w}")
    return ENTRY_COMPAT


def ensure_data(name):
    path = os.path.join(DATA_DIR, f"{name}.json")
    if os.path.exists(path):
        print(f"[data] {path} already there")
        return path
    print("[data] generating training data (upstream generate_training_data.py)...")
    rc = subprocess.call([sys.executable, "generate_training_data.py"], cwd=DATA_DIR)
    if rc != 0 or not os.path.exists(path):
        sys.exit(f"[data] generation failed (exit={rc}); expected {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get(
        "LOCAL_TARGET_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"))
    ap.add_argument("--dataset", default="llama_derta",
                    choices=["llama_derta", "llama_vanilla", "llama_recaug"])
    ap.add_argument("--full", action="store_true", help="2 epochs like upstream")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    train_file = ensure_data(a.dataset)
    out = a.out or os.path.join(HERE, "train_out", f"lora_{a.dataset}")
    os.makedirs(out, exist_ok=True)

    env = dict(os.environ)
    env["WANDB_DISABLED"] = "true"

    cmd = [
        sys.executable, make_compat_entry(),
        "--model_name_or_path", a.base,
        "--train_file", os.path.relpath(train_file, REPO).replace("\\", "/"),
        "--validation_file", "data/train/example_data_to_read.json",
        "--use_lora", "True",
        "--lora_config", "train_config/lora_config.json",
        "--preprocessing_num_workers", "4",
        "--dataloader_num_workers", "0",
        "--per_device_train_batch_size", str(a.batch),
        "--per_device_eval_batch_size", "1",
        "--gradient_accumulation_steps", "2",
        "--learning_rate", "1e-4",
        "--weight_decay", "0.",
        "--warmup_ratio", "0.00",
        "--logging_steps", "1",
        "--block_size", "512",
        "--do_train",
        "--evaluation_strategy", "no",
        "--bf16", "True",
        "--seed", "1",
        "--gradient_checkpointing", "True",
        "--output_dir", out,
        "--overwrite_output_dir",
        "--save_strategy", "no",
    ]
    # Single GPU -> no deepspeed, no torchrun. Upstream passes --streaming; keep it only for
    # the full run since streaming + max_steps interact badly on tiny runs.
    if a.full:
        cmd += ["--num_train_epochs", "2", "--streaming"]
    else:
        cmd += ["--max_steps", str(a.steps), "--num_train_epochs", "1"]

    print("[run]", " ".join(cmd), f"\n[cwd] {REPO}\n")
    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=REPO, env=env)
    print(f"\n[done] exit={rc} | {time.perf_counter() - t0:.1f}s")
    if rc == 0:
        print("Point method.py at the adapter you just trained:")
        print(f"  DERTA_BASE={a.base} DERTA_LORA={out} python method.py response --limit 5")
    sys.exit(rc)


if __name__ == "__main__":
    main()
