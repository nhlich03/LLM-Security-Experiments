"""
DRO (Directed Representation Optimization) - train the continuous safety prompt.

Upstream is a THREE-STAGE pipeline, all of it needed before you can generate:

  1. forward.py   -> dump last-layer hidden states for 100 harmful (data/custom.txt)
                     and 100 harmless (data_harmless/custom.txt) queries
  2. estimate.py  -> from those hidden states, estimate the refusal direction and the
                     harmfulness direction (PCA basis + logistic boundaries)
  3. train.py     -> optimise a soft prompt so harmful representations move ALONG the
                     refusal direction and harmless ones move AGAINST it.
                     Saves trained_prompts/<model>/type.all_length.<len>.safetensors

Two things upstream cannot do out of the box, both patched into a COPY (repo/ untouched):

  A. Llama-3 is not in the model whitelist. `train.py`, `forward.py`, `generate.py`,
     `forward_with_soft.py`, `train_unlikelihood.py` all end their model check with
         else: raise ValueError(f"Unsupported or untuned model: {model_name}")
     so anything that is not Llama-2 / CodeLlama / Orca-2 / Mistral / Vicuna / OpenChat
     dies immediately. We add a Llama-3 branch.

  B. Those files force-override the tokenizer chat template from a .jinja file, and
     `chat_templates/` has no Llama-3 one. Rather than hand-writing a template (easy to
     get subtly wrong, and it decides every tokenised position), we let the Llama-3
     tokenizer keep its OWN built-in template - that one is Meta's official.
     Patch: when the generation config has "chat_template": null, skip the override.

Run (needs GPU):
  python train_smoke.py                 # smoke: 20 query moi ben, it step
  python train_smoke.py --full          # cau hinh upstream day du (100 + 100)
  python train_smoke.py --stage forward # chay lai mot stage
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_CODE = os.path.join(HERE, "repo", "code")
WORK = os.path.join(HERE, "work")                  # ban da va - repo/ giu nguyen

DEFAULT_BASE = os.environ.get(
    "DRO_BASE", "NousResearch/Meta-Llama-3-8B-Instruct")

# ----- Patch A: them nhanh Llama-3 vao whitelist model -----
WHITELIST_ANCHOR = "    else:\n        raise ValueError(f\"Unsupported or untuned model: {model_name}\")"
WHITELIST_PATCH = """    elif 'Llama-3' in model_name or 'llama-3' in model_name:
        generation_config_file = './generation_configs/llama-3-instruct.json'
    else:
        raise ValueError(f"Unsupported or untuned model: {model_name}")"""

# ----- Patch B: chat_template = null -> giu template goc cua tokenizer -----
TEMPLATE_ANCHOR = """    chat_template_file = generation_config['chat_template']
    chat_template = open(chat_template_file).read()
    chat_template = chat_template.replace('    ', '').replace('\\n', '')
    toker.chat_template = chat_template"""
TEMPLATE_PATCH = """    chat_template_file = generation_config['chat_template']
    if chat_template_file:                      # None -> giu template built-in cua tokenizer
        chat_template = open(chat_template_file).read()
        chat_template = chat_template.replace('    ', '').replace('\\n', '')
        toker.chat_template = chat_template"""

PATCH_TARGETS = ["forward.py", "train.py", "generate.py",
                 "forward_with_soft.py", "train_unlikelihood.py"]

# stop_token_ids cua Llama-3: <|eot_id|> = 128009, <|end_of_text|> = 128001
LLAMA3_CONFIG = """{
    "chat_template": null,
    "stop_str": null,
    "stop_token_ids": [128001, 128009],
    "system_prompt": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\\n\\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
}
"""


def log(msg):
    print(f"[dro] {msg}", flush=True)


# ----- Dung ban da va -----
def build_work(smoke_n=None):
    if os.path.isdir(WORK):
        shutil.rmtree(WORK)
    shutil.copytree(REPO_CODE, WORK)

    n_patched = 0
    for fname in PATCH_TARGETS:
        path = os.path.join(WORK, fname)
        if not os.path.exists(path):
            continue
        src = open(path, encoding="utf-8").read()
        before = src
        if WHITELIST_ANCHOR in src:
            src = src.replace(WHITELIST_ANCHOR, WHITELIST_PATCH)
        if TEMPLATE_ANCHOR in src:
            src = src.replace(TEMPLATE_ANCHOR, TEMPLATE_PATCH)
        if src != before:
            open(path, "w", encoding="utf-8").write(src)
            n_patched += 1
    log(f"da va {n_patched}/{len(PATCH_TARGETS)} file")

    with open(os.path.join(WORK, "generation_configs", "llama-3-instruct.json"),
              "w", encoding="utf-8") as f:
        f.write(LLAMA3_CONFIG)

    # Smoke: cat bot query cho nhanh. Full thi giu nguyen 100+100 cua upstream.
    if smoke_n:
        for rel in ("data/custom.txt", "data_harmless/custom.txt"):
            p = os.path.join(WORK, rel)
            lines = [l for l in open(p, encoding="utf-8").read().split("\n") if l.strip()]
            open(p, "w", encoding="utf-8").write("\n".join(lines[:smoke_n]) + "\n")
        log(f"smoke: cat con {smoke_n} query moi ben")


def run(stage_name, argv, env):
    log(f"--- {stage_name}: {' '.join(argv)}")
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable] + argv, cwd=WORK, env=env)
    dt = time.perf_counter() - t0
    log(f"--- {stage_name}: exit={r.returncode} trong {dt:.1f}s")
    if r.returncode != 0:
        sys.exit(r.returncode)
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--full", action="store_true", help="dung du 100+100 query nhu upstream")
    ap.add_argument("--prompt-length", default="default", choices=["default", "short", "mistral"])
    ap.add_argument("--stage", default="all",
                    choices=["all", "build", "forward", "estimate", "train"])
    args = ap.parse_args()

    smoke_n = None if args.full else 20
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")   # MIG

    if args.stage in ("all", "build"):
        build_work(smoke_n)
    if args.stage == "build":
        return

    total = 0.0
    if args.stage in ("all", "forward"):
        total += run("forward", ["forward.py", "--pretrained_model_path", args.base], env)
    if args.stage in ("all", "estimate"):
        total += run("estimate", ["estimate.py", "--system_prompt_type", "all",
                                  "--config", "sampling",
                                  "--pretrained_model_path", args.base], env)
    if args.stage in ("all", "train"):
        total += run("train", ["train.py", "--system_prompt_type", "all",
                               "--prompt_length", args.prompt_length,
                               "--config", "sampling",
                               "--pretrained_model_path", args.base], env)

    name = args.base.rstrip("/").split("/")[-1]
    out = os.path.join(WORK, "trained_prompts", name,
                       f"type.all_length.{args.prompt_length}.safetensors")
    log(f"TONG {total:.1f}s")
    log(f"soft prompt -> {out}  ({'CO' if os.path.exists(out) else 'THIEU'})")
    if os.path.exists(out):
        log(f"chay thu:  DRO_SOFT_PROMPT={out} python method.py response --task harmbench --limit 3")


if __name__ == "__main__":
    main()
