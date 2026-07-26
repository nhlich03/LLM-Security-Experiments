"""
CAT / CAPO - retrain on our own base model (smoke-sized by default).

This does NOT reimplement anything: it writes the one config file upstream asks you to
create, then calls their Hydra entry point `repo/src/run_experiments.py`. Everything about
the algorithm stays theirs.

Upstream recipe (repo/config/adv_train_ul.yaml + the dataclass defaults in run_experiments.py):
  data      : adv_training_behaviors (HarmBench AT set) + ultrachat_200k utility,
              mixed 0.125 adv / 0.875 utility
  attack    : embedding-space, 10 iterations, sign optimiser lr=1e-4, eps=0.05
  peft      : LoRA r=64, alpha=16, target_modules="all-linear"
  quant     : bitsandbytes 4-bit (nf4)
  training  : 5 epochs, batch 8 x grad-accum 8, max_seq_length 256
  losses    : toward/away weight 0.5 each, utility weight 1.0, cutoffs -5.0 / 0.5
Reported: ~42 min for CAT and ~19 min for CAPO on a single A100.

Smoke mode (default) cuts it to a handful of steps with batch 1 so you can prove the whole
path works on one 40GB MIG before committing to a real run. Pass --full for the paper config.

NOTE: upstream imports `fcntl` -> POSIX only. Run this on the GPU server, not on Windows.

Usage:
  python train_smoke.py                                   # ~10 steps, CAT
  python train_smoke.py --algo ipo                        # CAPO
  python train_smoke.py --base meta-llama/Meta-Llama-3-8B-Instruct
  python train_smoke.py --full                            # paper-sized run
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "repo")

PATH_CFG_NAME = "local_run"
PATH_CFG = os.path.join(REPO, "config", "path", f"{PATH_CFG_NAME}.yaml")

# eps is per-model in the paper: 0.1 for Gemma / Phi-3-Mini, 0.05 for Mistral / Llama-2,
# 0.075 for Zephyr. Llama-3 is not in the paper table -> reuse the Llama-2 value.
EPS_BY_MODEL = {"gemma": 0.1, "phi": 0.1, "zephyr": 0.075, "mistral": 0.05, "llama": 0.05}


def pick_eps(base):
    low = base.lower()
    for k, v in EPS_BY_MODEL.items():
        if k in low:
            return v
    return 0.05


# ----- Llama-3 chat template: upstream never added one (Llama-3 is not in the paper) -----
# Their src/model_utils.py hardcodes a whitelist of chat formats and raises
#     NotImplementedError: Model Meta-Llama-3-8B-Instruct not supported
# for anything else, and get_model_name() raises ValueError before that. Llama-3 needs both.
# Formats below are Llama-3's official special tokens.
LLAMA3_TEMPLATE_BRANCH = '''    elif "llama-3" == model_name:
        found += 1
        first_user_msg = "<|start_header_id|>user<|end_header_id|>\\n\\n{instruction}<|eot_id|>"
        user_chat_template = "<|start_header_id|>user<|end_header_id|>\\n\\n{instruction}<|eot_id|>"
        response_key = "<|start_header_id|>assistant<|end_header_id|>\\n\\n"
        response_template = response_key + "{target}<|eot_id|>"
'''
MODEL_UTILS_PATCHES = [
    # add the template branch just before the existing phi branch
    ('    elif "phi" in model_name:\n        found += 1\n'
     '        first_user_msg = "<|system|>',
     LLAMA3_TEMPLATE_BRANCH +
     '    elif "phi" in model_name:\n        found += 1\n'
     '        first_user_msg = "<|system|>'),
    # and make get_model_name() recognise it (BEFORE "llama-2", which would not match anyway,
    # but keep it above the generic branches for clarity)
    ('    elif "llama-2" in model_name:\n        return "llama-2"',
     '    elif "llama-3" in model_name or "llama3" in model_name:\n        return "llama-3"\n'
     '    elif "llama-2" in model_name:\n        return "llama-2"'),
]


# Llama-3 has NO unk_token, so upstream's `tokenizer.pad_token = tokenizer.unk_token`
# leaves pad_token as None and the collator dies at the first step with
#     ValueError: Asking to pad but the tokenizer does not have a padding token.
# Same line appears twice (ul and dpo trainer branches) - replace_all.
ADV_TRAIN_PATCHES = [
    ("        tokenizer.pad_token = tokenizer.unk_token",
     "        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token"),
]


def patch_adversarial_training():
    path = os.path.join(REPO, "src", "adversarial_training.py")
    with open(path, encoding="utf-8") as f:
        code = f.read()
    if "unk_token or tokenizer.eos_token" in code:
        print("[patch] adversarial_training.py already has the pad_token fallback")
        return
    backup = path + ".orig"
    if not os.path.exists(backup):
        with open(backup, "w", encoding="utf-8") as f:
            f.write(code)
    for old, new in ADV_TRAIN_PATCHES:
        if old not in code:
            raise SystemExit(f"[patch] anchor not found in adversarial_training.py: {old[:60]}")
        code = code.replace(old, new)          # both branches
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"[patch] pad_token falls back to eos_token in src/adversarial_training.py "
          f"(backup: {backup})")


def patch_model_utils():
    """Add a Llama-3 branch to repo/src/model_utils.py (idempotent, keeps a .orig backup).

    Unlike Circuit Breakers we cannot launch a patched COPY here: model_utils is imported as
    a module by run_experiments.py, and the script's own directory always wins on sys.path.
    So this edits in place - but only this one file, only additively, and the original is
    saved next to it as model_utils.py.orig.
    """
    path = os.path.join(REPO, "src", "model_utils.py")
    with open(path, encoding="utf-8") as f:
        code = f.read()
    if '"llama-3" == model_name' in code:
        print("[patch] model_utils.py already has the Llama-3 branch")
        return
    backup = path + ".orig"
    if not os.path.exists(backup):
        with open(backup, "w", encoding="utf-8") as f:
            f.write(code)
    for old, new in MODEL_UTILS_PATCHES:
        if old not in code:
            raise SystemExit(f"[patch] anchor not found in model_utils.py:\n{old[:80]}...")
        code = code.replace(old, new, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"[patch] added Llama-3 chat template to src/model_utils.py (backup: {backup})")


# `path.model_name` is NOT the HF id - it is upstream's internal KEY that selects the chat
# template (see example_path.yaml: model_path=zephyr-7b-beta but model_name=mistral).
# Passing the HF id gives: NotImplementedError: Model Meta-Llama-3-8B-Instruct not supported
def upstream_model_key(base):
    low = base.split("/")[-1].lower()
    for needle, key in [("llama-3", "llama-3"), ("llama3", "llama-3"),
                        ("llama-2", "llama-2"), ("phi-3", "phi-3"),
                        ("zephyr", "mistral"), ("mistral", "mistral-instruct"),
                        ("gemma-1.1-2b-it", "gemma-1.1-2b-it"), ("gemma-2b-it", "gemma-2b-it"),
                        ("gemma-1.1-7b-it", "gemma-1.1-7b-it"), ("gemma-2b", "gemma-2b")]:
        if needle in low:
            return key
    raise SystemExit(f"[cfg] no upstream chat-template key for {base}; "
                     f"add one in src/model_utils.py first")


def write_path_config(base, out_root):
    """Upstream: 'Create a config in config/path (see example_path.yaml)'."""
    os.makedirs(os.path.dirname(PATH_CFG), exist_ok=True)
    key = upstream_model_key(base)
    print(f"[cfg] chat-template key for {base} -> {key}")
    body = f"""# @package _global_
# Generated by train_smoke.py - do not hand-edit, it is overwritten on every run.
path:
  experiments_path: {out_root}/experiments_db/
  logging_path: {out_root}/
  checkpoint_path: {out_root}/checkpoints/
  model_path: {base}
  model_name: {key}
"""
    with open(PATH_CFG, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"[cfg] wrote {PATH_CFG}\n{body}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get(
        "LOCAL_TARGET_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"))
    ap.add_argument("--algo", choices=["ul", "ipo"], default="ul",
                    help="ul = CAT (needs utility data) | ipo = CAPO (no utility data)")
    ap.add_argument("--full", action="store_true", help="paper-sized run instead of smoke")
    ap.add_argument("--steps", type=int, default=10, help="smoke: max_steps")
    ap.add_argument("--out", default=os.path.join(HERE, "train_out"))
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    if "llama-3" in a.base.lower() or "llama3" in a.base.lower():
        patch_model_utils()
        patch_adversarial_training()
    write_path_config(a.base, os.path.abspath(a.out).replace("\\", "/"))

    cmd = [sys.executable, "src/run_experiments.py",
           f"--config-name=adv_train_{a.algo}",
           f"path={PATH_CFG_NAME}",
           f"adversarial.eps={pick_eps(a.base)}"]

    if not a.full:
        # Smoke: few steps, batch 1, fewer attack iterations - just prove the path runs.
        cmd += [f"training.max_steps={a.steps}",
                "training.num_train_epochs=1",
                "training.per_device_train_batch_size=1",
                "training.gradient_accumulation_steps=1",
                "training.save_steps=1000000",
                "adversarial.iters=2"]
    else:
        # Paper config, but batch has to come down for a single 40GB card; keep the
        # effective batch (8x8=64) by moving it into gradient accumulation.
        cmd += ["training.per_device_train_batch_size=2",
                "training.gradient_accumulation_steps=32"]

    # Llama-3 prefers bf16; the shipped yaml sets fp16 (it targets older cards).
    if "llama-3" in a.base.lower() or "llama3" in a.base.lower():
        cmd += ["training.fp16=False", "training.bf16=True"]

    # The legacy venv pins datasets==2.17.1 while the main venv has 5.x, and they share
    # ~/.cache/huggingface. The old reader chokes on metadata the new one wrote:
    #     TypeError: must be called with a dataclass type or instance
    #     (datasets/features/features.py -> generate_from_dict -> fields(class_type))
    # Give this env its own DATASETS cache; the model hub cache stays shared, so no model
    # gets re-downloaded.
    env = dict(os.environ)
    env.setdefault("HF_DATASETS_CACHE", os.path.join(HERE, ".hf_datasets_cache"))

    print("[run]", " ".join(cmd), f"\n[cwd] {REPO}"
          f"\n[env] HF_DATASETS_CACHE={env['HF_DATASETS_CACHE']}\n")
    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=REPO, env=env)
    print(f"\n[done] exit={rc} | {time.perf_counter() - t0:.1f}s")
    if rc == 0:
        print(f"Checkpoint under {a.out}/checkpoints/ -> point method.py at it:")
        print(f"  CAT_MODEL={a.base} CAT_LORA=<ckpt dir> python method.py response --limit 5")
    sys.exit(rc)


if __name__ == "__main__":
    main()
