"""
Circuit Breakers - retrain on our own base model (smoke-sized by default).

Wraps upstream's own launcher `repo/scripts/lorra_circuit_breaker_llama3_8b.sh`, which calls
`repo/src/lorra_circuit_breaker.py` through accelerate. Nothing is reimplemented here; we only
override the sizes so it fits one 40GB MIG and finishes in minutes.

Upstream Llama-3-8B config (from that script):
  target_layers 10,20 | transform_layers -1 | lorra_alpha 10 (Mistral uses 5)
  LoRA r=16, alpha=16, dropout=0.05
  max_steps 150 | batch 16 | grad-accum 1 | lr 1e-4 | bf16 | --use_refusal_retain
Reported: ~20 minutes on 1x A100-80GB.

Training data ships with the repo, nothing to download:
  repo/data/circuit_breakers_train.json   (the "circuit breaker set")
  repo/data/circuit_breakers_val.json
  retain side pulls UltraChat + XSTest

WARNING - the retain set includes XSTest, which is our over-refusal benchmark. Any model
trained here (and the released checkpoint too) has seen XSTest prompts. Say so in the report.

Usage:
  python train_smoke.py                       # ~10 steps, batch 1
  python train_smoke.py --full                # upstream's 150 steps
  python train_smoke.py --base mistralai/Mistral-7B-Instruct-v0.2 --lorra-alpha 5
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "repo")


ENTRY = os.path.join("src", "lorra_circuit_breaker.py")
ENTRY_COMPAT = os.path.join("src", "lorra_circuit_breaker_compat.py")

# transformers used to re-export the deepspeed integration at top level; that alias is gone
# in v5 (verified on 5.14.1):
#   ImportError: cannot import name 'deepspeed' from 'transformers'
# It only supplies `deepspeed.is_deepspeed_zero3_enabled()`, which still exists under
# transformers.integrations. Rather than editing the upstream clone (repo/ stays pristine),
# write a patched COPY next to it and launch that.
                            # (old source, new source, what it fixes)
V5_PATCHES = [
    ("from transformers import Trainer, deepspeed, AutoTokenizer, "
     "AutoModelForCausalLM, AutoConfig",
     "from transformers import Trainer, AutoTokenizer, AutoModelForCausalLM, AutoConfig\n"
     "try:                                            # transformers < 4.40\n"
     "    from transformers import deepspeed\n"
     "except ImportError:                             # transformers >= 4.40, incl. v5\n"
     "    from transformers.integrations import deepspeed",
     "ImportError: cannot import name 'deepspeed' from 'transformers'"),

    ("len(training_args.fsdp) > 0",
     "bool(training_args.fsdp)",
     "TypeError: object of type 'NoneType' has no len()  (v5 defaults fsdp to None, v4 used [])"),

    # Only the CustomTrainer(...) call site - NOT the other `tokenizer=tokenizer` uses on
    # lines 268/318, which are upstream's own helpers and still take that keyword.
    ("model=model, tokenizer=tokenizer, args=training_args",
     "model=model, processing_class=tokenizer, args=training_args",
     "TypeError: Trainer.__init__() got an unexpected keyword argument 'tokenizer' "
     "(v5 renamed it to processing_class)"),

    # v5's training_step calls compute_loss(..., num_items_in_batch=N). Upstream's override
    # (line 310) has a fixed v4 signature, so it blows up on the very first step.
    ("        def compute_loss(self, model, inputs, return_outputs=False):",
     "        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):",
     "TypeError: CustomTrainer.compute_loss() got an unexpected keyword argument "
     "'num_items_in_batch' (v5 passes it; v4 did not)"),
]


def make_compat_entry():
    """Launch a PATCHED COPY of upstream's entry script so repo/ stays a pristine clone.

    Upstream targets transformers v4; the server runs v5.14.1. Each entry in V5_PATCHES is
    an incompatibility that was hit for real, in order, one run at a time.
    """
    with open(os.path.join(REPO, ENTRY), encoding="utf-8") as f:
        code = f.read()
    applied = []
    for old, new, why in V5_PATCHES:
        if old in code:
            code = code.replace(old, new)
            applied.append(why)
    if not applied:
        print(f"[compat] no patch needed -> using {ENTRY}")
        return ENTRY
    with open(os.path.join(REPO, ENTRY_COMPAT), "w", encoding="utf-8") as f:
        f.write(code)
    print(f"[compat] {len(applied)} patch -> {ENTRY_COMPAT}")
    for w in applied:
        print(f"[compat]   fixes: {w}")
    return ENTRY_COMPAT


def tf_major():
    import transformers
    return int(transformers.__version__.split(".")[0])


def eval_strategy_flags():
    """transformers v5 dropped `--overwrite_output_dir` entirely and renamed
    `--evaluation_strategy` to `--eval_strategy`. Passing the v4 names to v5 gives:
      ValueError: Some specified arguments are not used by the HfArgumentParser:
                  ['--overwrite_output_dir', '--evaluation_strategy', 'no']
    """
    if tf_major() >= 5:
        return ["--eval_strategy", "no"]
    return ["--overwrite_output_dir", "--evaluation_strategy", "no"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get(
        "LOCAL_TARGET_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"))
    ap.add_argument("--lorra-alpha", type=float, default=None,
                    help="upstream: 10 for Llama-3, 5 for Mistral (auto by model name)")
    ap.add_argument("--full", action="store_true", help="150 steps like upstream")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "train_out"))
    ap.add_argument("--accelerate", action="store_true",
                    help="launch through accelerate + DeepSpeed ZeRO-1 like upstream. "
                         "OFF by default: NCCL fails on our MIG slice "
                         "(ncclUnhandledCudaError), and a single-GPU LoRA run needs neither.")
    a = ap.parse_args()

    alpha = a.lorra_alpha
    if alpha is None:
        alpha = 5 if "mistral" in a.base.lower() else 10

    steps = 150 if a.full else a.steps
    # Upstream uses batch 16 on an 80GB card; halve it (and pay it back in accumulation)
    # so it fits 40GB.
    batch = a.batch or (4 if a.full else 1)
    accum = max(1, 16 // batch) if a.full else 1

    os.makedirs(a.out, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    entry = make_compat_entry()
    launcher = (["accelerate", "launch", "--config_file", "configs/accelerate_zero1.yaml",
                 "--num_processes", "1"] if a.accelerate else [sys.executable])

    cmd = [
        *launcher,
        entry,
        "--model_name_or_path", a.base,
        "--target_layers", "10,20",
        "--transform_layers", "-1",
        "--lorra_alpha", str(alpha),
        "--lora_r", "16",
        "--lora_alpha", "16",
        "--lora_dropout", "0.05",
        "--output_dir", os.path.abspath(a.out).replace("\\", "/"),
        "--max_steps", str(steps),
        "--bf16", "True",
        "--per_device_train_batch_size", str(batch),
        "--per_device_eval_batch_size", str(max(1, batch)),
        "--gradient_accumulation_steps", str(accum),
        "--use_refusal_retain",
        *eval_strategy_flags(),
        "--save_total_limit", "0",
        "--learning_rate", "1e-4",
        "--weight_decay", "0.",
        "--lr_scheduler_type", "constant",
        "--logging_steps", "1",
        "--gradient_checkpointing", "True",
    ]

    print("[run]", " ".join(cmd), f"\n[cwd] {REPO}\n")
    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=REPO, env=env)
    print(f"\n[done] exit={rc} | {time.perf_counter() - t0:.1f}s")
    if rc == 0:
        print("Point method.py at the adapter you just trained:")
        print(f"  CB_MODEL={a.base} CB_LORA={a.out} python method.py response --limit 5")
    sys.exit(rc)


if __name__ == "__main__":
    main()
