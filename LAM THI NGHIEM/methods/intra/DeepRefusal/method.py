"""
DeepRefusal - Beyond Surface Alignment (intra, Findings EMNLP 2025) - GENERATION entry point.

Repo: https://github.com/YuanBoXie/DeepRefusal (clone trong repo/)

Mechanism: normal alignment only teaches the model to refuse at the FIRST token, so any
attack that gets past that token wins. DeepRefusal deliberately ABLATES the refusal
direction across many layers during training and forces the model to recover the refusal
anyway, with prefill augmentation (PAA, p=0.5) on top. The result is a refusal behaviour
that is rebuilt at every depth instead of sitting only at the surface.

Reported training: ~45 min, 1 epoch, 1x A100-80GB, LoRA alpha=16 r=16, batch 16.
Data (all public): 2,000 harmful from CircuitBreaker + 4,000 benign from UltraChat
+ 500 from XSTest/Or-bench.

INTRA method -> defense is baked into the weights, zero inference overhead, hook =
local_generate. The released checkpoint is a FULL merged model (4 shards), not a LoRA.

WARNING (must be stated in the report): the training set contains 500 XSTest prompts,
which is exactly our over-refusal benchmark -> our XSTest number is optimistically
biased. Same issue as Circuit Breakers. The authors also report 28.5% over-refusal at
p=0.5 themselves, so expect this method to look bad on that axis.

USEFUL: Table 1 of the paper already reports LAT / CAT / CircuitBreaker on the same
Llama-3-8B, so it doubles as a cross-check for our own pipeline.

Run (needs GPU):
  python method.py response --task harmbench
  python method.py response --task xstest
  python method.py judge    --task xstest
  python method.py judge    --task harmbench
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from core.runner import run_method                            # noqa: E402
from core.local_client import local_generate, resolve_model   # noqa: E402

# ----- Released checkpoint: full merged Llama-3-8B-Instruct, khong phai LoRA -----
CKPT = resolve_model("skysys00/Meta-Llama-3-8B-Instruct-DeepRefusal", env_var="DR_MODEL")
LORA = os.environ.get("DR_LORA") or None      # dat khi dung adapter tu train


if __name__ == "__main__":
    run_method(
        name="DeepRefusal",
        slug="deeprefusal",
        defense_type="intra",
        backend="local",
        model=CKPT,
        lora=LORA,
        generate=local_generate,
        out_dir=os.path.join(HERE, "outputs"),
    )
