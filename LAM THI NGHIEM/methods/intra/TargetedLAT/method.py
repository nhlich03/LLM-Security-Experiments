"""
Targeted LAT - Targeted Latent Adversarial Training (intra, arXiv 2407.15549)
             - GENERATION entry point.

Repo: https://github.com/aengusl/latent-adversarial-training
Checkpoints: HF org https://huggingface.co/LLM-LAT

Mechanism: adversarial training where the perturbation lives in the LATENT space
(residual-stream activations between layers) instead of discrete tokens. Same spirit as
CAT - dodge the expensive discrete search of GCG - but the perturbation sits deeper:
  R2D2   -> perturb discrete tokens      (most expensive)
  CAT    -> perturb the INPUT EMBEDDING
  LAT    -> perturb ACTIVATIONS mid-network
Having all three lets the survey cover the whole adversarial-training axis.

Reported cost: RT-EAT-LAT uses ~36x fewer GPU hours than R2D2 (no absolute number given).

INTRA method -> defense is in the weights, zero inference overhead, hook = local_generate.

Checkpoint note: `LLM-LAT/robust-llama3-8b-instruct` is the flagship (full merged weights,
7 shards, ~900 downloads) and is the one the DeepRefusal paper benchmarks as "LAT".
The org also hosts `llama3-8b-instruct-{lat,rt}-jailbreak-robust{1,2,3}` - the `lat-*`
ones are LoRA adapters with no downloads, treat them as supplementary.

Repo caveat: upstream is ALL NOTEBOOKS, no CLI - so `repo/` is not vendored here; we only
consume the released weights. Retraining would mean lifting code out of the notebooks
(see train_smoke.py).

Run (needs GPU):
  python method.py response --task harmbench
  python method.py response --task xstest
  python method.py judge    --task xstest
  python method.py judge    --task harmbench

  LAT_MODEL=LLM-LAT/llama3-8b-instruct-rt-jailbreak-robust1 python method.py response ...
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from core.runner import run_method                            # noqa: E402
from core.local_client import local_generate, resolve_model   # noqa: E402

# ----- Released checkpoint: full merged weights (khong phai LoRA) -----
CKPT = resolve_model("LLM-LAT/robust-llama3-8b-instruct", env_var="LAT_MODEL")
LORA = os.environ.get("LAT_LORA") or None     # dat khi dung adapter lat-jailbreak-robust*


if __name__ == "__main__":
    run_method(
        name="TargetedLAT",
        slug="targeted_lat",
        defense_type="intra",
        backend="local",
        model=CKPT,
        lora=LORA,
        generate=local_generate,
        out_dir=os.path.join(HERE, "outputs"),
    )
