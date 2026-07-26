"""
no_defense_local (baseline for the LOCAL table) - GENERATION entry point.

`methods/no_defense/` is the baseline for the API table (Groq llama-3.1-8b-instant). Every
in / intra method runs on LOCAL weights instead, so it needs its own baseline on the SAME
local base model - otherwise the numbers compare a defense against a different model, not
against no defense.

It also gives us the denominator for local cost: overhead of SafeDecoding / JBShield is
`local_sec(method) / local_sec(no_defense_local)`, measured on the same GPU, same
max_tokens, same prompts.

Base model is still undecided, so it is read from LOCAL_TARGET_MODEL and defaults to
Meta-Llama-3-8B-Instruct (the base every intra checkpoint we have was built on).

Run (needs GPU):
  LOCAL_TARGET_MODEL=meta-llama/Meta-Llama-3-8B-Instruct python method.py response --task all
  python method.py judge --task xstest
  python method.py judge --task harmbench
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from core.runner import run_method                    # noqa: E402
from core.local_client import local_generate, resolve_model   # noqa: E402

# NousResearch mirror: meta-llama/* is gated and the server has no HF token.
BASE = resolve_model("NousResearch/Meta-Llama-3-8B-Instruct")


if __name__ == "__main__":
    run_method(
        name="no_defense_local",
        slug="no_defense_local",
        defense_type="none",
        backend="local",
        model=BASE,
        generate=local_generate,     # plain call, no defense - this is the reference point
        out_dir=os.path.join(HERE, "outputs"),
    )
