"""
SAGE (pre-processing, P1) - GENERATION entry point.

The whole method is: wrap the prompt with SAGE's instruction (imported verbatim
from the cloned repo) and let the shared harness do the rest.

Run (2 stage, moi stage co --task {all|harmbench|xstest|justeval}):
  conda activate sage
  python method.py response --task all        # sinh response 300 + 250
  python method.py judge    --task xstest       # cham over-refusal (API)
  python method.py judge    --task harmbench    # cham ASR (classifier GPU)
  python method.py response --task harmbench --limit 5   # smoke test
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))   # LAM THI NGHIEM
sys.path.insert(0, ROOT)                                       # for `core`
sys.path.insert(0, os.path.join(HERE, "repo"))                 # for defense_prompts

from core.runner import run_method            # noqa: E402
from defense_prompts import make_sage_prompt  # noqa: E402  (SAGE verbatim)

if __name__ == "__main__":
    run_method(
        name="SAGE",
        slug="sage",
        defense_type="pre",
        model="llama-3.1-8b-instant",
        transform_prompt=make_sage_prompt,
        out_dir=os.path.join(HERE, "outputs"),
    )
