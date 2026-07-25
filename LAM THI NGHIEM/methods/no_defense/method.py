"""
no_defense (baseline) - GENERATION entry point.

The reference / mốc so sánh: gọi thẳng target, KHÔNG áp cơ chế phòng thủ nào
(transform_prompt = identity). Dùng chung harness core.runner như mọi method,
nên baseline được đo cost + chấm over-refusal y hệt cách các method khác được đo
-> con số so sánh công bằng tuyệt đối.

Thay cho 2 notebook Kaggle gốc trong reference/ (harmbench + xstest); core.runner
lo cả 2 task. Chạy local (chỉ cần Groq API, không GPU).

Run (2 stage, moi stage co --task {all|harmbench|xstest|justeval}):
  conda activate sage
  python method.py response --task all       # sinh response 300 + 250
  python method.py judge    --task xstest      # baseline over-refusal (API)
  python method.py judge    --task harmbench   # baseline ASR (classifier GPU)
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))   # LAM THI NGHIEM (no_defense is 1 level under methods/)
sys.path.insert(0, ROOT)                                 # for `core`

from core.runner import run_method            # noqa: E402

if __name__ == "__main__":
    run_method(
        name="no_defense",
        slug="no_defense",
        defense_type="none",
        model="llama-3.1-8b-instant",
        transform_prompt=lambda prompt: prompt,   # identity = no defense
        out_dir=os.path.join(HERE, "outputs"),
    )
