"""
CAT / CAPO - Efficient Adversarial Training in LLMs with Continuous Attacks (intra, NeurIPS 2024 Spotlight)
             - GENERATION entry point.

Paper: https://arxiv.org/abs/2405.15589 | repo: sophie-xhonneux/Continuous-AdvTrain
Mechanism: adversarial training where the attack is computed in the CONTINUOUS EMBEDDING
space instead of searching discrete tokens - ~299x cheaper than R2D2 over a full run.
  - CAT  (C-AdvUL): toward/away unlikelihood loss under a continuous embedding attack,
                    plus a utility loss on UltraChat200k to keep capability.
  - CAPO (C-AdvIPO): adversarial IPO; the implicit KL to the reference model stops the
                    model collapsing into "refuse everything", so no utility data needed.

This is an INTRA method: the whole defense lives in the weights. At inference there is
NO overhead and NO extra call - it is literally `no_defense` with a different checkpoint.
That is why the hook is just `local_generate`.

Default = the authors' released checkpoint (no training needed). To train your own, see
`train_smoke.py` in this folder.

Run (needs GPU; no Groq key for the target):
  python method.py response --task all
  python method.py judge    --task xstest        # over-refusal (API judge)
  python method.py judge    --task harmbench     # ASR (classifier GPU)
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))   # LAM THI NGHIEM
sys.path.insert(0, ROOT)                                        # for `core`

from core.runner import run_method                    # noqa: E402
from core.local_client import local_generate, resolve_model   # noqa: E402

# ----- Released checkpoint (authors trained Llama-3-8B-Instruct with CAT and published it,
# ----- even though the paper table only covers Gemma/Phi-3/Mistral/Zephyr/Llama-2) -----
CKPT = resolve_model("ContinuousAT/Llama3-8B-IT-CAT", env_var="CAT_MODEL")

# Other released checkpoints, in case the base model decision changes:
#   ContinuousAT/Llama-2-7B-CAT  |  ContinuousAT/Zephyr-CAT (LoRA)
#   ContinuousAT/Phi-CAT         |  ContinuousAT/Phi-CAPO
# A LoRA-only checkpoint needs LORA=<repo> below plus its base model in CKPT.
LORA = os.environ.get("CAT_LORA") or None

# The CAT checkpoint ships WEIGHTS ONLY - no tokenizer files at all (verified: the repo has
# just config.json, generation_config.json and the safetensors shards). So the tokenizer has
# to come from the base model it was trained from.
# Default is the NousResearch MIRROR, not meta-llama/*: the official repo is gated and our
# GPU server has no HF token. The mirror is a straight copy of the same tokenizer.
TOKENIZER = os.environ.get("CAT_TOKENIZER", "NousResearch/Meta-Llama-3-8B-Instruct")


if __name__ == "__main__":
    run_method(
        name="CAT",
        slug="cat",
        defense_type="intra",
        backend="local",
        model=CKPT,
        lora=LORA,
        local_kwargs={"tokenizer_id": TOKENIZER},
        generate=local_generate,       # weights already carry the defense -> 1 plain call
        out_dir=os.path.join(HERE, "outputs"),
    )
