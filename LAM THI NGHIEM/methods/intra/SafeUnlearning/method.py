"""Safe Unlearning (arXiv:2407.02855) - intra-processing (permanent weight edit).

Generation entry point: the defense is baked into a LoRA adapter trained by su_train.py, so we
just load base + adapter and generate normally (same as the other intra methods). Cost = local
seconds + tokens.

Prereq: PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync python su_train.py
Run:
  python method.py response --task all
  python method.py judge    --task harmbench
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from core.runner import run_method                                    # noqa: E402
from core.local_client import LocalClient, resolve_model, local_generate  # noqa: E402

BASE = resolve_model("NousResearch/Meta-Llama-3-8B-Instruct", env_var="SU_BASE")
LORA = os.environ.get("SU_LORA", os.path.join(HERE, "train_out", "lora_llama_safeunlearning"))


def _factory(model, temperature, max_tokens):
    return LocalClient(BASE, lora=LORA, temperature=temperature, max_tokens=max_tokens)


def _generate(client, raw, meter):
    return local_generate(client, raw, meter, label="safe_unlearning_decode")


if __name__ == "__main__":
    run_method(
        name="SafeUnlearning",
        slug="safe_unlearning",
        defense_type="intra",
        model=BASE,
        client_factory=_factory,
        generate=_generate,
        out_dir=os.path.join(HERE, "outputs"),
    )
