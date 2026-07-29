"""ReFAT (arXiv:2409.20089, ICLR 2025) - intra-processing (permanent weight edit via LoRA).

Generation entry point: the defense is baked into the LoRA adapter trained by refat_train.py
(refusal-feature adversarial training), so we load base + adapter and generate normally.

Prereq: PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync python refat_train.py
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

BASE = resolve_model("NousResearch/Meta-Llama-3-8B-Instruct", env_var="REFAT_BASE")
LORA = os.environ.get("REFAT_LORA", os.path.join(HERE, "train_out", "lora_llama_refat"))


def _factory(model, temperature, max_tokens):
    return LocalClient(BASE, lora=LORA, temperature=temperature, max_tokens=max_tokens)


def _generate(client, raw, meter):
    return local_generate(client, raw, meter, label="refat_decode")


if __name__ == "__main__":
    run_method(
        name="ReFAT",
        slug="refat",
        defense_type="intra",
        model=BASE,
        client_factory=_factory,
        generate=_generate,
        out_dir=os.path.join(HERE, "outputs"),
    )
