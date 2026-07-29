"""LED (Layer-specific Editing, Findings EMNLP 2024) - intra-processing (permanent weight edit).

Generation entry point: the defense is baked into the edited layer weights saved by led_train.py,
so we just load the edited model and generate normally. Cost = local seconds + tokens.

Prereq:
  PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync python led_build_data.py
  PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync python led_train.py
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

MODEL = os.environ.get("LED_MODEL", os.path.join(HERE, "led_out"))


def _factory(model, temperature, max_tokens):
    return LocalClient(MODEL, temperature=temperature, max_tokens=max_tokens)


def _generate(client, raw, meter):
    return local_generate(client, raw, meter, label="led_decode")


if __name__ == "__main__":
    run_method(
        name="LED",
        slug="led",
        defense_type="intra",
        model=MODEL,
        client_factory=_factory,
        generate=_generate,
        out_dir=os.path.join(HERE, "outputs"),
    )
