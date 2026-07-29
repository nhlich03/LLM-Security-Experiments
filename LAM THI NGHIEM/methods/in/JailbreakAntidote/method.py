"""Jailbreak Antidote (ICLR 2025, arXiv:2410.02298) - in-processing, training-free.

At every decoder layer during generation, shift the residual stream by a scaled sparse
safety direction (paper Eq. 5):
    h'_l = h_l + alpha * (d_safe_l (+) mask_l)
A single scalar `alpha` slides the model along the safety<->utility axis; positive = safer.
No extra tokens, no weight change -> in-processing. The masked unit directions are produced
once by calibrate.py.

Config (declare in report):
  ANTIDOTE_ALPHA  strength; paper Table A.1 gives Llama-3-8B-Instruct range [-0.6, 0.6],
                  positive = safer. Default 0.4 (mid-high safety operating point).
  ANTIDOTE_LAYERS 'all' (default) or comma list. Paper leaves the layer set L unspecified
                  (l in L subset of {1..L}); we default to ALL layers and declare it.

Prereq: PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync python calibrate.py
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

BASE = resolve_model("NousResearch/Meta-Llama-3-8B-Instruct", env_var="ANTIDOTE_BASE")
ALPHA = float(os.environ.get("ANTIDOTE_ALPHA", "0.4"))
VEC = os.path.join(HERE, "vectors", "llama3", "antidote_directions.pt")


class AntidoteClient(LocalClient):
    """LocalClient with a forward hook on each decoder layer that adds the scaled sparse
    safety direction to the hidden state. `.chat()` is inherited unchanged."""

    def __init__(self, model_id, alpha=ALPHA, **kw):
        super().__init__(model_id, **kw)
        import torch

        if not os.path.exists(VEC):
            sys.exit(f"[Antidote] missing {VEC}\n  -> run: python calibrate.py")
        dirs = torch.load(VEC, map_location="cpu")             # [L, D], masked unit dirs
        self.dirs = dirs.to(self.device).to(self.model.dtype)
        self.alpha = alpha

        n_layers = self.model.config.num_hidden_layers
        env_layers = os.environ.get("ANTIDOTE_LAYERS", "all")
        self.layers = (list(range(n_layers)) if env_layers == "all"
                       else [int(x) for x in env_layers.split(",") if x.strip() != ""])

        self._handles = []
        base = self.model.model if hasattr(self.model, "model") else self.model
        for l in self.layers:
            self._handles.append(base.layers[l].register_forward_hook(self._make_hook(l)))
        print(f"[Antidote] alpha={self.alpha} layers={len(self.layers)}/{n_layers} "
              f"|dir[0]|={self.dirs[0].norm():.3f}")

    def _make_hook(self, l):
        vec = self.dirs[l]
        a = self.alpha

        def hook(module, inp, out):
            # transformers v5 LlamaDecoderLayer.forward may return a bare tensor or a tuple.
            if isinstance(out, tuple):
                return (out[0] + a * vec, *out[1:])
            return out + a * vec

        return hook

    def free(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        super().free()


def _factory(model, temperature, max_tokens):
    return AntidoteClient(BASE, temperature=temperature, max_tokens=max_tokens)


def _generate(client, raw, meter):
    return local_generate(client, raw, meter, label="antidote_decode")


if __name__ == "__main__":
    run_method(
        name="JailbreakAntidote",
        slug="jailbreak_antidote",
        defense_type="in",
        model=BASE,
        client_factory=_factory,
        generate=_generate,
        out_dir=os.path.join(HERE, "outputs"),
    )
