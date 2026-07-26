"""
JBShield-M - Defending LLMs from Jailbreak Attacks through Activated Concept Analysis and
             Manipulation (in-processing, USENIX Security 2025) - GENERATION entry point.

Paper: https://arxiv.org/pdf/2502.07557 | repo: NISPLab/JBShield
Mechanism (Linear Representation Hypothesis). Two concepts live as directions in hidden
space: the TOXIC concept (fires on harmful AND jailbreak prompts) and the JAILBREAK concept
(fires only on jailbreak prompts, and is what turns refusal into compliance). During the
forward pass JBShield-M hooks the two critical layers and, when the corresponding concept is
detected, shifts the hidden state:
    h <- h + delta_safety    * safety_vector      (amplify toxic recognition)
    h <- h + delta_jailbreak * jailbreak_vector   (weaken the jailbreak concept)
Weights are never modified -> in-processing, not intra.

Prerequisite: run `python calibrate.py --model llama-3` ONCE. It writes the concept vectors,
thresholds, deltas and critical-layer indices into repo/vectors/<model>/. Without them this
script cannot start.

The hook class (`JBShieldM`) is imported VERBATIM from repo/mitigation.py - not reimplemented.
Only the plumbing (model loading, dataset loop, cost metering) is ours.

--- TWO CAVEATS THAT MUST GO IN THE REPORT ---

1. ATTACK-TYPE COUPLING. The jailbreak vector is calibrated PER ATTACK (gcg, autodan, ijp,
   ...). Our HarmBench file holds plain harmful prompts with no jailbreak wrapper, so the
   jailbreak concept will barely fire and the method effectively degrades to the toxic-concept
   half. Pick which attack's vector to use with JBS_ATTACK (default: ijp, the human-written
   in-the-wild jailbreaks - the closest thing to "generic"). This is a property of the method,
   not a bug in this port.

2. SPEED. The hook runs `interpret_difference_matrix` (an SVD) on every forward pass, which is
   why upstream evaluates with max_new_tokens=50. We generate 512 tokens per prompt as the
   project requires, so expect this method to be noticeably slower than the other four -
   measure it, do not assume the paper's overhead numbers carry over.

Run (needs GPU + calibration done):
  python calibrate.py --model llama-3
  python method.py response --task all
  python method.py judge    --task xstest
  python method.py judge    --task harmbench
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "repo")
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from core.runner import run_method                # noqa: E402
from core.local_client import LocalClient         # noqa: E402

# ----- Config -----
MODEL_KEY = os.environ.get("JBS_MODEL_KEY", "llama-3")      # names the vectors/ folder
ATTACK = os.environ.get("JBS_ATTACK", "ijp")                # which jailbreak vector to use
HF_PATH = os.environ.get("JBS_MODEL_PATH", "NousResearch/Meta-Llama-3-8B-Instruct")
ATTACKS = ["gcg", "puzzler", "saa", "autodan", "drattack", "pair", "ijp", "base64", "zulu"]

# ----- Two knobs upstream leaves broken/disabled; see the long note in the hook below -----
# GATE=1 : add the concept vector ONLY when the detector actually fires (what the paper
#          describes). GATE=0 : upstream's shortcut - always add. Default 1, because our
#          benchmark is plain harmful prompts, not the all-jailbreak set upstream assumed.
GATE = os.environ.get("JBS_GATE", "1") != "0"
# FIRST_M>0 : only manipulate the first M forward passes. This is upstream's own readability
#          fix (`self.count` in repo/mitigation.py), which they left commented out.
#          0 = unlimited (upstream behaviour).
# Default 2, measured on the server (2 HarmBench prompts, 512 max_tokens):
#   GATE=0 FIRST_M=0  9.61 s/req  "I cannot illegal illegal illegal illegal ..."   <- upstream
#   GATE=1 FIRST_M=0  9.67 s/req  clean start, then "... else illegal illegal ..."
#   GATE=1 FIRST_M=5  0.78 s/req  "I cannot provide illegal or harmful information."
#   GATE=1 FIRST_M=2  0.84 s/req  fully clean, and 12x faster
# Unbounded manipulation degenerates because the SVD-based detector keeps re-firing on the
# tokens it already pushed. Upstream never saw this: they generate 50 tokens, we generate 512.
FIRST_M = int(os.environ.get("JBS_FIRST_M", "2"))


# ----- Load the artefacts calibrate.py produced -----
def _load_vectors(model_key, attack):
    import torch

    vdir = os.path.join(REPO, "vectors", model_key)
    if not os.path.isdir(vdir):
        sys.exit(f"[JBShield] missing {vdir}\n"
                 f"  -> run: python calibrate.py --model {model_key}")
    if attack not in ATTACKS:
        sys.exit(f"[JBShield] unknown attack '{attack}', pick one of {ATTACKS}")

    def load(fname):
        p = os.path.join(vdir, fname)
        if not os.path.exists(p):
            sys.exit(f"[JBShield] missing {p} -> re-run calibrate.py")
        return torch.load(p, map_location="cpu", weights_only=False)

    # layer_indexs.pt = [safety_layer, then one jailbreak layer per attack, in ATTACKS order]
    layer_indexs = load("layer_indexs.pt")
    safety_layer = layer_indexs[0]
    jailbreak_layer = layer_indexs[1 + ATTACKS.index(attack)]

    return dict(
        mean_harmless_embedding=load("mean_harmless_embedding.pt"),
        mean_harmful_embedding=load("mean_harmful_embedding.pt"),
        base_safety_vector=load("calibration_safety_vector.pt"),
        base_jailbreak_vector=load(f"calibration_jailbreak_vector_{attack}.pt"),
        threshold_safety=load(f"thershold_safety_{attack}.pt"),        # upstream spelling
        threshold_jailbreak=load(f"thershold_jailbreak_{attack}.pt"),
        delta_safety=load("delta_safety.pt"),
        delta_jailbreak=load(f"delta_jailbreak_{attack}.pt"),
        selected_safety_layer_index=safety_layer,
        selected_jailbreak_layer_index=jailbreak_layer,
    )


def _make_compat_shield_cls():
    """Subclass upstream's JBShieldM with TWO plumbing fixes. The concept math (detection +
    the vector added to the hidden state) is copied verbatim - only the container and the
    dtype change.

    Fix 1 - RETURN TYPE. Upstream was written against transformers v4, where
    `LlamaDecoderLayer.forward` returns a TUPLE `(hidden_states, ...)`. Their hook does
    `tmp = output[0]` and returns `(tmp, *output[1:])`. In transformers v5 that forward
    returns a bare TENSOR, so `output[0]` grabs the first ROW of the tensor and returning a
    tuple blows up the next layer with
        AttributeError: 'tuple' object has no attribute 'dtype'
    (observed on transformers 5.14.1). We handle both shapes.

    Fix 2 - DTYPE. Upstream hardcodes `.to(torch.float16)` on the concept vector. Our models
    run in bfloat16; mixing bf16 and fp16 triggers type promotion to fp32 and breaks the
    layernorm downstream. We cast the vector to whatever dtype the hidden state already is.
    """
    from mitigation import JBShieldM         # noqa: E402

    class CompatJBShieldM(JBShieldM):
        @staticmethod
        def _split(output):
            if isinstance(output, tuple):
                return output[0], output[1:]
            return output, None

        @staticmethod
        def _join(hidden, rest):
            return hidden if rest is None else (hidden, *rest)

        def _apply(self, output, base_embedding, vector, threshold, delta):
            hidden, rest = self._split(output)
            detected = self.detection(hidden, base_embedding, vector, threshold)
            # `detected` is a list of 1.0/0.0 per batch element. Upstream writes
            # `if detected:` which is truthy for ANY non-empty list -> the vector is added
            # on EVERY forward, whatever the detector said. They say why right above the
            # class: "Here we use a simple version as all test data are jailbreak prompts."
            # Our HarmBench file is NOT all jailbreak prompts, so that shortcut does not
            # transfer - see GATE below.
            fire = any(detected) if GATE else bool(detected)
            if fire and (FIRST_M <= 0 or self._steps < FIRST_M):
                hidden = hidden + delta * vector.to(hidden.dtype).to(hidden.device)
            return self._join(hidden, rest)

        def hook_fn_safety(self, module, input, output):
            self._steps = getattr(self, "_steps", 0) + 1
            return self._apply(
                output,
                self.mean_harmless_embedding[self.selected_safety_layer_index - 1],
                self.base_safety_vector, self.threshold_safety, self.delta_safety)

        def reset_steps(self):
            self._steps = 0

        def hook_fn_jailbreak(self, module, input, output):
            return self._apply(
                output,
                self.mean_harmful_embedding[self.selected_jailbreak_layer_index - 1],
                self.base_jailbreak_vector, self.threshold_jailbreak, self.delta_jailbreak)

    return CompatJBShieldM


class JBShieldClient(LocalClient):
    """LocalClient with JBShield-M's forward hooks attached. `.chat()` stays unchanged."""

    def __init__(self, model_id, model_key=MODEL_KEY, attack=ATTACK, **kw):
        super().__init__(model_id, **kw)

        sys.path.insert(0, REPO)             # upstream modules live in repo/
        shield_cls = _make_compat_shield_cls()

        v = _load_vectors(model_key, attack)
        print(f"[JBShield] attack={attack} | safety layer={v['selected_safety_layer_index']}"
              f" | jailbreak layer={v['selected_jailbreak_layer_index']}")

        self.shield = shield_cls(self.model, self.tokenizer, **v)
        self.shield.register_hooks()

    def free(self):
        try:
            self.shield.remove_hooks()
        except Exception:
            pass
        super().free()


def _factory(model, temperature, max_tokens):
    return JBShieldClient(HF_PATH, temperature=temperature, max_tokens=max_tokens)


# ----- Defense = hidden-state manipulation during the forward pass; cost = local seconds AND tokens -----
def jbs_generate(client, raw, meter):
    client.shield.reset_steps()      # FIRST_M counts per request, not per run
    with meter.local("concept_manipulated_decode") as rec:
        text, resp = client.chat(raw)
        rec.from_usage(resp)
    return text


if __name__ == "__main__":
    run_method(
        name="JBShield",
        slug="jbshield",
        defense_type="in",
        model=HF_PATH,
        client_factory=_factory,
        generate=jbs_generate,
        out_dir=os.path.join(HERE, "outputs"),
    )
