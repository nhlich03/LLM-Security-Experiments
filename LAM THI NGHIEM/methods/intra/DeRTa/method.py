"""
DeRTa - Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal
        Training (intra, ACL 2025 main) - GENERATION entry point.

Paper: https://arxiv.org/abs/2407.09121 | repo: RobustNLP/DeRTa
Mechanism: standard alignment only ever teaches the model to refuse at token 0, so once an
attacker forces a compliant prefix the model happily continues. DeRTa decouples the two:
  1. MLE with Harmful Response Prefix - training samples prepend a harmful prefix and the
     target is still a refusal, so the model learns to bail out MID-generation.
  2. RTO (Reinforced Transition Optimization) - reinforces the transition into refusal at
     every position of the harmful continuation, not just the first one.
Result: strong against prefilling attacks, which is exactly the hole shallow alignment leaves.

INTRA method -> defense is in the weights, no inference overhead, hook = local_generate.

Default = the checkpoint the authors mark "Recommend": a LoRA adapter on top of the stock
Meta-Llama-3-8B-Instruct, so BASE + LORA are loaded separately here (unlike CAT / Circuit
Breakers, which publish merged full weights).

Run (needs GPU):
  python method.py response --task all
  python method.py judge    --task xstest
  python method.py judge    --task harmbench
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

import json                                           # noqa: E402

from core.runner import run_method                    # noqa: E402
from core.local_client import local_generate, resolve_model   # noqa: E402


# ----- Fix a self-trained adapter whose target_modules list names modules Llama-3 lacks -----
def sanitize_local_adapter(lora_dir):
    """Upstream's LoRA config targets `w1`/`w2`/`w3` (Mixtral-style names) alongside the
    Llama ones. Llama-3 has no such modules, so nothing is ever trained for them:
    inspecting the produced adapter shows LoRA weights for exactly
    q/k/v/o_proj + gate/up/down_proj and nothing else.

    peft 0.9 (the legacy training venv) silently ignores the missing names; peft 0.19 (our
    inference venv) does not, and refuses to load:
        RuntimeError: Error(s) in loading state_dict for PeftModelForCausalLM
    So drop the dead entries from the config. Only touches adapters we trained ourselves -
    the released checkpoint on the Hub is left alone.
    """
    cfg_path = os.path.join(lora_dir, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return lora_dir
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    dead = {"w1", "w2", "w3"}
    mods = cfg.get("target_modules") or []
    keep = [m for m in mods if m not in dead]
    if len(keep) != len(mods):
        with open(cfg_path + ".orig", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        cfg["target_modules"] = keep
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"[DeRTa] removed {sorted(dead & set(mods))} from target_modules "
              f"(no weights exist for them); backup: {cfg_path}.orig")
    return lora_dir

# ----- Base + LoRA (authors' recommended pair) -----
# NousResearch mirror because meta-llama/* is gated and the server has no HF token.
BASE = resolve_model("NousResearch/Meta-Llama-3-8B-Instruct", env_var="DERTA_BASE")
LORA = os.environ.get("DERTA_LORA", "Youliang/llama3-8b-instruct-lora-derta-100step")

# Other released checkpoints:
#   Youliang/llama3-8b-derta                     (full weights, base Meta-Llama-3-8B)
#   Youliang/llama3-8b-instruct-derta-100step    (full weights, authors mark "Not Recommended")
#   Youliang/llama3-70b-lora-derta               (70B - does not fit our 40GB)
# For a full-weight checkpoint: put it in BASE and set DERTA_LORA="".


if __name__ == "__main__":
    local_kwargs = {}
    if LORA and os.path.isdir(LORA):          # local dir = we trained it -> may need fixing
        sanitize_local_adapter(LORA)
        # Upstream's trainer adds a pad token, resizes the embedding matrix and saves
        # embed_tokens/lm_head inside the adapter (128257 rows vs the stock 128256). Take
        # the tokenizer from the adapter and resize the base to match, or peft refuses:
        #   size mismatch for base_model.model.model.embed_tokens.weight
        if os.path.exists(os.path.join(LORA, "tokenizer_config.json")):
            local_kwargs = {"tokenizer_id": LORA, "resize_to_tokenizer": True}

    run_method(
        name="DeRTa",
        slug="derta",
        defense_type="intra",
        backend="local",
        model=BASE,
        lora=LORA or None,
        local_kwargs=local_kwargs,
        generate=local_generate,
        out_dir=os.path.join(HERE, "outputs"),
    )
