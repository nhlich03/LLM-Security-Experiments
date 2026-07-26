"""
SafeDecoding - Safety-Aware Decoding (in-processing, ACL 2024 long) - GENERATION entry point.

Paper: https://aclanthology.org/2024.acl-long.303.pdf | repo: uw-nsl/SafeDecoding
Mechanism (decoding-time, weights untouched): run the base model and a small "expert"
(the same model + a LoRA fine-tuned on 72 refusal pairs) side by side. For the FIRST m
tokens only, build a sample space from tokens both models rank highly, then amplify the
direction the expert disagrees in:

    p_new(x) = p_base(x) + alpha * (p_expert(x) - p_base(x))

After m tokens the expert is dropped and the base model finishes normally. Rationale: the
safety disclaimer is decided in the first couple of tokens, so steering there is enough -
which is why the reported overhead is only 1.03-1.07x (ATGR).

Hyperparameters are the defaults of the upstream CLI (`repo/exp/defense.py`):
    alpha=3, first_m=2, top_k=10, num_common_tokens=5

Sample-space detail, verbatim from `repo/utils/safe_decoding.py`: start from the top
`num_common_tokens` of each model and WIDEN the window until at least `num_common_tokens`
token ids are common to both. (`top_k` is only used for verbose logging upstream - it does
not affect the output; kept here for fidelity.)

DEVIATION FROM UPSTREAM (I/O only, math identical): upstream duplicates the batch and calls
`generate(adapter_names=["base","expert"])`, a mixed-adapter-batch feature that only exists
in the PEFT FORK vendored at `repo/peft/`. Depending on that fork would pin our whole
environment to an old PEFT. Instead we do two forwards per step - one with the adapter
enabled (expert) and one inside `disable_adapter()` (base). Same logits, same tokens, no
fork required.

Model: upstream ships experts for vicuna / llama2 / guanaco / falcon / dolphin only - there
is NO Llama-3 expert. Default here is their llama2 expert on Llama-2-7b-chat-hf, i.e. an
authors' released artifact with no training on our side. To get an expert for another base
model, see `train_expert.py` (72 samples, under a minute).

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

from core.runner import run_method                # noqa: E402
from core.local_client import LocalClient, resolve_model   # noqa: E402

# ----- Config (upstream defaults from repo/exp/defense.py) -----
ALPHA = float(os.environ.get("SD_ALPHA", 3))
FIRST_M = int(os.environ.get("SD_FIRST_M", 2))
TOP_K = int(os.environ.get("SD_TOP_K", 10))
NUM_COMMON_TOKENS = int(os.environ.get("SD_NUM_COMMON", 5))

# Base model + its matching expert adapter. The pair MUST match: an expert LoRA trained on
# llama2 is meaningless on top of another base.
# NousResearch mirror: meta-llama/* is gated and the server has no HF token.
BASE = resolve_model("NousResearch/Llama-2-7b-chat-hf", env_var="SD_BASE")
EXPERT = os.environ.get("SD_EXPERT", os.path.join(HERE, "repo", "lora_modules", "llama2"))
# Other released pairs (base -> expert dir under repo/lora_modules/):
#   lmsys/vicuna-7b-v1.5              -> vicuna
#   timdettmers/guanaco-7b (merged)   -> guanaco
#   tiiuae/falcon-7b-instruct         -> falcon
#   cognitivecomputations/dolphin-... -> dolphin


class SafeDecodingClient(LocalClient):
    """LocalClient + the two-model decoding loop. Keeps `.chat()` so core.runner is unchanged."""

    def __init__(self, model_id, expert_lora, alpha=ALPHA, first_m=FIRST_M,
                 top_k=TOP_K, num_common_tokens=NUM_COMMON_TOKENS, **kw):
        # LocalClient loads the base and attaches the adapter; do NOT merge it - the whole
        # method depends on being able to switch the adapter off again.
        super().__init__(model_id, lora=expert_lora, merge_lora=False, **kw)
        self.alpha = alpha
        self.first_m = first_m
        self.top_k = top_k
        self.num_common_tokens = num_common_tokens
        print(f"[SafeDecoding] alpha={alpha} first_m={first_m} "
              f"top_k={top_k} num_common_tokens={num_common_tokens}")

    # ----- Sample space: widen the window until enough token ids are common to both -----
    def _common_tokens(self, sorted_base, sorted_expert):
        common = set()
        iter_range = self.num_common_tokens
        limit = min(len(sorted_base), len(sorted_expert))
        while len(common) < self.num_common_tokens:
            common |= (set(sorted_base[:iter_range].tolist())
                       & set(sorted_expert[:iter_range].tolist()))
            iter_range += 1
            if iter_range > limit:
                break
        return common

    # ----- One decoding step: (base logits, expert logits) -> chosen token id -----
    def _steered_token(self, input_ids, attention_mask):
        import torch

        with torch.no_grad():
            # expert = adapter ON
            logits_expert = self.model(input_ids=input_ids,
                                       attention_mask=attention_mask).logits[0, -1, :]
            # base = adapter OFF (replaces upstream's mixed-adapter batch)
            with self.model.disable_adapter():
                logits_base = self.model(input_ids=input_ids,
                                         attention_mask=attention_mask).logits[0, -1, :]

        scores_base = torch.nn.functional.log_softmax(logits_base, dim=-1)
        scores_expert = torch.nn.functional.log_softmax(logits_expert, dim=-1)
        sorted_base = torch.argsort(scores_base, descending=True)
        sorted_expert = torch.argsort(scores_expert, descending=True)

        common = self._common_tokens(sorted_base, sorted_expert)
        if not common:                                   # never observed, but do not crash
            return int(sorted_base[0].item())
        ids = torch.tensor(sorted(common), device=scores_base.device)

        # p_new = p_base + alpha * (p_expert - p_base), floored at 1e-8 before log
        p_base = torch.exp(scores_base[ids])
        p_expert = torch.exp(scores_expert[ids])
        p_new = p_base + self.alpha * (p_expert - p_base)
        p_new = torch.where(p_new > 0, p_new, torch.full_like(p_new, 1e-8))
        updated_scores = torch.log(p_new)

        normalized = torch.nn.functional.softmax(updated_scores.float(), dim=0)
        return int(ids[int(torch.argmax(normalized).item())].item())

    # ----- Full generation: m steered tokens, then the base model finishes -----
    def chat_messages(self, messages, max_tokens=None, temperature=None, **extra):
        import torch
        from types import SimpleNamespace

        max_new = max_tokens or self.max_tokens
        inputs = self.build_inputs(messages)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get(
            "attention_mask", torch.ones_like(input_ids))
        in_len = input_ids.shape[1]

        generated = []
        hit_eos = False
        for _ in range(min(max_new, self.first_m)):
            tid = self._steered_token(input_ids, attention_mask)
            generated.append(tid)
            if tid in self.terminators:
                hit_eos = True
                break
            nxt = torch.tensor([[tid]], device=input_ids.device)
            input_ids = torch.cat([input_ids, nxt], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype,
                                            device=attention_mask.device)], dim=1)

        # Remaining tokens come from the BASE model (upstream: adapter_names=["base"]).
        if not hit_eos:
            remaining = max_new - min(max_new, self.first_m)
            if remaining > 0:
                with torch.no_grad(), self.model.disable_adapter():
                    out = self.model.generate(
                        input_ids=input_ids, attention_mask=attention_mask,
                        max_new_tokens=remaining, do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.terminators)
                generated = out[0][in_len:].tolist()

        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        usage = SimpleNamespace(prompt_tokens=int(in_len), completion_tokens=int(len(generated)))
        return text, SimpleNamespace(usage=usage)


def _factory(model, temperature, max_tokens):
    return SafeDecodingClient(model, EXPERT, temperature=temperature, max_tokens=max_tokens)


# ----- Defense = guided decoding; cost is local SECONDS (overhead vs no_defense) -----
def sd_generate(client, raw, meter):
    with meter.local("guided_decode"):
        text, _ = client.chat(raw)
    return text


if __name__ == "__main__":
    run_method(
        name="SafeDecoding",
        slug="safedecoding",
        defense_type="in",
        model=BASE,
        client_factory=_factory,
        generate=sd_generate,
        out_dir=os.path.join(HERE, "outputs"),
    )
