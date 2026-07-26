"""
DRO - Prompt-Driven LLM Safeguarding via Directed Representation Optimization
      (in-processing, ICML 2024) - GENERATION entry point.

Paper: https://arxiv.org/abs/2401.18018 | repo: chujiezheng/LLM-Safeguard (clone trong repo/)

Co che: quan sat rang model DA "biet" phan biet harmful/harmless trong khong gian bieu
dien, chi la ranh gioi chua du tach. DRO toi uu mot SOFT PROMPT lien tuc sao cho bieu
dien cua prompt harmful bi day THEO huong tu choi, con harmless bi day NGUOC lai.
Deploy = prepend soft prompt do -> khong co overhead decoding, ~1xT.

Cach nap soft prompt (VERBATIM logic tu repo/code/generate.py):
  1. Them N token gia `<soft_prompt_0..N-1>` vao tokenizer.
  2. Mo rong bang embedding, ghi soft prompt da train vao dung N o moi do.
  3. Prepend mot system message gom dung N token do.
Nho vay soft prompt di qua chat template binh thuong, khong phai hack inputs_embeds.

Muon co soft prompt thi phai chay `train_smoke.py` truoc (3 stage: forward -> estimate
-> train). Upstream KHONG phat hanh soft prompt da train san.

Phan loai: xet chat thi day la borderline in/pre (soft prompt = embedding da train roi
prepend vao input, cung ho voi RPO). Giu o IN theo dinh nghia "in = can thiep tang bieu
dien"; xem docs/PHUONG_PHAP.md section 5.1.

Run (needs GPU):
  python train_smoke.py                                  # tao soft prompt truoc
  DRO_SOFT_PROMPT=<path.safetensors> python method.py response --task harmbench
  python method.py judge --task harmbench
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from core.runner import run_method                                    # noqa: E402
from core.local_client import LocalClient, resolve_model              # noqa: E402

BASE = resolve_model("NousResearch/Meta-Llama-3-8B-Instruct", env_var="DRO_BASE")

# Mac dinh tro toi cho train_smoke.py ghi ra
_DEFAULT_SP = os.path.join(
    HERE, "work", "trained_prompts",
    BASE.rstrip("/").split("/")[-1], "type.all_length.default.safetensors")
SOFT_PROMPT = os.environ.get("DRO_SOFT_PROMPT", _DEFAULT_SP)


class DROClient(LocalClient):
    """LocalClient + soft prompt nhung vao bang embedding. `.chat()` giu nguyen."""

    def __init__(self, model_id, soft_prompt_path, **kw):
        super().__init__(model_id, **kw)
        self.n_soft = self._install_soft_prompt(soft_prompt_path)
        # System message = dung N token gia, y het generate.py:38
        self.soft_system = "".join(f"<soft_prompt_{i}>" for i in range(self.n_soft))
        print(f"[DRO] soft prompt {self.n_soft} token <- {soft_prompt_path}")

    def _install_soft_prompt(self, path):
        import torch
        from torch import nn
        from safetensors import safe_open

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Khong thay soft prompt: {path}\n"
                f"Chay `python train_smoke.py` truoc (upstream khong phat hanh ban da train).")
        with safe_open(path, framework="pt") as f:
            soft_prompt = f.get_tensor("soft_prompt")

        n = soft_prompt.size(0)
        # VERBATIM logic tu generate.py::process_soft_prompt_as_word_embedding
        old_size = len(self.tokenizer)
        self.tokenizer.add_tokens([f"<soft_prompt_{i}>" for i in range(n)], special_tokens=True)
        new_size = len(self.tokenizer)

        old_emb = self.model.get_input_embeddings()
        new_emb = nn.Embedding(max(new_size, old_emb.num_embeddings),
                               old_emb.embedding_dim, self.model.config.pad_token_id)
        new_emb.weight.data[:old_size] = old_emb.weight.data[:old_size]
        new_emb.weight.data[old_size:new_size] = soft_prompt.data.to("cpu")
        self.model.set_input_embeddings(
            new_emb.to(device=old_emb.weight.device, dtype=old_emb.weight.dtype))
        # lm_head khong doi -> cac token gia khong bao gio duoc SINH ra, dung y upstream
        return n

    def chat(self, user_content, system=None, **extra):
        # Soft prompt luon la system message dau tien
        messages = [{"role": "system", "content": self.soft_system},
                    {"role": "user", "content": user_content}]
        return self.chat_messages(messages, **extra)


def _factory(model, temperature, max_tokens):
    return DROClient(model, SOFT_PROMPT, temperature=temperature, max_tokens=max_tokens)


# ----- Defense nam trong soft prompt -> 1 lan sinh, do ca giay lan token -----
def dro_generate(client, raw, meter):
    with meter.local("soft_prompt_decode") as rec:
        text, resp = client.chat(raw)
        rec.from_usage(resp)
    return text


if __name__ == "__main__":
    run_method(
        name="DRO",
        slug="dro",
        defense_type="in",
        model=BASE,
        client_factory=_factory,
        generate=dro_generate,
        out_dir=os.path.join(HERE, "outputs"),
    )
