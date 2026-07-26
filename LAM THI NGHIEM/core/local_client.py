"""
LocalClient - target model running LOCALLY (HF transformers), same interface as GroqClient.

Every API method so far talks to Groq through `GroqClient.chat(user, system) -> (text, resp)`.
White-box methods (in / intra) need the weights, so they need a local target instead. This
class keeps the exact same call signature so `core.runner` and every existing method hook
work unchanged - only the backend differs.

Extra surface that white-box methods need (GroqClient cannot offer these):
  .model      - the HF model (hooks, logits, adapters)
  .tokenizer  - the tokenizer
  .device     - where the model lives
  .build_inputs(messages) -> dict of tensors already on device (chat template applied)

Generation defaults mirror the API side so the two tables stay comparable:
temperature=0 (greedy), max_new_tokens=512.

`resp.usage` is filled with real token counts so `meter.record_api()` still works, but for a
local target the project convention (CLAUDE.md section 5) is to measure SECONDS via
`meter.local(label)` instead - tokens are there only for reference.
"""

import os
from types import SimpleNamespace


def transformers_version():
    import transformers
    return transformers.__version__


class LocalClient:
    def __init__(self, model_id, lora=None, dtype="bfloat16", device_map="auto",
                 temperature=0.0, max_tokens=512, load_in_4bit=False,
                 trust_remote_code=False, attn_implementation=None,
                 tokenizer_id=None, merge_lora=False, resize_to_tokenizer=False):
        """
        model_id   : HF repo id or local path of the base model
        lora       : optional LoRA adapter (repo id or path) applied on top
        merge_lora : merge the adapter into the base weights (faster inference,
                     but you lose the ability to disable it - SafeDecoding needs it OFF)
        load_in_4bit: bitsandbytes 4-bit (use when two models must share one GPU)
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.lora = lora
        self.temperature = temperature
        self.max_tokens = max_tokens

        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id or model_id, trust_remote_code=trust_remote_code)
        # Most chat models ship without a pad token; generate() needs one.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # transformers v5 renamed `torch_dtype` -> `dtype`; v4 only knows the old name.
        dtype_kw = "dtype" if int(transformers_version().split(".")[0]) >= 5 else "torch_dtype"
        kw = {dtype_kw: torch_dtype, "device_map": device_map,
              "trust_remote_code": trust_remote_code}
        if attn_implementation:
            kw["attn_implementation"] = attn_implementation
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            kw.pop(dtype_kw, None)

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)

        # Some training scripts add a special token and resize the embedding matrix, then
        # save embed_tokens/lm_head inside the adapter. Loading that onto an unresized base
        # fails with a size mismatch (e.g. 128257 vs 128256 for Llama-3 + DeRTa). Resizing
        # first makes the shapes line up; the extra row is overwritten by the checkpoint.
        if resize_to_tokenizer:
            cur = self.model.get_input_embeddings().weight.shape[0]
            want = len(self.tokenizer)
            if cur != want:
                self.model.resize_token_embeddings(want)
                print(f"[LocalClient] resized embeddings {cur} -> {want} to match tokenizer")

        if lora:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora)
            if merge_lora:
                self.model = self.model.merge_and_unload()

        self.model.eval()
        self.device = next(self.model.parameters()).device

        # Llama-3 stops on <|eot_id|>, not on the generic eos - miss this and every
        # response runs to max_tokens with junk appended.
        self.terminators = [self.tokenizer.eos_token_id]
        for t in ("<|eot_id|>", "<|end_of_text|>"):
            tid = self.tokenizer.convert_tokens_to_ids(t)
            if tid is not None and tid >= 0 and tid not in self.terminators:
                self.terminators.append(tid)

        print(f"[LocalClient] {model_id}"
              + (f" + LoRA {lora}" if lora else "")
              + f" | dtype={dtype}{' 4bit' if load_in_4bit else ''} | device={self.device}")

    # ----- Chat template -> tensors on device -----
    def build_inputs(self, messages, add_generation_prompt=True):
        if self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        else:
            # Fallback for base models with no template (keep it dumb and explicit).
            text = "".join(f"{m['role']}: {m['content']}\n" for m in messages) + "assistant:"
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return {k: v.to(self.device) for k, v in enc.items()}

    # ----- Same signature as GroqClient.chat -----
    def chat(self, user_content, system=None, **extra):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_content})
        return self.chat_messages(messages, **extra)

    def chat_messages(self, messages, max_tokens=None, temperature=None, **extra):
        import torch

        inputs = self.build_inputs(messages)
        in_len = inputs["input_ids"].shape[1]
        temp = self.temperature if temperature is None else temperature

        gen_kw = dict(max_new_tokens=max_tokens or self.max_tokens,
                      pad_token_id=self.tokenizer.pad_token_id,
                      eos_token_id=self.terminators)
        if temp and temp > 0:
            gen_kw.update(do_sample=True, temperature=temp)
        else:
            gen_kw.update(do_sample=False)
        gen_kw.update(extra)

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kw)

        new_ids = out[0][in_len:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        usage = SimpleNamespace(prompt_tokens=int(in_len), completion_tokens=int(len(new_ids)))
        return text, SimpleNamespace(usage=usage)

    # ----- Free the GPU (useful when a method loads several models in sequence) -----
    def free(self):
        import gc
        import torch
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ----- Default hook for a local target: measure SECONDS, not tokens -----
def local_generate(client, raw, meter, label="target"):
    """Drop-in `generate` hook for any method whose whole defense is baked into the
    weights (all intra methods): call the target once, time it on the GPU."""
    with meter.local(label):
        text, _ = client.chat(raw)
    return text


# ----- Resolve a model id: allow an env override so the base model is set in ONE place -----
def resolve_model(default_id, env_var="LOCAL_TARGET_MODEL"):
    """Base model is still undecided (Llama-3 vs Llama-3.1 vs Qwen). Every local method
    calls this, so switching base = one env var, not editing five files."""
    return os.environ.get(env_var) or default_id
