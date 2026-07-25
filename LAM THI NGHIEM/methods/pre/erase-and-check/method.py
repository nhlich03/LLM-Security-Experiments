"""
erase-and-check - Certifying LLM Safety against Adversarial Prompting (pre-processing DETECTOR, P4)
                  - GENERATION entry point.

Paper: https://arxiv.org/pdf/2309.02705 | repo: aounon/certified-llm-safety
Co che (detector, verbatim logic tu defenses.py::erase_and_check_suffix):
  1. Tokenize prompt bang tokenizer cua safety filter (bo token dac biet dau: input_ids[1:]).
  2. Tao danh sach ung vien = prompt goc + moi ban bi xoa 1..min(max_erase, n) token TU CUOI (suffix mode).
  3. Chay safety filter (DistilBERT) tren tung ung vien.
  4. Flag HARMFUL neu BAT KY ung vien nao bi flag (LABEL_0 = harmful, LABEL_1 = safe).
Flow: flag -> tra ve REFUSAL (khong goi target). safe -> goi target tra loi binh thuong.

Filter = model phu tro (chay LOCAL tren GPU), KHONG dung Groq:
  DistilBERT (distilbert-base-uncased) + trong so fine-tuned `models/distilbert_suffix.pt`
  (tai tu Dropbox trong README upstream). Cost filter = local giay (meter.local),
  chi call `target` moi tinh token API.

Setting (mac dinh paper): mode=suffix, max_erase=20.
Cost: suffix = O(n) -> min(max_erase, n)+1 lan classify (batch 1 forward), RE (local GPU).

Run (can GPU cho filter + Groq key cho target -> chay tren server):
  python method.py response --task all
  python method.py judge    --task xstest        # over-refusal (API)
  python method.py judge    --task harmbench     # ASR (classifier GPU 40GB)
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))   # LAM THI NGHIEM
sys.path.insert(0, ROOT)                                        # for `core`

from core.runner import run_method            # noqa: E402

# ----- Config (mac dinh paper) -----
MODE = "suffix"
MAX_ERASE = 20
WEIGHTS = os.path.join(HERE, "models", "distilbert_suffix.pt")   # fine-tuned filter weights
REFUSAL = "I'm sorry, but I cannot help with that request."

# ----- Safety filter (DistilBERT, local GPU) - load 1 lan -----
_filter = None
_tok = None


def _load_filter():
    global _filter, _tok
    if _filter is not None:
        return
    import torch
    import transformers
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    _tok = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(WEIGHTS, map_location=dev))
    model.eval()
    _filter = transformers.pipeline(
        "text-classification", model=model, tokenizer=_tok,
        device=0 if torch.cuda.is_available() else -1)


# ----- erase-and-check suffix (VERBATIM logic tu defenses.py) -----
def _erase_and_check_suffix(prompt, max_erase=MAX_ERASE):
    prompt_tokens = _tok(prompt)["input_ids"][1:]        # drop leading special token
    prompt_length = len(prompt_tokens)
    prompt_list = [prompt]
    for i in range(min(max_erase, prompt_length)):
        erased = _tok.decode(prompt_tokens[:-(i + 1)])
        prompt_list.append(erased)
    # truncation=True: DistilBERT max 512 token (HarmBench contextual prompt co the dai) - chi la I/O guard
    outputs = _filter(prompt_list, truncation=True)
    return any(o["label"] == "LABEL_0" for o in outputs)  # LABEL_0 = harmful -> flag neu bat ky


# ----- Defense = detector (local filter) + target neu pass -----
def ec_generate(client, raw, meter):
    _load_filter()
    with meter.local("filter_check"):                    # O(n) filter forward - local GPU
        flagged = _erase_and_check_suffix(raw, MAX_ERASE)
    if flagged:
        return REFUSAL                                   # chan -> tu choi co dinh, khong goi target
    answer, r = client.chat(raw)                         # safe -> target tra loi binh thuong
    meter.record_api("target", r)
    return answer


if __name__ == "__main__":
    run_method(
        name="erase_and_check",
        slug="erase_and_check",
        defense_type="pre",
        model="llama-3.1-8b-instant",
        generate=ec_generate,          # multi-call hook (detector + target)
        out_dir=os.path.join(HERE, "outputs"),
    )
