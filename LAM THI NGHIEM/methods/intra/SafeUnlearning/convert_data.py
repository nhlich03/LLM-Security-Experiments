"""Re-template Safe Unlearning training data: vicuna format -> Llama-3 chat format.

The repo ships the training data with the vicuna prompt template baked into the 'input'
string. Our target is Llama-3-8B-Instruct, so we recover the raw user query from the vicuna
wrapper and re-wrap it with Llama-3's chat template. 'output' (the response) and 'type'
(loss branch: 0=normal / 1=unlearn-harmful / 2=learn-refusal) are preserved verbatim.

DEVIATION (declare): data re-templated vicuna->Llama-3; content unchanged.
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from core.local_client import resolve_model  # noqa: E402
from transformers import AutoTokenizer       # noqa: E402

BASE = resolve_model("NousResearch/Meta-Llama-3-8B-Instruct", env_var="SU_BASE")
SRC = os.path.join(HERE, "repo", "data", "ft_full_data", "vicuna_format")
OUT = os.path.join(HERE, "data_llama3")
os.makedirs(OUT, exist_ok=True)

tok = AutoTokenizer.from_pretrained(BASE)


def raw_query(vic_input):
    """Strip the vicuna wrapper '... USER: {q} ASSISTANT:' -> {q} (last turn)."""
    s = vic_input
    if "USER:" in s:
        s = s.split("USER:")[-1]
    if "ASSISTANT:" in s:
        s = s.split("ASSISTANT:")[0]
    return s.strip()


def convert(fn):
    data = json.load(open(os.path.join(SRC, fn), encoding="utf-8"))
    out = []
    for d in data:
        q = raw_query(d["input"])
        inp = tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        rec = {"input": inp, "output": d["output"], "type": d["type"]}
        out.append(rec)
    json.dump(out, open(os.path.join(OUT, fn), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    types = {}
    for r in out:
        types[r["type"]] = types.get(r["type"], 0) + 1
    print(f"[convert] {fn}: {len(out)} records, type counts={types}")


convert("train.json")
convert("dev.json")
print(f"[convert] saved to {OUT}")
