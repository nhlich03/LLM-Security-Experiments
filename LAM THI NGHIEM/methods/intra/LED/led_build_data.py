"""LED - build editing data: (harmful query X_harm -> safe refusal Y_safe) pairs.

Paper (Sec 5.1) uses 200 harmful prompts from TDC-2023 as the edit set. We substitute 200
AdvBench harmful behaviors (public, HELD OUT from the HarmBench eval set). Y_safe = a safe
refusal (paper: "desired safe response"; Table 4 uses the model's own natural refusals) -- we
use a small pool of varied refusal templates in Llama-3 style. Declared deviations: TDC->AdvBench,
templated refusals instead of per-query generated ones.

No GPU needed. Output: led_data.json  [{input: <llama3-templated prompt>, output: <refusal>}]
"""
import os
import sys
import csv
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from core.local_client import resolve_model    # noqa: E402
from transformers import AutoTokenizer         # noqa: E402

BASE = resolve_model("NousResearch/Meta-Llama-3-8B-Instruct", env_var="LED_BASE")
N = int(os.environ.get("LED_N", "200"))
HARMFUL_CSV = os.environ.get(
    "LED_HARMFUL",
    os.path.join(ROOT, "methods/intra/CAT/repo/data/behavior_datasets/"
                       "extra_behavior_datasets/advbench_behaviors.csv"))
OUT = os.path.join(HERE, "led_data.json")

REFUSALS = [
    "I cannot fulfill this request. Providing that information could cause real harm, so I "
    "won't help with it. If there's something safe I can help you with instead, I'm happy to.",
    "I can't help with that. This request involves potential harm or illegal activity, which "
    "I won't assist with. Let me know if there's a safe topic I can help you with instead.",
    "I'm sorry, but I cannot provide this. Helping with this would be unsafe and goes against "
    "my guidelines. I'd be glad to help you with something else, though.",
    "I won't be able to help with that request, as it could lead to harm. If you have a "
    "different, safe question, I'm here and happy to help.",
]


def load_harmful(path, n):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            b = (r.get("Behavior") or "").strip()
            if b:
                rows.append(b)
    return rows[:n]


tok = AutoTokenizer.from_pretrained(BASE)
harmful = load_harmful(HARMFUL_CSV, N)
pairs = []
for i, q in enumerate(harmful):
    inp = tok.apply_chat_template(
        [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    pairs.append({"input": inp, "output": REFUSALS[i % len(REFUSALS)]})

json.dump(pairs, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"[led_build] {len(pairs)} (harmful -> refusal) pairs -> {OUT}")
