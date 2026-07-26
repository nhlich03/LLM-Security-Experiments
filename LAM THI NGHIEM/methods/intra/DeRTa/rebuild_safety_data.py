"""
DeRTa - rebuild the safety half of the training data, which upstream does NOT ship.

`repo/data/train/generate_training_data.py` opens `safety_beaver_safe_and_unsafe_response.json`
as its very first line. That file is NOT in the repo (only the six helpfulness shards are),
and the README never mentions downloading it, so upstream's own data-generation step fails:

    FileNotFoundError: [Errno 2] No such file or directory:
    'safety_beaver_safe_and_unsafe_response.json'

The paper says the safety data comes from BeaverTails. Reading the consumer, each record
needs exactly three fields:

    instruction    - the harmful question
    output         - an UNSAFE response to it   (used as the harmful prefix + as the
                     "refusal token prediction" target in Sorry_data)
    safe_response  - a SAFE response to it      (what the model should switch to)

`PKU-Alignment/PKU-SafeRLHF` has precisely that shape: every row holds one prompt and two
responses with per-response safety labels. Whenever exactly one of the two is safe we get an
(unsafe, safe) pair for the same prompt - which is what DeRTa's mid-generation transition
training needs.

⚠️ THIS IS A RECONSTRUCTION, NOT THE AUTHORS' FILE. Same source dataset and same schema, but
the concrete pairs (and therefore the exact refusal wording) will not match theirs. Any model
trained on it is "DeRTa-style", not a bit-exact reproduction - say so in the report.

The output is JSON LINES (one object per line), because upstream's `read_from_json` iterates
lines and `json.loads` each one - despite the `.json` name.

Usage:
  python rebuild_safety_data.py                 # default 6000 pairs, enough for llama_derta
  python rebuild_safety_data.py --limit 60000
Then the normal path works:
  cd repo/data/train && python generate_training_data.py
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "repo", "data", "train",
                   "safety_beaver_safe_and_unsafe_response.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="PKU-Alignment/PKU-SafeRLHF")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=6000,
                    help="generate_training_data.py only consumes 6000 (Concate_refuse_data[:6000])")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    from datasets import load_dataset

    print(f"[data] loading {a.dataset} split={a.split} ...")
    ds = load_dataset(a.dataset, split=a.split)
    print(f"[data] {len(ds)} rows | columns: {ds.column_names}")

    rows, skipped = [], 0
    for r in ds:
        s0, s1 = r.get("is_response_0_safe"), r.get("is_response_1_safe")
        if s0 is None or s1 is None or s0 == s1:
            skipped += 1               # need exactly one safe + one unsafe
            continue
        safe_resp = r["response_0"] if s0 else r["response_1"]
        unsafe_resp = r["response_1"] if s0 else r["response_0"]
        if not (safe_resp or "").strip() or not (unsafe_resp or "").strip():
            skipped += 1
            continue
        rows.append({"instruction": r["prompt"],
                     "output": unsafe_resp,          # harmful continuation
                     "safe_response": safe_resp})    # what it should switch to
        if len(rows) >= a.limit:
            break

    if not rows:
        raise SystemExit("[data] no usable pairs - check the dataset schema")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:      # JSON LINES, see docstring
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[data] wrote {len(rows)} pairs (skipped {skipped}) -> {a.out}")
    print("[data] next: cd repo/data/train && python generate_training_data.py")


if __name__ == "__main__":
    main()
