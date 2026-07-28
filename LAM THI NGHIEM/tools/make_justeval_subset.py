"""
make_justeval_subset.py - tao subset JustEval 200 cau (stratified proportional theo `dataset`).

Giu DUNG ti le 7 nguon nhu ban 800 goc -> "JustEval thu nho", dai dien nhat.
Seed CO DINH -> tai lap 100%. Dung CHUNG cho MOI method de so sanh cong bang.

Chay:  python tools/make_justeval_subset.py
Xuat:  data/justeval_200.csv  (cung schema: id, prompt, source_id, dataset, category)
"""

from pathlib import Path

import pandas as pd

SEED = 42
TARGET = 200
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "data" / "justeval.csv"
OUT = HERE.parent / "data" / "justeval_200.csv"


# ----- largest-remainder: chia TARGET theo ti le sao cho tong khop chinh xac -----
def largest_remainder(counts, target):
    total = sum(counts.values())
    exact = {k: v / total * target for k, v in counts.items()}
    alloc = {k: int(v) for k, v in exact.items()}          # floor
    left = target - sum(alloc.values())
    order = sorted(exact, key=lambda k: exact[k] - alloc[k], reverse=True)
    for k in order[:left]:                                  # phan du -> nhom co le lon nhat
        alloc[k] += 1
    return alloc


def main():
    df = pd.read_csv(SRC)
    counts = df["dataset"].value_counts().to_dict()
    n_per = largest_remainder(counts, TARGET)

    parts = [df[df.dataset == ds].sample(n=n, random_state=SEED) for ds, n in n_per.items()]
    sub = pd.concat(parts).sort_values("id").reset_index(drop=True)
    sub.to_csv(OUT, index=False, encoding="utf-8")

    print(f"seed={SEED} | target={TARGET} -> {len(sub)} cau")
    print("--- phan bo theo dataset (subset / goc) ---")
    for ds in sorted(counts, key=counts.get, reverse=True):
        print(f"  {ds:13s} {n_per[ds]:3d} / {counts[ds]}")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
