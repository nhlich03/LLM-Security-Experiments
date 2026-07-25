"""
compare_methods.py - Bang so sanh cac phuong phap phong thu (thay cho tools/view_outputs.ipynb).

Doc outputs/ cua MOI method -> gom 4 metric va in bang so sanh:
  - ASR (HarmBench, %)          : cang THAP cang tot  <- harmbench_*_judged.csv (cot jailbroken)
  - over-refusal (XSTest, %)     : cang THAP cang tot  <- xstest_*_judged.csv   (judge2 refusal)
  - utility (JustEval, 1-5)      : cang CAO cang tot   <- justeval_*_judged.csv (je_avg)
  - cost (call/req, token, sec)  : cang THAP cang re   <- *_cost_summary.csv

Chay:
  python tools/compare_methods.py                 # in ra man hinh + luu tools/comparison.md
  python tools/compare_methods.py --md results.md  # doi duong dan file markdown xuat ra
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows console hay la cp1252 -> ep utf-8
except Exception:
    pass

# ----- Metric definitions (khop metrics/xstest.py) -----
REFUSED = ["2_full_refusal", "3_partial_refusal"]     # judge2 -> tinh la tu choi
INVALID = ["0_missing", "error", "unparseable"]        # loai khoi mau so
NO_DEFENSE_FIRST = "no_defense"                         # dat baseline len dau bang


# ----- Locate methods/ dir (script nam o tools/) -----
def find_methods_dir():
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "methods").is_dir():
            return p / "methods"
    raise SystemExit("Khong tim thay thu muc methods/")


def _one(outdir, task, kind):
    hits = sorted(outdir.glob(f"{task}_*_{kind}.csv"))
    return hits[0] if hits else None


# ----- Per-metric extractors -----
def get_asr(outdir):
    f = _one(outdir, "harmbench", "judged")
    if f is None:
        return None
    d = pd.read_csv(f)
    return d["jailbroken"].mean() if "jailbroken" in d.columns else None


def get_over_refusal(outdir):
    f = _one(outdir, "xstest", "judged")
    if f is None:
        return None, None
    d = pd.read_csv(f)
    orr1 = (d["judge1"] == "2_full_refusal").mean() if "judge1" in d.columns else None
    orr2 = None
    if "judge2" in d.columns:
        valid = d[~d["judge2"].isin(INVALID)]
        orr2 = valid["judge2"].isin(REFUSED).mean() if len(valid) else None
    return orr1, orr2


def get_utility(outdir):
    f = _one(outdir, "justeval", "judged")
    if f is None:
        return None
    d = pd.read_csv(f)
    if "je_avg" in d.columns:
        return d["je_avg"].mean()
    aspects = [c for c in d.columns if c.startswith("je_") and c != "je_avg"]
    return d[aspects].mean().mean() if aspects else None


def get_cost(outdir):
    frames = [pd.read_csv(f) for f in sorted(outdir.glob("*_cost_summary.csv"))]
    if not frames:
        return {}
    allc = pd.concat(frames, ignore_index=True)
    # train = MOT LAN (max, khong trung binh); infer (call/token/local) = trung binh moi request
    train = float(allc["train_sec"].max()) if "train_sec" in allc.columns else 0.0
    return {
        "call/req": allc["n_calls"].mean(),
        "tok_in/req": allc["api_in_tokens"].mean(),
        "tok_out/req": allc["api_out_tokens"].mean(),
        "local_s/req": allc["local_sec"].mean(),
        "train_s": train,
    }


# ----- Collect all methods -----
def collect(methods_dir):
    rows = []
    for outdir in sorted(methods_dir.glob("*/outputs")) + sorted(methods_dir.glob("*/*/outputs")):
        name = outdir.relative_to(methods_dir).parent.as_posix()
        asr = get_asr(outdir)
        orr1, orr2 = get_over_refusal(outdir)
        util = get_utility(outdir)
        cost = get_cost(outdir)
        if asr is None and orr2 is None and util is None and not cost:
            continue
        rows.append({
            "method": name,
            "ASR% ↓": None if asr is None else round(100 * asr, 1),
            "over-refusal% (LLM-judge) ↓": None if orr2 is None else round(100 * orr2, 1),
            "over-refusal% (string-match) ↓": None if orr1 is None else round(100 * orr1, 1),
            "utility/5 ↑": None if util is None else round(util, 2),
            "call/req": None if not cost else round(cost["call/req"], 1),
            "tok_in/req": None if not cost else round(cost["tok_in/req"]),
            "tok_out/req": None if not cost else round(cost["tok_out/req"]),
            "local_s/req": None if not cost else round(cost["local_s/req"], 3),
            "train_s(1lần)": None if not cost else round(cost["train_s"], 1),
        })
    return rows


# ----- Which direction is "best" per column (in dam gia tri tot nhat) -----
DIRECTION = {
    "ASR% ↓": "min",
    "over-refusal% (LLM-judge) ↓": "min",
    "over-refusal% (string-match) ↓": "min",
    "utility/5 ↑": "max",
    "call/req": "min",
    "tok_in/req": "min",
    "tok_out/req": "min",
    "local_s/req": "min",
}


def best_indices(rows, col):
    vals = [(i, r[col]) for i, r in enumerate(rows) if r[col] is not None]
    if not vals:
        return set()
    target = (min if DIRECTION[col] == "min" else max)(v for _, v in vals)
    return {i for i, v in vals if v == target}


def _cell(v):
    return "-" if v is None else str(v)


def render_console(rows, cols):
    bold_o, bold_c = "\033[1m", "\033[0m"                 # ANSI dam cho terminal
    best = {c: best_indices(rows, c) for c in cols if c in DIRECTION}
    width = {c: max(len(c), *(len(_cell(r[c])) for r in rows)) for c in cols}
    def row_line(cells_raw, mark=None):
        out = []
        for c in cols:
            s = cells_raw[c]
            pad = s.ljust(width[c]) if c == "method" else s.rjust(width[c])
            if mark is not None and c in best and mark in best[c]:
                pad = bold_o + pad + bold_c
            out.append(pad)
        return "  ".join(out)
    lines = [row_line({c: c for c in cols})]
    lines.append("  ".join("-" * width[c] for c in cols))
    for i, r in enumerate(rows):
        lines.append(row_line({c: _cell(r[c]) for c in cols}, mark=i))
    return "\n".join(lines)


def render_md(rows, cols):
    best = {c: best_indices(rows, c) for c in cols if c in DIRECTION}
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for i, r in enumerate(rows):
        cells = []
        for c in cols:
            s = _cell(r[c])
            if s != "-" and c in best and i in best[c]:
                s = f"**{s}**"
            cells.append(s)
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def order_key(name):
    base = name.split("/")[-1]
    return (0 if base == NO_DEFENSE_FIRST else 1, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=str(Path(__file__).resolve().parent / "comparison.md"),
                    help="duong dan file markdown xuat ra")
    args = ap.parse_args()

    methods_dir = find_methods_dir()
    rows = collect(methods_dir)
    if not rows:
        raise SystemExit("Chua co outputs nao co the cham. Chay method.py truoc.")
    rows.sort(key=lambda r: order_key(r["method"]))
    cols = list(rows[0].keys())                          # method truoc, roi cac metric

    print("\n" + "=" * 90)
    print("  BANG SO SANH PHUONG PHAP PHONG THU  (in dam = tot nhat moi cot)")
    print("  ASR / over-refusal / cost: THAP tot | utility: CAO tot")
    print("  over-refusal: (LLM-judge) = gpt-oss-20b (chinh) | (string-match) = do mau tu choi")
    print("=" * 90 + "\n")
    print(render_console(rows, cols))
    print()

    # ----- Also write a markdown table for reports (best-in-column bolded) -----
    md = ["# Bang so sanh phuong phap phong thu", "",
          "ASR (HarmBench, Llama-13b) ↓ · over-refusal (XSTest) ↓ · utility (JustEval) ↑ · "
          "**in đậm = tốt nhất mỗi cột**", "",
          "over-refusal: **(LLM-judge)** = gpt-oss-20b (số chính) · **(string-match)** = đo mẫu từ chối (tham chiếu)", "",
          render_md(rows, cols)]
    Path(args.md).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Da luu bang markdown: {args.md}")


if __name__ == "__main__":
    main()
