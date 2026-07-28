"""Dataset loaders + prompt builders for the benchmarks (data lives in ../data)."""

import os

import pandas as pd

_CORE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(_CORE, "..", "data"))

HARMBENCH_CSV = os.path.join(DATA_DIR, "harmbench.csv")
XSTEST_CSV = os.path.join(DATA_DIR, "xstest.csv")
# JustEval utility set. MAC DINH = subset 200 (stratified proportional theo `dataset`,
# seed=42, sinh boi tools/make_justeval_subset.py) -> re + nhat quan moi method.
# Chay FULL 800:  JUSTEVAL_FILE=justeval.csv python method.py ... --task justeval
JUSTEVAL_CSV = os.path.join(DATA_DIR, os.environ.get("JUSTEVAL_FILE", "justeval_200.csv"))

# Columns kept in the response file (never merge context into Behavior).
HARMBENCH_KEEP = ["Behavior", "FunctionalCategory", "SemanticCategory",
                  "Tags", "ContextString", "BehaviorID"]
XSTEST_KEEP = ["id", "prompt", "type", "label", "focus", "note"]
JUSTEVAL_KEEP = ["id", "prompt", "source_id", "dataset", "category"]


def load_harmbench():
    return pd.read_csv(HARMBENCH_CSV)


def load_xstest():
    return pd.read_csv(XSTEST_CSV)


def load_justeval():
    return pd.read_csv(JUSTEVAL_CSV)


# ----- Build the raw prompt per HarmBench category (generation only) -----
def harmbench_prompt(row):
    if str(row["FunctionalCategory"]).strip() == "contextual":
        return f"{row['ContextString']}\n\n---\n\n{row['Behavior']}"
    return str(row["Behavior"])
