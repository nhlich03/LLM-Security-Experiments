"""Jailbreak Antidote (ICLR 2025, arXiv:2410.02298) - CALIBRATION step.

Build the per-layer SPARSE SAFETY DIRECTION and save it for method.py.
Recipe (paper Eq. 1-4): for each decoder layer l, take the last-token hidden state of a
set of harmful + harmless prompts, compute the FIRST PRINCIPAL COMPONENT (PC1) of that
cloud, keep only the top-k% |coordinates| (default 5%), orient the sign toward the harmful
cluster. Direction is a unit vector; the sparse mask is folded into it before saving.

DEVIATION (declare in report): the paper derives the direction from Phan (2023)
`harmful_harmless_instructions`; we use AdvBench (harmful) + Alpaca-eval (harmless), both
PUBLIC and HELD OUT from the HarmBench eval set -> no leakage into the benchmark. The method
itself (PC1 + top-5% mask + unit-norm) is identical.

Run ONCE (needs GPU):
  PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync python calibrate.py
"""
import os
import sys
import csv
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

import torch                                              # noqa: E402
from core.local_client import LocalClient, resolve_model  # noqa: E402

# ----- Config -----
BASE = resolve_model("NousResearch/Meta-Llama-3-8B-Instruct", env_var="ANTIDOTE_BASE")
N = int(os.environ.get("ANTIDOTE_N", "128"))              # prompts per class
KFRAC = float(os.environ.get("ANTIDOTE_KFRAC", "0.05"))   # keep top 5% dims (paper default)
OUT = os.path.join(HERE, "vectors", "llama3")
HARMFUL_CSV = os.environ.get(
    "ANTIDOTE_HARMFUL",
    os.path.join(ROOT, "methods/intra/CAT/repo/data/behavior_datasets/"
                       "extra_behavior_datasets/advbench_behaviors.csv"))
HARMLESS_JSON = os.environ.get(
    "ANTIDOTE_HARMLESS",
    os.path.join(ROOT, "methods/intra/DeRTa/repo/data/test/helpfulness_alpaca_eval_300.json"))


# ----- Data loaders (held out from HarmBench) -----
def load_harmful(path, n):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            b = (r.get("Behavior") or "").strip()
            if b:
                rows.append(b)
    return rows[:n]


def load_harmless(path, n):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line)["question"])
            except Exception:
                pass
    return rows[:n]


harmful = load_harmful(HARMFUL_CSV, N)
harmless = load_harmless(HARMLESS_JSON, N)
print(f"[calibrate] harmful={len(harmful)} harmless={len(harmless)} base={BASE}")

client = LocalClient(BASE)
model = client.model
L = model.config.num_hidden_layers
D = model.config.hidden_size


# ----- Collect last-token hidden state per layer -----
def last_token_hs(prompts, tag):
    acc = [[] for _ in range(L)]
    for i, p in enumerate(prompts):
        ids = client.build_inputs([{"role": "user", "content": p}])
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        for l in range(L):
            acc[l].append(out.hidden_states[l + 1][0, -1].float().cpu())  # hs[0]=embeddings
        if (i + 1) % 32 == 0:
            print(f"  [{tag}] {i + 1}/{len(prompts)}", flush=True)
    return [torch.stack(v) for v in acc]                  # list of [n, D]


print("[calibrate] forward harmful ...", flush=True)
Hh = last_token_hs(harmful, "harmful")
print("[calibrate] forward harmless ...", flush=True)
Hb = last_token_hs(harmless, "harmless")

# ----- Per-layer PC1 -> top-k% mask -> signed unit direction -----
directions = torch.zeros(L, D)
k = max(1, int(KFRAC * D))
for l in range(L):
    X = torch.cat([Hh[l], Hb[l]], 0)
    Xc = X - X.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    v = Vh[0]
    v = v / v.norm()                                      # unit PC1
    if (Hh[l].mean(0) - Hb[l].mean(0)) @ v < 0:           # orient +dir -> harmful cluster
        v = -v
    idx = v.abs().topk(k).indices
    m = torch.zeros_like(v)
    m[idx] = 1.0
    directions[l] = v * m                                 # fold mask in
    if l % 8 == 0:
        print(f"  layer {l:2d}: kept {k}/{D} dims |masked dir|={directions[l].norm():.3f}", flush=True)

os.makedirs(OUT, exist_ok=True)
torch.save(directions, os.path.join(OUT, "antidote_directions.pt"))
with open(os.path.join(OUT, "meta.json"), "w") as f:
    json.dump({"base": BASE, "n_per_class": N, "kfrac": KFRAC, "k_dims": k,
               "L": L, "D": D, "harmful_src": HARMFUL_CSV,
               "harmless_src": HARMLESS_JSON}, f, indent=2)
print(f"[calibrate] saved {OUT}/antidote_directions.pt shape={tuple(directions.shape)}")
