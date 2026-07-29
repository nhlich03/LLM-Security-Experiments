"""ReFAT - Refusal Feature Adversarial Training (arXiv:2409.20089, ICLR 2025) - intra, LoRA.

Faithful reimplementation (no official repo). Per-layer refusal directions r^l (mean harmful -
mean harmless of last-token residual acts) are projected OUT of the residual stream during a
fraction of harmful-batch forwards (RFA, prob p_RFA=0.5), while the model is LoRA-fine-tuned to
(a) refuse harmful requests and (b) stay helpful on benign ones. Directions are recomputed every
k=4 steps from the current model.

Data (reuse existing, declared substitutes for the paper's UltraChat/Zou-2024 sets):
  D_r harmful->refusal : SafeUnlearning type-2 safe refusals + LED (harmful->refusal templates)
  D_u harmless->helpful: SafeUnlearning type-0 (benign -> GPT-4 answers, from UltraFeedback)
  directions          : AdvBench harmful (500) vs Alpaca harmless (500), last user-token acts

DEVIATIONS (declare): LoRA r=128 (paper LoRA too); 500-step fair-cost cap (paper ~313 steps/1ep);
target layers 8..31 (Table 4 Llama-3 [8,32]); refusal-target data substituted as above;
ablation applied at all positions of target layers. p_RFA=0.5, k=4, n=32 as paper.

Run: PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync python refat_train.py
"""
import os
import sys
import csv
import json
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

import torch                                                    # noqa: E402
import transformers                                             # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer    # noqa: E402
from peft import LoraConfig, get_peft_model                     # noqa: E402
from core.local_client import resolve_model                     # noqa: E402

BASE = resolve_model("NousResearch/Meta-Llama-3-8B-Instruct", env_var="REFAT_BASE")
LAYERS = list(range(int(os.environ.get("REFAT_L0", "8")), 32))   # target layers 8..31
STEPS = int(os.environ.get("REFAT_STEPS", "500"))
P_RFA = float(os.environ.get("REFAT_P", "0.5"))
K = int(os.environ.get("REFAT_K", "4"))                          # recompute interval
NDIR = int(os.environ.get("REFAT_NDIR", "500"))                  # dir estimate size (initial)
NREC = int(os.environ.get("REFAT_NREC", "32"))                   # dir recompute size
MAXLEN = int(os.environ.get("REFAT_MAXLEN", "512"))
LR = float(os.environ.get("REFAT_LR", "2e-5"))
OUT = os.path.join(HERE, "train_out")
V5 = int(transformers.__version__.split(".")[0]) >= 5
DTYPE_KW = "dtype" if V5 else "torch_dtype"

ADV = os.path.join(ROOT, "methods/intra/CAT/repo/data/behavior_datasets/"
                         "extra_behavior_datasets/advbench_behaviors.csv")
ALP = os.path.join(ROOT, "methods/intra/DeRTa/repo/data/test/helpfulness_alpaca_eval_300.json")
SU_DATA = os.path.join(ROOT, "methods/intra/SafeUnlearning/data_llama3/train.json")
LED_DATA = os.path.join(ROOT, "methods/intra/LED/led_data.json")

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token


def chat_prompt(q):
    return tok.apply_chat_template([{"role": "user", "content": q}],
                                   tokenize=False, add_generation_prompt=True)


# ----- direction data (raw harmful / harmless prompts) -----
def load_adv(n):
    rows = []
    with open(ADV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            b = (r.get("Behavior") or "").strip()
            if b:
                rows.append(b)
    return rows[:n]


def load_alp(n):
    rows = []
    with open(ALP, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line)["question"])
                except Exception:
                    pass
    return rows[:n]


harmful_prompts = load_adv(NDIR)
harmless_prompts = load_alp(NDIR)

# ----- training pairs (input templated prompt, output target) -----
def build_pairs():
    dr, du = [], []
    su = json.load(open(SU_DATA, encoding="utf-8"))
    for d in su:
        if d["type"] == 2:            # safe refusal for harmful
            dr.append((d["input"], d["output"]))
        elif d["type"] == 0:          # benign -> helpful
            du.append((d["input"], d["output"]))
    for d in json.load(open(LED_DATA, encoding="utf-8")):   # harmful -> refusal templates
        dr.append((d["input"], d["output"]))
    return dr, du


D_r, D_u = build_pairs()
print(f"[refat] D_r(harmful->refuse)={len(D_r)}  D_u(benign->help)={len(D_u)}  layers={LAYERS[0]}..{LAYERS[-1]}")


def tokenize_pair(inp, out):
    ii = tok(inp, add_special_tokens=False)["input_ids"]
    oo = tok(out + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (ii + oo)[:MAXLEN]
    labels = ([-100] * len(ii) + oo)[:MAXLEN]
    return ids, labels


# ----- model + LoRA -----
model = AutoModelForCausalLM.from_pretrained(BASE, use_cache=False,
                                             **{DTYPE_KW: torch.bfloat16}).to("cuda")
lora = LoraConfig(r=128, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lora)
model.enable_input_require_grads()
model.print_trainable_parameters()


def find_layers(m):
    node = m
    for _ in range(6):
        if hasattr(node, "layers"):
            return node.layers
        if hasattr(node, "model"):
            node = node.model
        elif hasattr(node, "base_model"):
            node = node.base_model
        else:
            break
    raise RuntimeError("cannot locate decoder layers")


decoder_layers = find_layers(model)

# ----- refusal directions (per layer) via last-token acts -----
r_hat = {l: torch.zeros(model.config.hidden_size, device="cuda", dtype=torch.bfloat16) for l in LAYERS}
hbar = {l: torch.zeros(model.config.hidden_size, device="cuda", dtype=torch.bfloat16) for l in LAYERS}


@torch.no_grad()
def last_acts(prompts):
    acc = {l: [] for l in LAYERS}
    for p in prompts:
        ids = tok(chat_prompt(p), return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
        out = model(ids, output_hidden_states=True)
        for l in LAYERS:
            acc[l].append(out.hidden_states[l + 1][0, -1].float())
    return {l: torch.stack(v) for l, v in acc.items()}


@torch.no_grad()
def recompute_dirs(nh):
    hp = last_acts(random.sample(harmful_prompts, min(nh, len(harmful_prompts))))
    bp = last_acts(random.sample(harmless_prompts, min(nh, len(harmless_prompts))))
    for l in LAYERS:
        mh, mb = hp[l].mean(0), bp[l].mean(0)
        r = mh - mb
        r = r / (r.norm() + 1e-6)
        r_hat[l] = r.to(torch.bfloat16)
        hbar[l] = mb.to(torch.bfloat16)


print("[refat] initial direction estimate ...")
recompute_dirs(NDIR)

# ----- ablation hooks (gated) -----
ablate = {"on": False}


def make_hook(l):
    def hook(module, inp, out):
        if not ablate["on"]:
            return out
        h = out[0] if isinstance(out, tuple) else out
        r = r_hat[l]
        proj = (h @ r).unsqueeze(-1) * r          # (h . r_hat) r_hat
        h2 = h - proj + hbar[l]                    # Eq.3: ablate refusal dir + patch harmless mean
        return (h2, *out[1:]) if isinstance(out, tuple) else h2
    return hook


handles = [decoder_layers[l].register_forward_hook(make_hook(l)) for l in LAYERS]

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
BS = int(os.environ.get("REFAT_BS", "2"))


def sample_batch(pool):
    picks = [random.choice(pool) for _ in range(BS)]
    toks = [tokenize_pair(i, o) for i, o in picks]
    m = max(len(t[0]) for t in toks)
    input_ids, labels, attn = [], [], []
    for ids, lab in toks:
        pad = m - len(ids)
        input_ids.append(ids + [tok.pad_token_id] * pad)
        labels.append(lab + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return (torch.tensor(input_ids, device="cuda"),
            torch.tensor(labels, device="cuda"),
            torch.tensor(attn, device="cuda"))


model.train()
random.seed(0)
losses = []
for step in range(STEPS):
    if step > 0 and step % K == 0:
        model.eval(); recompute_dirs(NREC); model.train()
    harmful = random.random() < 0.5
    pool = D_r if harmful else D_u
    input_ids, labels, attn = sample_batch(pool)
    ablate["on"] = harmful and (random.random() < P_RFA)     # RFA only on harmful, prob p
    out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
    loss = out.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step(); opt.zero_grad()
    ablate["on"] = False
    losses.append(loss.item())
    if (step + 1) % 25 == 0:
        print(f"[refat] step {step+1}/{STEPS} loss={sum(losses[-25:])/len(losses[-25:]):.4f}", flush=True)

for h in handles:
    h.remove()
os.makedirs(OUT, exist_ok=True)
model.save_pretrained(os.path.join(OUT, "lora_llama_refat"))
print(f"[refat] saved LoRA -> {OUT}/lora_llama_refat")
