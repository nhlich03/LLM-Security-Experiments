"""LED (Layer-specific Editing, Findings EMNLP 2024, arXiv:2405.18166) - EDITING stage.

Faithful reimplementation of Eq. 4: for each TOXIC layer t in T, project its hidden state
through the shared final-norm + LM head (logit lens) and minimise the NLL of the safe refusal
Y_safe. Only the weights of the EDIT layers E are trainable (everything else frozen), so
gradients from the late toxic layers flow back and edit the early/middle layers -- exactly the
paper's mechanism ("edit early layers so the late toxic layers decode to refusals").

Layer sets (Sec 5.1, reused from Llama-2-7B which has the same 32-layer depth + strong alignment):
  EDIT  E = {4,5,6,13,14,15}   TOXIC T = {29,30,31}

DEVIATIONS (declare): E/T reused from Llama-2-7B (paper reports no Llama-3 indices);
TDC->AdvBench + templated Y_safe (see led_build_data.py); optimizer/LR/steps are our choices
(paper reports none) -- fixed 500-step budget, AdamW-8bit lr 1e-5, to match the fair-cost setup.
FULL-weight edit of E layers (no LoRA), as the paper specifies.

Run: PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync python led_train.py
"""
import os
import sys
import json
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

import torch                                                    # noqa: E402
import torch.nn.functional as F                                 # noqa: E402
import transformers                                             # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer    # noqa: E402
import bitsandbytes as bnb                                      # noqa: E402
from core.local_client import resolve_model                     # noqa: E402

BASE = resolve_model("NousResearch/Meta-Llama-3-8B-Instruct", env_var="LED_BASE")
EDIT = [int(x) for x in os.environ.get("LED_EDIT", "4,5,6,13,14,15").split(",")]
TOXIC = [int(x) for x in os.environ.get("LED_TOXIC", "29,30,31").split(",")]
STEPS = int(os.environ.get("LED_STEPS", "500"))
LR = float(os.environ.get("LED_LR", "1e-5"))
MAXLEN = int(os.environ.get("LED_MAXLEN", "512"))
OUT = os.path.join(HERE, "led_out")
V5 = int(transformers.__version__.split(".")[0]) >= 5
DTYPE_KW = "dtype" if V5 else "torch_dtype"

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

# ----- tokenize pairs: labels = -100 on prompt, real ids on Y_safe -----
data = json.load(open(os.path.join(HERE, "led_data.json"), encoding="utf-8"))
samples = []
for d in data:
    inp = tok(d["input"], add_special_tokens=False)["input_ids"]
    out = tok(d["output"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (inp + out)[:MAXLEN]
    labels = ([-100] * len(inp) + out)[:MAXLEN]
    samples.append((ids, labels))
print(f"[led] {len(samples)} samples | EDIT={EDIT} TOXIC={TOXIC} steps={STEPS}")

model = AutoModelForCausalLM.from_pretrained(
    BASE, use_cache=False, **{DTYPE_KW: torch.bfloat16}).to("cuda")
model.train()
for p in model.parameters():
    p.requires_grad_(False)
trainable = []
for i in EDIT:
    for p in model.model.layers[i].parameters():
        p.requires_grad_(True)
        trainable.append(p)
n_tr = sum(p.numel() for p in trainable)
print(f"[led] trainable params (full weights of {len(EDIT)} layers): {n_tr/1e6:.0f}M")

opt = bnb.optim.PagedAdamW8bit(trainable, lr=LR)
norm, lm_head = model.model.norm, model.lm_head

random.seed(0)
order = []
losses = []
for step in range(STEPS):
    if not order:
        order = list(range(len(samples)))
        random.shuffle(order)
    ids, labels = samples[order.pop()]
    input_ids = torch.tensor([ids], device="cuda")
    label_ids = torch.tensor([labels], device="cuda")

    out = model(input_ids=input_ids, output_hidden_states=True)
    hs = out.hidden_states                       # tuple len L+1; hs[k] = output of block k-1
    loss = 0.0
    for t in TOXIC:
        logits_t = lm_head(norm(hs[t + 1]))      # logit lens (Eq. 4)
        loss = loss + F.cross_entropy(
            logits_t[:, :-1].reshape(-1, logits_t.size(-1)),
            label_ids[:, 1:].reshape(-1), ignore_index=-100)
    loss = loss / len(TOXIC)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step()
    opt.zero_grad()
    losses.append(loss.item())
    if (step + 1) % 25 == 0:
        print(f"[led] step {step+1}/{STEPS} loss={sum(losses[-25:])/len(losses[-25:]):.4f}", flush=True)

os.makedirs(OUT, exist_ok=True)
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print(f"[led] saved edited model -> {OUT}")
