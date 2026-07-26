"""
SafeDecoding - train an expert LoRA for a base model that upstream does not ship one for
(e.g. any Llama-3). Upstream provides experts for vicuna / llama2 / guanaco / falcon /
dolphin only; everything else needs this.

Upstream recipe (repo/exp/finetune.py):
  1. take 36 harmful queries covering 18 categories (Ganguli et al. 2022),
  2. let the TARGET model generate 2 refusals each (top_p=0.9, temperature=0.8),
  3. keep only the ones GPT-4 confirms are genuine refusals -> <= 72 pairs,
  4. LoRA fine-tune the target on those pairs: r=16, alpha=64, dropout=0.1,
     2 epochs, batch 1, lr=2e-3, max_seq_len=2048.

Two modes here:

  --reuse   (default) Take the 72 already-verified pairs that ship inside
            repo/lora_modules/<src>/ft_datasets_<src>.json, strip the llama-2 [INST] markup
            and re-render them with the TARGET tokenizer's own chat template. No GPT-4 key,
            no generation step. The refusal texts were written by Llama-2 rather than by the
            target model, which is a documented deviation from step 2 - state it in the report.

  --self-generate   Faithful to steps 1-2: the target model writes its own refusals. Step 3
            (GPT-4 verification) is replaced by a string-match filter, since we do not use
            an OpenAI key anywhere else in this project. Also a documented deviation.

Cost: upstream reports under one minute per model. 72 samples is nothing.

Usage:
  python train_expert.py --base meta-llama/Meta-Llama-3-8B-Instruct --out experts/llama3
  python train_expert.py --base ... --self-generate
Then point method.py at it:
  SD_BASE=meta-llama/Meta-Llama-3-8B-Instruct SD_EXPERT=experts/llama3 python method.py response
"""

import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "repo")

# ----- Upstream LoRA hyperparameters (repo/exp/finetune.py defaults) -----
LORA_R = 16
LORA_ALPHA = 64
LORA_DROPOUT = 0.1
EPOCHS = 2
BATCH_SIZE = 1
GRAD_ACCUM = 1
LR = 2e-3
MAX_SEQ_LEN = 2048
WARMUP_RATIO = 0.03
MAX_GRAD_NORM = 0.3
LR_SCHEDULER = "linear"

REFUSAL_MARKERS = ("i cannot", "i can't", "i'm sorry", "i am sorry", "i apologize",
                   "as a responsible", "i'm not able", "i am not able", "cannot fulfill",
                   "can't fulfill", "not appropriate")


# ----- Mode 1: reuse upstream's verified pairs, re-render with the target template -----
def load_upstream_pairs(src="llama2"):
    path = os.path.join(REPO, "lora_modules", src, f"ft_datasets_{src}.json")
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    pairs = []
    for r in rows:
        t = r["text"] if isinstance(r, dict) else str(r)
        m = re.search(r"\[INST\](.*?)\[/INST\](.*)", t, flags=re.S)
        if not m:
            continue
        pairs.append((m.group(1).strip(), m.group(2).strip()))
    if not pairs:
        raise RuntimeError(f"Parsed 0 pairs from {path} - upstream format changed?")
    return pairs


# ----- Mode 2: the target model writes its own refusals (upstream steps 1-2) -----
def self_generate_pairs(base, num_trials=2, max_trials=5):
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
    from core.local_client import LocalClient

    seed_path = os.path.join(REPO, "datasets", "seed_reject.json")
    with open(seed_path, encoding="utf-8") as f:
        seeds = json.load(f)
    queries = [s["prompt"] if isinstance(s, dict) else str(s) for s in seeds]
    print(f"[gen] {len(queries)} seed harmful queries from {os.path.basename(seed_path)}")

    client = LocalClient(base, temperature=0.8, max_tokens=256)
    pairs = []
    for q in queries:
        kept = 0
        for _ in range(max_trials):
            if kept >= num_trials:
                break
            text, _ = client.chat(q, temperature=0.8, top_p=0.9, min_new_tokens=10)
            # Upstream verifies with GPT-4; we string-match instead (documented deviation).
            if any(m in text.lower() for m in REFUSAL_MARKERS):
                pairs.append((q, text.strip()))
                kept += 1
        print(f"  {kept}/{num_trials} refusals kept for: {q[:60]}...", flush=True)
    client.free()
    return pairs


# ----- LoRA fine-tune -----
def train(base, pairs, out_dir):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Render every pair with the TARGET model's own chat template so the expert learns the
    # format it will actually be decoded in.
    texts = []
    for q, a in pairs:
        if tok.chat_template:
            texts.append(tok.apply_chat_template(
                [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                tokenize=False))
        else:
            texts.append(f"[INST] {q} [/INST] {a}")
    ds = Dataset.from_dict({"text": texts})
    print(f"[train] {len(ds)} samples | base={base}")

    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False

    peft_cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                          bias="none", task_type="CAUSAL_LM")
    targs = TrainingArguments(
        output_dir=os.path.join(out_dir, "_trainer"),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        max_grad_norm=MAX_GRAD_NORM,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        logging_steps=10,
        optim="adamw_torch",
        bf16=True,
        report_to=[],
    )
    # TRL renamed these across versions; try the new signature, fall back to the old one.
    try:
        trainer = SFTTrainer(model=model, args=targs, train_dataset=ds,
                             peft_config=peft_cfg, processing_class=tok)
    except TypeError:
        trainer = SFTTrainer(model=model, args=targs, train_dataset=ds,
                             peft_config=peft_cfg, tokenizer=tok,
                             dataset_text_field="text", max_seq_length=MAX_SEQ_LEN)

    t0 = time.perf_counter()
    trainer.train()
    secs = time.perf_counter() - t0

    os.makedirs(out_dir, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"base": base, "n_samples": len(ds), "train_sec": round(secs, 1),
                   "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "epochs": EPOCHS,
                   "lr": LR}, f, indent=2)
    print(f"\n[done] expert saved to {out_dir} | train_sec={secs:.1f}")
    print(f"Use it:  SD_BASE={base} SD_EXPERT={out_dir} python method.py response --limit 5")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base model to build an expert for")
    ap.add_argument("--out", default=os.path.join(HERE, "experts", "custom"))
    ap.add_argument("--self-generate", action="store_true",
                    help="target writes its own refusals (upstream steps 1-2)")
    ap.add_argument("--reuse-src", default="llama2",
                    help="which shipped ft_datasets_<src>.json to reuse (default: llama2)")
    a = ap.parse_args()

    pairs = (self_generate_pairs(a.base) if a.self_generate
             else load_upstream_pairs(a.reuse_src))
    print(f"[data] {len(pairs)} (query, refusal) pairs")
    train(a.base, pairs, a.out)
