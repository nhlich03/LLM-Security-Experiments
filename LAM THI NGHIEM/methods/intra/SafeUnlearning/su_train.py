"""Safe Unlearning (arXiv:2407.02855) - TRAINING, ported to single 40GB MIG + Llama-3.

The three-term loss (unlearn harmful via a DPO-style term vs a frozen reference; learn safe
refusals; keep general ability) is used VERBATIM from the repo (ft_code/trainers.py +
data_helper.py). Only the launcher is ours.

DEVIATIONS (declare in report):
  * Full FT (paper, 4xGPU deepspeed) -> LoRA on a single 40GB MIG slice. This matches the
    fair-cost 500-step LoRA protocol used for the other intra methods (CAT/DeRTa/CB).
  * reference_model = base in 4-bit (frozen) instead of a full-precision deepcopy, to fit two
    8B models on one 40GB slice. The reference only supplies log-probs of harmful responses.
  * Fixed 500 optimizer steps (fair-cost budget), not the paper's 3-4 epochs.
  * alpha/beta/theta kept at the paper defaults (0.3 / 1.0 / 0.5).

Run: PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync python su_train.py
"""
import os
import sys
import json
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REPO_FT = os.path.join(HERE, "repo", "ft_code")
sys.path.insert(0, REPO_FT)
sys.path.insert(0, ROOT)

import torch                                                          # noqa: E402
import transformers                                                  # noqa: E402
from transformers import (AutoModelForCausalLM, TrainingArguments,   # noqa: E402
                          default_data_collator, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model                          # noqa: E402
from data_helper import SafetyDatasetDecoderOnly                     # verbatim  # noqa: E402
from trainers import SafeUnlearningTrainer                           # verbatim  # noqa: E402
from core.local_client import resolve_model                          # noqa: E402

BASE = resolve_model("NousResearch/Meta-Llama-3-8B-Instruct", env_var="SU_BASE")
DATA = os.path.join(HERE, "data_llama3")
OUT = os.path.join(HERE, "train_out")
STEPS = int(os.environ.get("SU_STEPS", "500"))
ALPHA = float(os.environ.get("SU_ALPHA", "0.3"))
BETA = float(os.environ.get("SU_BETA", "1.0"))
THETA = float(os.environ.get("SU_THETA", "0.5"))
MAXLEN = int(os.environ.get("SU_MAXLEN", "1024"))
V5 = int(transformers.__version__.split(".")[0]) >= 5
DTYPE_KW = "dtype" if V5 else "torch_dtype"

# ----- Data (verbatim dataset class; args is a light namespace it expects) -----
ds_args = types.SimpleNamespace(max_length=MAXLEN, tokenizer_path=BASE, model_dir=BASE)
train_data = json.load(open(os.path.join(DATA, "train.json"), encoding="utf-8"))
valid_data = json.load(open(os.path.join(DATA, "dev.json"), encoding="utf-8"))
train_ds = SafetyDatasetDecoderOnly(ds_args, train_data, "safe_unlearning")
dev_ds = SafetyDatasetDecoderOnly(ds_args, valid_data, "safe_unlearning")
print(f"[su_train] train={len(train_ds)} dev={len(dev_ds)} maxlen={MAXLEN} steps={STEPS}")

# ----- Target: base bf16 + trainable LoRA -----
model = AutoModelForCausalLM.from_pretrained(
    BASE, use_cache=False, **{DTYPE_KW: torch.bfloat16}).to("cuda")
lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lora)
model.enable_input_require_grads()
model.gradient_checkpointing_enable()
model.print_trainable_parameters()

# ----- Reference: base in 4-bit, frozen (supplies harmful-response log-probs) -----
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
reference = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, device_map={"": 0})
reference.eval()
for p in reference.parameters():
    p.requires_grad_(False)

training_args = TrainingArguments(
    output_dir=OUT,
    learning_rate=2e-5,
    per_device_train_batch_size=int(os.environ.get("SU_BS", "2")),
    gradient_accumulation_steps=int(os.environ.get("SU_GA", "2")),
    max_steps=STEPS,
    warmup_steps=0,
    lr_scheduler_type="linear",
    bf16=True,
    gradient_checkpointing=True,
    adam_beta1=0.9, adam_beta2=0.95,
    logging_steps=10,
    save_strategy="no",
    eval_strategy="no",
    report_to="none",
    remove_unused_columns=False,
    seed=1000,
)

trainer = SafeUnlearningTrainer(
    model=model,
    reference_model=reference,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    data_collator=default_data_collator,
    alpha=ALPHA, beta=BETA, theta=THETA,
)

trainer.train(resume_from_checkpoint=False)
os.makedirs(OUT, exist_ok=True)
model.save_pretrained(os.path.join(OUT, "lora_llama_safeunlearning"))
print(f"[su_train] saved LoRA -> {OUT}/lora_llama_safeunlearning")
