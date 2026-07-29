# Safe Unlearning

**Nhóm:** intra (sửa trọng số vĩnh viễn) · **Venue:** arXiv 2024 (thu-coai)
**Paper:** https://arxiv.org/abs/2407.02855 · **Repo gốc:** https://github.com/thu-coai/SafeUnlearning (clone trong `repo/`)

## Method làm gì

Fine-tune với **3 loss** đồng thời (giữ VERBATIM từ `repo/ft_code/trainers.py`):
- **unlearn** phản hồi harmful (số hạng kiểu DPO-sigmoid so với reference model đóng băng, hệ số α),
- **learn** câu từ chối an toàn (θ),
- **maintain** năng lực chung trên data benign.

Data đi kèm theo `type`: 0=benign→helpful, 1=harmful-response (unlearn), 2=safe-refusal (learn).
GroupRandomSampler trộn nhóm 5:5:1.

## Chạy 3 bước

```bash
python convert_data.py                       # re-template vicuna -> llama3 -> data_llama3/ (CPU)
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync python su_train.py   # -> train_out/lora_llama_safeunlearning
python method.py response --task all
python method.py judge    --task harmbench
```

Knob: `SU_BS` (mặc định 4), `SU_GA` (3), `SU_STEPS` (500), α/β/θ = `SU_ALPHA`/`SU_BETA`/`SU_THETA` (0.3/1.0/0.5).

## Deviation phải khai báo

- **LoRA + reference 4-bit** thay full-FT 4-GPU deepspeed của paper (để vừa 1 MIG 40GB). Khớp
  protocol 500-step LoRA của CAT/DeRTa/CB. `batch 4 × ga 3 = eff 12` (≈ nhóm 5:5:1) × 500 step ≈ 5 epoch (~= paper).
- Data re-template vicuna→Llama-3 (nội dung giữ nguyên).
- **4 fix tương thích transformers v5 / numerical (khai báo — không đổi ngữ nghĩa loss):**
  1. `_get_train_sampler(self, *args, **kwargs)` (v5 truyền dataset).
  2. `compute_loss(..., num_items_in_batch=None, **kwargs)` (v5 truyền thêm arg).
  3. guard `get_subbatch_loss`: subgroup rỗng → 0 thay vì `nan` (batch nhỏ + grad-accum).
  4. `compute_logps`: `log_softmax` ở **fp32** + mẫu số `.clamp(min=1)` (chống nan ở số hạng DPO).

## Kiến trúc port

Loss (`trainers.py`) + dataset (`data_helper.py`) giữ **verbatim** — chỉ viết launcher `su_train.py`
(LoRA + reference 4-bit + 500 step) và `method.py` (nạp base + adapter sinh như thường).

## Trạng thái — ✅ CHẠY THẬT (29/07/2026, H100 MIG 40GB)

- **HarmBench ASR = 4.0%** (n=300) — hạ từ 11.0%. Utility giữ tốt (câu benign dài, coherent).
- Train ~1615 s (500 step, eff-batch 12). Infer **5.44 s/req**.
- Over-refusal + utility: ⏳ chờ chấm (judge API TPD-limited).
