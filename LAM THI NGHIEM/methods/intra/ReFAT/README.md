# ReFAT — Refusal Feature Adversarial Training

**Nhóm:** intra (sửa trọng số vĩnh viễn, LoRA) · **Venue:** ICLR 2025
**Paper:** https://arxiv.org/abs/2409.20089 · **Repo gốc:** không có → **reimplement từ paper**

## Method làm gì

Fine-tune LoRA để model từ chối harmful + giữ hữu ích, NHƯNG trong một phần forward của batch
harmful thì **chiếu bỏ "refusal feature"** khỏi residual stream (mô phỏng tấn công worst-case):

```
h'_l = h_l - (h_l·r̂_l) r̂_l + mean_harmless_l        # Eq.3, layer l ∈ [8,31], prob p_RFA=0.5
```

`r̂_l` = hướng refusal per-layer = (mean harmful − mean harmless) của hidden token cuối, chuẩn hoá.
Tính lại mỗi **k=4 step** (vì không gian activation trôi khi train). Loss = NLL trên (harmful→refuse)
+ (benign→help); nhánh benign KHÔNG bao giờ bị ablate.

## Chạy 2 bước

```bash
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync python refat_train.py   # -> train_out/lora_llama_refat
python method.py response --task all
python method.py judge    --task harmbench
```

Knob: `REFAT_STEPS` (500), `REFAT_P` (0.5), `REFAT_K` (4), `REFAT_L0` (8), `REFAT_BS` (2).

## Deviation phải khai báo

- **LoRA r=128 α=32** (paper cũng LoRA r=128) — không lệch. Target layer **8–31** theo Table 4 (Llama-3).
- **500 step** (fair-cost) thay ~313 step/1 epoch của paper.
- **Data thay thế:** paper dùng Zou-2024 refusals + UltraChat (5k+5k) + 150 XSTest; mình tái dùng
  data có sẵn: D_r = SafeUnlearning safe-refusal + LED (harmful→refusal), D_u = SafeUnlearning
  benign→GPT-4 (UltraFeedback). Direction từ AdvBench(500) + Alpaca(500).
- Ablate ở **mọi vị trí** của target layer (paper chỉ rõ token cuối để *tính* hướng, không nói rõ
  vị trí lúc train) — khai báo.

## Kiến trúc port

Không có repo → reimplement. `refat_train.py`: LoRA + forward hook trên layer 8–31 (bật/tắt theo cờ
`ablate`), tính lại hướng mỗi 4 step, vòng train thủ công interleave D_r/D_u. `method.py`: nạp base +
adapter. Cơ chế trích hướng theo Arditi 2024 (`andyrdt/refusal_direction`).

## Trạng thái — ✅ CHẠY THẬT (29/07/2026, H100 MIG 40GB)

- **HarmBench ASR = 6.3%** (n=300) — hạ từ 11.0% (yếu nhất trong 5 bài mới; ở liều 500-step LoRA,
  paper train nhiều hơn — khai báo caveat fair-cost).
- Train ~485 s (LoRA r128 + RFA ablation + recompute dir mỗi 4 step). Infer **5.46 s/req**.
- Over-refusal + utility: ⏳ chờ chấm (judge API TPD-limited).
