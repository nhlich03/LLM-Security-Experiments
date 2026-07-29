# Jailbreak Antidote — Sparse Representation Adjustment

**Nhóm:** in (dịch hidden state lúc decode, trọng số không đổi) · **Venue:** ICLR 2025
**Paper:** https://arxiv.org/abs/2410.02298 · **Repo gốc:** không có → **reimplement từ paper**

## Method làm gì

Thông tin an toàn nằm ở một **hướng thưa (sparse) trong residual stream**. Lúc sinh, cộng một
"safety direction" đã sparsify + scale vào hidden state của các decoder layer (paper Eq. 5):

```
h'_l = h_l + alpha * (d_safe_l ⊙ mask_l)
```

Một scalar `alpha` trượt model trên trục an toàn↔hữu ích (dương = an toàn hơn). Không thêm token,
không train. `d_safe_l` = **PC1 (PCA thành phần chính)** của hidden state token cuối trên tập
harmful+harmless; `mask_l` giữ **top-5%** chiều |lớn nhất|.

## Chạy 2 bước — bước 1 BẮT BUỘC

```bash
# Bước 1 (một lần / model): dựng direction, ghi vectors/llama3/antidote_directions.pt
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync python calibrate.py
# Bước 2: sinh response + chấm
python method.py response --task all
python method.py judge    --task harmbench   # (xstest / justeval sau)
```

Knob: `ANTIDOTE_ALPHA` (mặc định 0.4), `ANTIDOTE_LAYERS` (mặc định `all` = 32 layer).

## Deviation phải khai báo

- **Direction data:** paper dùng Phan (2023) `harmful_harmless_instructions`; mình dùng **AdvBench
  (harmful) + Alpaca-eval (harmless)**, đều public và **held-out khỏi HarmBench eval** (không leak).
  Cơ chế (PC1 + mask top-5% + hook cộng) giữ nguyên.
- **alpha = 0.4**, all 32 layer: paper cho Llama-3-8B khoảng `[-0.6, 0.6]` nhưng không chốt 1 giá
  trị + không liệt kê layer set → đây là lựa chọn của mình (khai báo). Muốn vẽ đường trade-off thì
  quét `alpha ∈ [0, 0.6]`.

## Kiến trúc port

`calibrate.py` (dựng direction) + `method.py` (`AntidoteClient(LocalClient)` gắn forward hook mỗi
layer cộng `alpha·dir`). Không có code gốc để bám → viết theo Eq. 1–5 của paper.

## Trạng thái — ✅ CHẠY THẬT (29/07/2026, H100 MIG 40GB)

- **HarmBench ASR = 1.3%** (n=300) — hạ từ 11.0% (no_defense_local). Refusal sạch, coherent.
- Infer **2.80 s/req** (combined HB+XS+JE). calibrate ~2.5 phút.
- Over-refusal (XSTest) + utility (JustEval): ⏳ chờ chấm (judge API TPD-limited, chạy local sau).
