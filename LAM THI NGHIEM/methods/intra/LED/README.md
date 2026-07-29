# LED — Layer-specific Editing

**Nhóm:** intra (sửa trọng số vĩnh viễn) · **Venue:** Findings EMNLP 2024
**Paper:** https://arxiv.org/abs/2405.18166 · **Repo gốc:** https://github.com/ledllm/ledllm (clone trong `repo/`)

## Method làm gì

Sửa trọng số của một **nhóm nhỏ layer đầu/giữa** ("edit layers" E) sao cho các **"toxic layer"
cuối** (T) decode ra câu từ chối cho input harmful — mà không xoá kiến thức. Loss (paper Eq. 4)
dùng **logit-lens (early-exit)**: chiếu hidden state của mỗi layer `t ∈ T` qua final-norm + LM
head rồi tối thiểu NLL của câu từ chối `Y_safe`; gradient chỉ chảy vào layer trong E (đóng băng
phần còn lại).

```
E (edit)  = {4,5,6,13,14,15}      # full-weight, trainable
T (toxic) = {29,30,31}            # nơi tính logit-lens loss
```

## Chạy 2 bước

```bash
python led_build_data.py                    # 200 (harmful -> refusal) pairs -> led_data.json (CPU)
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync python led_train.py   # edit -> led_out/
python method.py response --task all
python method.py judge    --task harmbench
```

## Deviation phải khai báo

- **Layer set E/T** tái dùng chỉ số Llama-2-7B của paper (cùng 32 layer, cùng độ mạnh alignment);
  paper KHÔNG báo chỉ số cho Llama-3. Faithful hơn thì chạy lại pruning-analysis (chưa làm).
- **Edit data:** paper dùng 200 prompt TDC-2023; mình dùng **200 AdvBench** (held-out khỏi
  HarmBench). `Y_safe` = pool 4 refusal template (paper: "desired safe response").
- **Full-weight edit** 6 layer (KHÔNG LoRA) đúng paper. LR/step/optimizer **paper không công bố**
  → chọn: 500 step (fair-cost), AdamW-8bit lr 1e-5.

## Kiến trúc port

Repo gốc chỉ có notebook phân tích (pruning/hidden) + `harmful_prompt.json` **rỗng** → phần
editing phải **reimplement**. `led_train.py`: freeze all trừ E, vòng train thủ công với
loss early-exit logit-lens (Eq. 4). `method.py`: nạp `led_out/` (full model đã sửa) sinh như thường.

## Trạng thái — ✅ CHẠY THẬT (29/07/2026, H100 MIG 40GB)

- **HarmBench ASR = 0.7%** (n=300) — hạ từ 11.0%. Response coherent (từ chối đúng template).
- Train ~110 s (6 layer, 500 step). Infer **3.09 s/req**.
- Over-refusal + utility: ⏳ chờ chấm (judge API TPD-limited).
