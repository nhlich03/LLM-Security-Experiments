# erase-and-check — Certifying LLM Safety (pre-processing DETECTOR, P4)

> Method **detector**: xoá dần token cuối prompt, chạy safety filter trên mỗi biến thể; **bất kỳ** biến thể bị flag → chặn (trả lời từ chối). Không chặn → target trả lời bình thường.
> **Kết quả (Llama-13b):** ASR **14.7%** · over-refusal (j2) **8.4%** · cost ~1.6 call + filter local 0.056s/req. Detector "lỏng": ít từ chối oan (≈ no_defense) nhưng phòng thủ vừa phải. (Utility JustEval: chưa chạy.)

## 1. Là gì
Paper: [Certifying LLM Safety](https://arxiv.org/pdf/2309.02705) · repo: [aounon/certified-llm-safety](https://github.com/aounon/certified-llm-safety)

Cơ chế (verbatim `defenses.py::erase_and_check_suffix`, mode **suffix**, `max_erase=20`):
1. Tokenize prompt bằng tokenizer của filter (bỏ token đặc biệt đầu: `input_ids[1:]`).
2. Ứng viên = prompt gốc + mỗi bản xoá 1..min(max_erase, n) token **từ cuối**.
3. Filter (DistilBERT) chấm từng ứng viên. `LABEL_0 = harmful`, `LABEL_1 = safe`.
4. Flag harmful nếu **bất kỳ** ứng viên harmful → trả về REFUSAL (không gọi target).

**Filter = model phụ trợ chạy LOCAL trên GPU** (không qua Groq): `distilbert-base-uncased` + trọng số fine-tuned `models/distilbert_suffix.pt`. Cost filter = giây (local), chỉ call `target` tính token API.

## 2. Cấu trúc
```
erase-and-check/
├── method.py        # load filter + erase_and_check_suffix verbatim + ec_generate
├── requirements.txt
├── models/          # distilbert_suffix.pt (tải riêng, xem mục 3)
├── outputs/
└── repo/            # clone aounon/certified-llm-safety (tham chiếu)
```

## 3. Cách chạy (cần GPU cho filter + Groq key cho target)
**Tải trọng số filter** (một lần) vào `models/`:
```bash
mkdir -p models && cd models
# models.zip tu Dropbox (link trong README upstream), giai nen lay distilbert_suffix.pt
curl -L -o models.zip "https://www.dropbox.com/scl/fi/ux4ew8y88uslu5064r2xh/models.zip?rlkey=4bo1njpnj4nc801tw1pkby52o&dl=1"
unzip -o models.zip && cd ..
```
Chạy:
```bash
python method.py response --task harmbench     # filter (local GPU) + target (API)
python method.py response --task xstest
python method.py judge    --task xstest         # over-refusal (API)
python method.py judge    --task harmbench      # ASR (classifier GPU Llama-13b)
```
- **Cost:** filter O(n) local (≤21 forward/prompt, batch 1 lần); chỉ prompt không bị chặn mới tốn 1 call target.
- Trên server MIG: đặt `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` (tránh lỗi NVML).

## 4. Khai báo
| Mục | Giá trị |
|---|---|
| Model target | `llama-3.1-8b-instant` (Groq) |
| Gọi API infer | Có, **≤1 call/request** (chỉ khi không bị chặn); temp=0, max_tokens=512 |
| Train | Không (dùng trọng số filter tác giả phát hành) |
| Model phụ trợ | **DistilBERT** filter (local GPU), `distilbert_suffix.pt` |
| Setting | mode=suffix, max_erase=20; `LABEL_0=harmful`; refusal cố định |
| Khác chuẩn | `truncation=True` cho filter (guard DistilBERT ≤512 token) |
