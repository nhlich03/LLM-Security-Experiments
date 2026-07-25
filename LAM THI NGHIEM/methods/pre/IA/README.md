# IA — Intention Analysis (pre-processing, P1)

> Method **multi-call** đầu tiên: 2 call/request (khác SAGE 1-call).
> **Kết quả (Llama-13b):** ASR (HarmBench) = **2.0%** · over-refusal (XSTest judge2) = **12.4%** · cost ~2 call/req. (Utility JustEval: chưa chạy.)

## 1. Là gì
Paper: [Intention Analysis](https://arxiv.org/pdf/2401.06561) · repo: [alphadl/SafeLLM_with_IntentionAnalysis](https://github.com/alphadl/SafeLLM_with_IntentionAnalysis)

Cơ chế 2 bước (inference-only), verbatim từ `repo/demo/IA_demo.py`:
1. **Intention**: hỏi target phân tích *ý định cốt lõi* của query (không trả lời) — `IA_PROMPT`.
2. **Target**: trong **cùng hội thoại**, đưa kết quả bước 1 + `CT_PROMPT` → target trả lời cuối.

Prompt lõi (`IA_PROMPT`, `CT_PROMPT`) giữ verbatim trong `method.py`.

## 2. Cấu trúc
```
IA/
├── method.py       # ia_generate (2 call) + gọi core.runner (hook generate)
├── requirements.txt
├── outputs/
└── repo/           # clone alphadl/SafeLLM_with_IntentionAnalysis (tham chiếu)
```

## 3. Cách chạy
```powershell
conda activate ia
cd "E:\DHBK\JAILBREAK\LAM THI NGHIEM\methods\pre\IA"
python method.py response --task all        # sinh response (2 call/prompt) cho 3 tập
python method.py judge    --task xstest        # over-refusal (API)
python method.py judge    --task justeval      # utility (API)
python method.py judge    --task harmbench     # ASR (classifier GPU 40GB) / hoặc Kaggle
python method.py response --task harmbench --limit 5   # smoke test
```
- Key: dùng chung pool `.env` (xoay vòng khi 429).
- **Cost: 2 call/request** (`intention` + `target`) — đắt gấp đôi single-call.

## 4. Khai báo
| Mục | Giá trị |
|---|---|
| Model target | `llama-3.1-8b-instant` (Groq) |
| Gọi API infer | Có, **2 call/request** (intention + target); temp=0, max_tokens=512 |
| Train | Không |
| Model phụ trợ | Không (dùng chính target cho cả 2 bước) |
