# G4D — Dynamic Guided and Domain Applicable Safeguards (pre-processing, P3)

> Method **multi-call** (3 call nếu safe, 4 nếu unsafe). Guidance pipeline sinh ra "intention + safety guidance" rồi mới cho target trả lời.
> **Kết quả (Llama-13b):** ASR **7.0%** · over-refusal (j2) **10.8%** · cost ~4 call (~928 in + 664 out tok)/req. (Utility JustEval: chưa chạy.)

## 1. Là gì
Paper: [G4D](https://aclanthology.org/2025.findings-naacl.368.pdf) · repo: [IDEA-XL/G4D](https://github.com/IDEA-XL/G4D)

Cơ chế (verbatim từ `G4D.py`, `prompt_tools.py`):
1. **intent_detect**: phân tích ý định, gán nhãn safe/unsafe, trích entity.
2. **paraphrase** (chỉ khi unsafe): viết lại câu hỏi.
3. **safety_analyze**: sinh Retrieval-Augmented-Intention + Guidance an toàn.
4. **target**: ghép `[question + intention + guidance]` → target trả lời cuối.

**Retrieval (Wikipedia) TẮT** đúng theo `main.py` (`retrieve=False`) → `knowledge=""`. Prompt/parsers giữ verbatim trong `method.py`.

## 2. Cấu trúc
```
G4D/
├── method.py       # 4 prompt + 3 parser verbatim + g4d_generate (hook generate)
├── requirements.txt
├── outputs/
└── repo/           # clone IDEA-XL/G4D (tham chiếu)
```

## 3. Cách chạy
```bash
python method.py response --task harmbench     # sinh response (3-4 call/prompt)
python method.py response --task xstest
python method.py judge    --task xstest         # over-refusal (API)
python method.py judge    --task harmbench      # ASR (classifier GPU Llama-13b)
python method.py response --task harmbench --limit 5   # smoke test
```
- Key: dùng chung pool `.env` (xoay vòng khi 429).
- **Cost: 3 call/request (safe) hoặc 4 (unsafe)** — đắt hơn IA.

## 4. Khai báo
| Mục | Giá trị |
|---|---|
| Model target | `llama-3.1-8b-instant` (Groq) |
| Gọi API infer | Có, **3-4 call/request**; temp=0, max_tokens=512 |
| Train | Không |
| Model phụ trợ | Không (dùng chính target cho mọi call; gốc dùng gpt-4o-mini) |
| Khác chuẩn | Retrieval TẮT (đúng main.py); các call phụ dùng cùng target thay vì gpt-4o-mini |
