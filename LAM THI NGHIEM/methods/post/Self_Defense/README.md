# LLM Self Defense (Self-Examination) — post-processing (P2)

> Method **post** đầu tiên: target sinh response rồi **tự soi** response đó có hại không; có hại → thay bằng từ chối.

## 1. Là gì
Paper: [LLM Self Defense](https://arxiv.org/pdf/2308.07308) · repo: [poloclub/llm-self-defense](https://github.com/poloclub/llm-self-defense)

Cơ chế (2 call, cùng target `llama-3.1-8b`):
1. **target**: sinh response `T` cho prompt gốc.
2. **self_check**: đưa `T` vào template hỏi "Does this describe harmful behaviour?" (verbatim từ `harm_filter.py`) → examiner trả lời `yes/no ... harmful`.
3. Nếu "yes ... harmful" → trả **REFUSAL**; ngược lại giữ `T`.

**Khác repo gốc:** repo chỉ là batch classifier (đọc CSV, ghi nhận xét), **không** có chuỗi refusal → `REFUSAL` là phần tự thêm (đã khai báo trong `method.py`). Examiner chỉ nhận **response**, không nhận prompt gốc (soi output, không soi attack).

## 2. Cách chạy
```bash
python method.py response --task harmbench     # sinh (2 call/prompt)
python method.py response --task xstest
python method.py judge    --task xstest          # over-refusal (API)
python method.py judge    --task harmbench       # ASR (classifier GPU Llama-13b)
```

## 3. Khai báo
| Mục | Giá trị |
|---|---|
| Model target | `llama-3.1-8b-instant` (Groq) |
| Gọi API infer | Có, **2 call/request** (target + self_check); temp=0, max_tokens=512 |
| Train | Không |
| Model phụ trợ | Không (examiner = chính target) |
| Khác chuẩn | Chuỗi REFUSAL tự thêm (repo gốc không có) |
