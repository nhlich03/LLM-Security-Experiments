# Backtranslation — post-processing (P2)

> Sinh response → suy ngược ra prompt từ response → chạy lại target trên prompt đó; nếu prompt suy ngược bị từ chối → response gốc là hại → trả từ chối.

## 1. Là gì
Paper: [Defending LLMs via Backtranslation](https://arxiv.org/pdf/2402.16459) (Findings ACL 2024) · repo: [YihanWang617/LLM-Jailbreaking-Defense-Backtranslation](https://github.com/YihanWang617/LLM-Jailbreaking-Defense-Backtranslation)

Cơ chế (verbatim `defenses/backtranslation.py`, cùng target `llama-3.1-8b`):
1. **target**: sinh response cho prompt gốc.
2. Nếu response **đã là từ chối** → trả REFUSAL luôn (short-circuit).
3. **backtranslate**: 1 call suy ngược prompt từ response (lộ intent thật).
4. **recheck**: chạy target trên prompt backtranslate.
5. recheck **không** bị từ chối → intent lành → trả **response gốc**; ngược lại → REFUSAL.

`check_rejection` (string-match) + chuỗi REFUSAL giữ **verbatim** từ repo. **Bỏ** bộ lọc likelihood (cần logprobs, API chat không có) = `threshold -inf`.

## 2. Cách chạy
```bash
python method.py response --task harmbench      # 1-3 call/prompt
python method.py response --task xstest
python method.py judge    --task xstest
python method.py judge    --task harmbench
```

## 3. Khai báo
| Mục | Giá trị |
|---|---|
| Model target | `llama-3.1-8b-instant` (Groq) |
| Gọi API infer | Có, **1-3 call/request** (target + backtranslate + recheck) |
| Train | Không |
| Model phụ trợ | Không (backtranslator = chính target) |
| Khác chuẩn | Bỏ likelihood filter (`threshold=-inf`) vì API không có logprobs |
