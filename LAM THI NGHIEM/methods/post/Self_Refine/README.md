# Self-Refine (safety adapt) — post-processing (P2)

> target trả lời → tự phê bình độ an toàn → viết lại cho an toàn, lặp tối đa k=2 vòng, dừng khi `VERDICT: SAFE`.

## 1. Là gì
Paper: [Self-Refine](https://arxiv.org/pdf/2303.17651) (NeurIPS 2023) · repo: [madaan/self-refine](https://github.com/madaan/self-refine)

**⚠️ Đây là bản ADAPT.** Self-Refine gốc là refinement chung (code/sentiment...), KHÔNG phải jailbreak defense. Ta giữ **nguyên khung** `init → feedback → refine` + early-stop (stop marker `VERDICT: SAFE` = tương đương `"it is correct"` của task GSM), nhưng prompt feedback/refine hướng **safety** (tự viết, khai báo trong `method.py`).

Cơ chế (cùng target `llama-3.1-8b`):
1. **init**: trả lời prompt gốc.
2. **feedback**: same model phê bình response có hại không → ra `VERDICT: SAFE/UNSAFE`.
3. **refine**: viết lại theo feedback (hại → từ chối; lành → giữ, bỏ phần bị flag).
4. Lặp ≤ k=2, dừng ngay khi SAFE.

## 2. Cách chạy
```bash
python method.py response --task harmbench      # 1 + 2k call/prompt (early-stop)
python method.py response --task xstest
python method.py judge    --task xstest
python method.py judge    --task harmbench
```

## 3. Khai báo
| Mục | Giá trị |
|---|---|
| Model target | `llama-3.1-8b-instant` (Groq) |
| Gọi API infer | Có, **1 + 2k call** (k=2 → tối đa 5); temp=0, max_tokens=512 |
| Train | Không |
| Model phụ trợ | Không (feedback/refine = chính target — đúng tinh thần "self") |
| Khác chuẩn | Prompt feedback/refine tự viết cho safety (Self-Refine gốc không có bản jailbreak) |
