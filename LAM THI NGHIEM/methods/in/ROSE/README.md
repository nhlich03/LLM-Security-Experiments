# ROSE — Reverse Prompt Contrastive Decoding

**Nhóm:** in (can thiệp logits lúc decode, training-free) · **Venue:** Findings/ACL-era 2024
**Paper:** https://arxiv.org/abs/2402.11889 · **Repo gốc:** clone trong `repo/` (tham khảo)

## Method làm gì

Contrastive decoding giữa 2 lần forward khác nhau **chỉ ở system prompt** (paper Eq. 1):

```
adjusted_logits = logit(y | POS, x, y<t)  -  alpha * logit(y | REV, x, y<t)
```

- **POS** = system prompt an toàn (bản Llama-2 chuẩn).
- **REV** = system prompt "đảo" (unhelpful/dishonest/ignore-constraints) — trừ đi để **triệt tiêu
  cái model sẽ nói nếu nó mất an toàn**.
- Trừ trên **raw logits** rồi softmax 1 lần, greedy. **Không** có adaptive-plausibility threshold.

Không train, không đổi trọng số → **in**. Chi phí ≈ **2× forward/token** (2 KV cache song song).

## Chạy (không cần bước train)

```bash
python method.py response --task harmbench   # ASR trước (refusal ngắn, nhanh)
python method.py judge    --task harmbench
python method.py response --task xstest
python method.py response --task justeval
```

Knob: `ROSE_ALPHA` (mặc định 0.5 — giá trị generation của paper).

## Deviation phải khai báo

- **alpha = 0.5**: default generation của paper, **chưa tune cho Llama-3** (paper test
  Alpaca/Vicuna/Qwen/InternLM). Có thể quét 0.3–0.7 trên dev slice.
- Prompt POS/REV inject qua **role `system`** của chat template Llama-3 (paper dùng model cũ nối
  system dạng plain text) — thích ứng, khai báo.
- Trừ trên **raw logits** đúng Eq. 1 (không phải log-prob như CD cổ điển).

## Kiến trúc port

`method.py`: `rose_generate(client, raw, meter)` — vòng decode thủ công với 2 `past_key_values`
(POS + REV), mỗi bước cộng token vừa chọn vào CẢ hai. Prompt POS/REV lấy verbatim từ Appendix
Table 6 của paper.

## Trạng thái — ✅ CHẠY THẬT (29/07/2026, H100 MIG 40GB)

- **HarmBench ASR = 0.3%** (n=300) — mạnh nhất trong 5 bài mới (từ 11.0%). Refusal coherent.
- Infer chậm do 2× decode (đọc kèm token/req).
- Over-refusal + utility: ⏳ đang sinh XS/JE response → chấm sau (judge API TPD-limited).
