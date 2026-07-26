# WildGuard — post-processing detector

**Nhóm:** post · **Venue:** NeurIPS 2024 Datasets & Benchmarks
**Paper:** https://arxiv.org/abs/2406.18495 · **Repo gốc:** https://github.com/allenai/wildguard
**Model:** `allenai/wildguard` (Mistral-7B-v0.3 fine-tuned)

## Method làm gì

Target sinh response bình thường → đưa **cặp (prompt, response)** vào WildGuard → nó trả về **ba nhãn cùng một lúc**:

| Nhãn | Giá trị |
|---|---|
| `prompt_harmfulness` | harmful / unharmful |
| `response_refusal` | refusal / compliance |
| `response_harmfulness` | harmful / unharmful ← dùng để chặn |

`response_harmfulness == harmful` → vứt response, thay bằng câu từ chối.

**Điểm riêng của bài này:** nó là guard duy nhất tự cho luôn **cả hai trục** mình đang đo (harmfulness + refusal) trong một lần gọi — không cần judge riêng. Nhãn `response_refusal` đối chiếu được thẳng với XSTest judge của mình.

## ⚠️ Vì sao KHÔNG dùng `pip install wildguard`

Package của họ pin **`vllm>=0.4.2` là hard dependency**, kể cả khi gọi `load_wildguard(use_vllm=False)`.

Đã thử cài vào venv chung và **nó phá hỏng thật**: `torch 2.6.0+cu124` → `2.11.0+cu130`, `torch.cuda.is_available()` thành **False**, kéo theo cả 5 method local khác chết. Phải gỡ vllm và force-reinstall torch cu124 mới khôi phục được.

→ `method.py` nạp thẳng model bằng `transformers`, nhưng **prompt format và parser lấy verbatim từ chính source của họ**:

| Thứ | Nguồn |
|---|---|
| `WILDGUARD_INPUT_FORMAT` | `wildguard/utils.py` |
| regex parse 3 nhãn | `wildguard/wildguard.py::parse_model_generation_to_output` |

Đây đúng là đường `use_vllm=False` của họ, chỉ khác chỗ không đi qua lớp package. Sai khác **phải khai báo**: engine sinh của guard là HF `generate()` chứ không phải vLLM → thời gian không so được với số của paper (nhưng nhãn thì giống, vì greedy decode).

## Cách chạy

Cần **cả GPU** (guard 7B) **lẫn Groq key** (target) → chạy trên server. Giống hệt erase-and-check.

```bash
python method.py response --task harmbench
python method.py response --task xstest
python method.py judge    --task xstest
python method.py judge    --task harmbench
```

Lần đầu tự tải `allenai/wildguard` (~15GB).

## Cấu hình

- Target: `llama-3.1-8b-instant` (Groq) — giữ đúng quy ước để so với 4 bài post kia
- Guard: `allenai/wildguard` local, bf16, greedy, `max_new_tokens=128` (theo upstream)
- Cost: **1 call API** (target) + **giây GPU** (guard) → là bài post duy nhất tốn cả hai loại

## Trạng thái — 🔧 ĐÃ CODE, đang smoke test trên server

## Lưu ý khi báo cáo

- **Chuỗi từ chối là phần mình tự thêm** — repo chỉ là classifier. Dùng đúng chuỗi của `Self_Defense` và `SelfDefend` để 3 bài post so sánh được.
- **Fail-open khi parse lỗi.** Nếu output của guard không khớp regex 3 dòng, `method.py` **giữ nguyên response** thay vì chặn. Chọn fail-open để không thổi phồng số over-refusal; phải đếm xem có bao nhiêu ca như vậy khi chạy full.
- Guard là **Mistral-7B**, khác họ với target Llama — đúng như paper, không phải sai khác của mình.
