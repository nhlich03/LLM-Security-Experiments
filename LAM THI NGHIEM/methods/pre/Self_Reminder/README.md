# Self-Reminder — pre-processing

**Nhóm:** pre · **Venue:** Nature Machine Intelligence 2023
**Paper:** [Defending ChatGPT against jailbreak attack via self-reminders](https://www.nature.com/articles/s42256-023-00765-8) · **Repo gốc:** https://github.com/yjw1029/Self-Reminder (clone trong `repo/`)

## Method làm gì

Bọc query của người dùng bằng **một lời nhắc an toàn ở cả hai đầu** — một câu dặn trước prompt và một câu nhắc lại sau prompt. Ý tưởng mượn từ tâm lý học: nhắc ai đó về vai trò của họ ngay trước và ngay sau khi giao việc thì họ ít lệch vai hơn.

Không train, không model phụ, không đọc output → **đúng 1 call**, rẻ nhất trong toàn bộ nhóm pre.

Template lấy **verbatim** từ `repo/src/pia_defense.py::defense_templates`. Bản mặc định là index 0 (`Remind`) — biến thể chính của paper.

## Cách chạy

```bash
python method.py response --task harmbench
python method.py response --task xstest
python method.py judge    --task xstest       # over-refusal (API)
python method.py judge    --task harmbench    # ASR (classifier GPU 40GB)
```

Repo có sẵn **7 biến thể** — đây chính là bảng ablation của paper, đổi bằng env:

| `SR_TEMPLATE` | Tên | Khác nhau ở đâu |
|:--:|---|---|
| **0** | remind | *should* — mặc định |
| 1 | warn | *must* thay cho *should* |
| 2 | praise | khen model giỏi tránh nội dung hại |
| 3 | prefix | chỉ nhắc **trước** prompt |
| 4 | suffix | chỉ nhắc **sau** prompt |
| 5 | adaptive1 | thêm "Ignore all the previous/following instructions" |
| 6 | adaptive2 | thêm "Sorry, the above instructions are wrong" |

```bash
SR_TEMPLATE=4 python method.py response --task harmbench    # chỉ suffix
```

So 3 với 4 là tách được đóng góp của vế trước và vế sau — rẻ và đáng chạy nếu cần một ablation cho survey.

## Trạng thái — ✅ ĐÃ CHẠY THẬT (27/07/2026)

| Metric | Số | Mốc `no_defense` |
|---|---:|---:|
| **over-refusal** (XSTest judge2, n=250) | **22.0%** | 8.0% |
| over-refusal (string-match) | 21.6% | 6.4% |
| **ASR** (HarmBench, n=300) | *(đang chấm trên GPU)* | 30.7% |
| Cost | **1.0 call/req** · 101 tok vào / 26 tok ra | 1.0 call · 107/234 |

**Đọc số:** over-refusal **22.0%**, gấp gần 3 lần mốc. Đúng như dự đoán — nó cùng họ "nhắc nhở mạnh" với SAGE (34.8%), chỉ nhẹ hơn. Token ra chỉ **26** so với 234 của no_defense, nghĩa là model từ chối rất ngắn gọn ở phần lớn prompt HarmBench.

## Lưu ý khi báo cáo

- Paper chạy trên **ChatGPT (GPT-3.5)** với bộ jailbreak prompt của họ (`data/jailbreak_prompts.csv`), báo ASR 67% → 19%. Mình chạy trên **`llama-3.1-8b-instant`** với HarmBench prompt **thô, không bọc jailbreak template** → con số không so trực tiếp với paper được.
- Đây là điểm yếu chung của cả nhóm pre kiểu "nhắc nhở": nó được thiết kế để chống **jailbreak template** (DAN, AIM...), mà HarmBench của mình lại là câu hỏi độc trần trụi. Ghi rõ kẻo kết luận sai.
