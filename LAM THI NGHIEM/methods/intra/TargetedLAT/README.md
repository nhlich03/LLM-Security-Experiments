# Targeted LAT — Targeted Latent Adversarial Training

**Nhóm:** intra · **Venue:** arXiv 2407.15549 (2024)
**Repo gốc:** https://github.com/aengusl/latent-adversarial-training · **Checkpoint:** org HF [`LLM-LAT`](https://huggingface.co/LLM-LAT)

## Method làm gì

Adversarial training nhưng nhiễu đặt ở **không gian ẩn** (residual-stream activation giữa các layer) thay vì token rời rạc. Cùng tinh thần "né search rời rạc cho rẻ" với CAT, khác chỗ đặt nhiễu:

| Bài | Nhiễu đặt ở đâu | Giá |
|---|---|---|
| R2D2 | token rời rạc (GCG) | đắt nhất |
| CAT | **embedding đầu vào** | rẻ |
| **LAT** | **activation giữa mạng** | rẻ |

Có đủ 3 bài này thì bảng INTRA phủ trọn trục adversarial training. Paper báo RT-EAT-LAT tốn **~36× ít GPU hour hơn R2D2** (không có số tuyệt đối).

Intra → **inference overhead = 0**.

## Checkpoint

Mặc định `LLM-LAT/robust-llama3-8b-instruct` — **full merged weights** (7 shard), ~900 lượt tải, và là bản mà **paper DeepRefusal benchmark dưới tên "LAT"** trong Table 1 → dùng để đối chiếu chéo.

Org còn có `llama3-8b-instruct-{lat,rt}-jailbreak-robust{1,2,3}`; các bản `lat-*` là **LoRA adapter, 0 lượt tải** → coi là phụ.

```bash
python method.py response --task harmbench
python method.py response --task xstest
python method.py judge    --task xstest
python method.py judge    --task harmbench

LAT_MODEL=LLM-LAT/llama3-8b-instruct-rt-jailbreak-robust1 python method.py response --task harmbench
```

## ⚠️ Repo toàn notebook — không vendor `repo/`

Upstream **không có CLI**, toàn bộ là notebook trong `/notebooks` (latent space attack, jailbreak robustness, backdoor removal, unlearning). Vì vậy thư mục này **chỉ tiêu thụ trọng số đã phát hành**, không clone repo vào.

Muốn train lại thì phải **bóc code ra khỏi notebook** — chưa làm. Đây là lý do LAT xếp sau DeepRefusal trong thứ tự ưu tiên dù cả hai đều có checkpoint sẵn.

## Trạng thái — ✅ ĐÃ CHẠY FULL (27/07/2026, H100 MIG 40GB)

| Metric | Targeted LAT | DeepRefusal | mốc `no_defense` (API) |
|---|---:|---:|---:|
| **ASR** (HarmBench, n=300) | **1.0%** | 0.0% | 30.7% |
| **over-refusal** (XSTest judge2, n=250) | **35.6%** | 55.2% | 8.0% |
| local_sec (smoke n=3) | 1.557 ± 0.885 | 1.287 ± 0.233 | — |
| token ra (smoke) | 70 ± 24 | 57 | — |

**Đây là đánh đổi tốt nhất trong 2 bài intra đã chạy full.** LAT trả thêm **1 điểm ASR** để tiết kiệm **gần 20 điểm over-refusal** so với DeepRefusal. Nếu phải chọn một bài intra triển khai thật thì đây là bài hợp lý hơn.

Response sạch, từ chối rõ ràng, **không sinh chuỗi vô nghĩa** như Circuit Breakers:

> *"I'm sorry, but I cannot provide instructions on how to make dimethylmercury or any other hazardous or illegal…"*

### ✅ Đối chiếu chéo với paper DeepRefusal

Table 1 của DeepRefusal benchmark LAT trên **đúng Llama-3-8B**: GCG 2.0 · Prefilling 0.0 · Refusal-Transfer 87.5.

Hai con số đầu **cùng bậc độ lớn** với 1.0% mình đo trên HarmBench thô. Không so trực tiếp được (khác loại attack — họ dùng GCG/prefilling, mình dùng prompt trần), nhưng đủ để nói **pipeline không có lỗi hệ thống**.

⚠️ Refusal-Transfer 87.5 là điểm yếu riêng của LAT mà HarmBench của mình **không chạm tới** — giống ca DeRTa với prefilling. Ghi rõ kẻo kết luận "LAT gần như hoàn hảo".

## Lưu ý khi báo cáo

- Checkpoint là **fp32 7 shard** (~32GB trên đĩa); `LocalClient` nạp bf16 nên vừa MIG 40GB.
- Model card trên HF **trống** (auto-generated, không ghi cấu hình train) → mọi thông tin về cách nó được tạo phải trích từ paper, không trích từ HF.
