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

## Trạng thái — 🔧 ĐÃ CODE, đang chạy trên server

## Lưu ý khi báo cáo

- Checkpoint là **fp32 7 shard** (~32GB trên đĩa); `LocalClient` nạp bf16 nên vừa MIG 40GB.
- Model card trên HF **trống** (auto-generated, không ghi cấu hình train) → mọi thông tin về cách nó được tạo phải trích từ paper, không trích từ HF.
