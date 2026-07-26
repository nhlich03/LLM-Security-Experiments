# no_defense_local — mốc quy chiếu cho bảng LOCAL

**Nhóm:** none (baseline) · **Không phải paper** — đây là dòng đối chứng.

## Vì sao cần file này

`methods/no_defense/` đã là mốc rồi, nhưng nó chạy **API Groq** (`llama-3.1-8b-instant`). Toàn bộ nhóm **in / intra** chạy **trọng số local**, nên phải có mốc riêng **trên đúng base model đó**. Không có nó thì con số của SafeDecoding/CAT/... đang so với *một model khác*, chứ không phải so với *không phòng thủ*.

Nó còn là **mẫu số của cost local**: overhead của một method in = `local_sec(method) / local_sec(no_defense_local)`, đo trên cùng GPU, cùng `max_tokens`, cùng bộ prompt.

Không có cơ chế phòng thủ nào — gọi target đúng một lần, `generate=local_generate`.

## Cách chạy

```bash
python method.py response --task all
python method.py judge    --task xstest
python method.py judge    --task harmbench
python method.py judge    --task justeval
```

**Base model chưa chốt** → đọc từ env, mặc định `NousResearch/Meta-Llama-3-8B-Instruct`:

```bash
LOCAL_TARGET_MODEL=<hf_id> python method.py response --task all
```

Dùng mirror `NousResearch/*` vì `meta-llama/*` bị gated và server không có HF token — đã đối chiếu SHA256 cả 4 shard, **trùng khít** bản chính thức.

Trên server MIG phải đặt trước:

```bash
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
```

## Trạng thái — ✅ ĐÃ CHẠY THẬT (26/07/2026, server H100 MIG 40GB)

Smoke test `--task harmbench --limit 3` → **PASS**. **local_sec = 0.662 ± 0.384 / request.**

⚠️ n=3 nên **chưa dùng con số này làm mẫu số được**. Phải chạy full 300 rồi mới chốt.

## Lưu ý khi so bảng

Bảng **target-API** và bảng **target-local** không cùng thang — base khác nhau. Báo cáo phải **tách 2 bảng**, mỗi bảng có dòng `no_defense` của chính nó. Muốn bắc cầu giữa hai bảng thì chạy `no_defense` ở **cả hai chế độ trên cùng một model** rồi so chênh lệch.

Cost local ghi **cả giây lẫn token** (`local_sec`, `local_in_tokens`, `local_out_tokens`) — chưa chốt dùng đơn vị nào, xem `docs/PHUONG_PHAP.md` §7.
