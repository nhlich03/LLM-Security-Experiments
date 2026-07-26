# DeRTa — Decoupled Refusal Training (Refuse Whenever You Feel Unsafe)

**Nhóm:** intra · **Venue:** ACL 2025 main
**Paper:** https://arxiv.org/abs/2407.09121 · **Repo gốc:** https://github.com/RobustNLP/DeRTa (clone trong `repo/`)

## Method làm gì

Alignment thông thường chỉ dạy model từ chối **ở token đầu tiên**. Hệ quả: khi attacker ép sẵn một prefix tuân thủ (prefilling attack), model vui vẻ viết tiếp. DeRTa tách đôi vấn đề:

1. **MLE với Harmful Response Prefix** — sample train được prepend sẵn một đoạn mở đầu độc hại, nhưng target vẫn là câu từ chối → model học cách **bẻ lái sang từ chối giữa chừng**.
2. **RTO (Reinforced Transition Optimization)** — reinforce điểm chuyển sang từ chối ở **mọi vị trí** của continuation độc, không chỉ vị trí đầu.

Intra → **inference overhead = 0**.

## Cách chạy (mặc định: checkpoint tác giả)

```bash
python method.py response --task all
python method.py judge    --task xstest
python method.py judge    --task harmbench
python method.py judge    --task justeval
```

Mặc định là **base + LoRA rời** (khác CAT / Circuit Breakers vốn publish full weight đã merge):
- BASE = `meta-llama/Meta-Llama-3-8B-Instruct`
- LORA = `Youliang/llama3-8b-instruct-lora-derta-100step` ← tác giả đánh dấu **"Recommend"**

Checkpoint khác:
```bash
DERTA_BASE=Youliang/llama3-8b-derta DERTA_LORA= python method.py response          # full weight, base Meta-Llama-3-8B
DERTA_BASE=Youliang/llama3-8b-instruct-derta-100step DERTA_LORA= python method.py response   # tác giả ghi "Not Recommended"
```
Bản 70B (`Youliang/llama3-70b-lora-derta`) **không vừa 40GB**.

## Train lại

```bash
python train_smoke.py                          # tự sinh data nếu thiếu, ~10 step
python train_smoke.py --dataset llama_vanilla  # baseline
python train_smoke.py --full                   # 2 epoch như upstream
```

Wrapper gọi `repo/run_files/run_clm_lora_derta_llama.py`. **Data sinh tại chỗ**, không tải gì:

```bash
cd repo/data/train && python generate_training_data.py
```

Sinh ra 3 file — chọn file nào **chính là ablation**:

| File | Nội dung |
|---|---|
| `llama_derta.json` | **DeRTa**: harmful-prefix samples + RTO transition targets ← đây là method |
| `llama_vanilla.json` | safety SFT thường (baseline) |
| `llama_recaug.json` | MLE với harmful response prefix, không có RTO (baseline) |

Cấu hình upstream (nhánh LoRA trong `repo/train.sh`): batch 8 × grad-accum 2, 2 epoch, lr 1e-4, block_size 512, bf16, gradient checkpointing.

## ⚠️ Sai khác lớn nhất: 8 GPU → 1 GPU

`repo/train.sh` viết cho **8 GPU + DeepSpeed** (`torchrun --nproc_per_node 8`), và nhánh 8B trong đó là **full-param** chứ không phải LoRA (nhánh LoRA nhắm 70B). `train_smoke.py` chạy **1 GPU, không DeepSpeed, không torchrun**, dùng đường LoRA cho 8B.

Hệ quả: **effective batch khác upstream** → nếu train lại thì kết quả sẽ không khớp số của paper. Muốn khớp thì phải bù bằng gradient accumulation cho đúng effective batch. **Upstream không báo cáo thời gian train**, nên chưa ước được trên 40GB.

→ Vì vậy **khuyến nghị mạnh: dùng checkpoint có sẵn**, chỉ train khi thật sự cần đổi base model.

## Điểm mạnh cần chú ý khi đọc kết quả

DeRTa nhắm thẳng **prefilling attack** — lỗ hổng mà shallow alignment để lại. Nhưng `harmbench_300.csv` của mình là **prompt thô, không có prefilling**, nên điểm mạnh nhất của method này **sẽ không thể hiện ra trong bảng**. Cần ghi rõ khi so sánh, kẻo kết luận sai là "DeRTa yếu".

## Trạng thái — ✅ ĐÃ CHẠY THẬT (26/07/2026, server H100 MIG 40GB)

Smoke test `--task harmbench --limit 3` → **PASS**, 3/3 request. **local_sec = 2.567 ± 1.444 / request.**

Response là câu từ chối **có giải thích lý do**, dài và tự nhiên hơn hẳn CAT:
> "I'm sorry, but I cannot assist with that request. It is illegal and dangerous to make dimethylmercury..."

→ Trái ngược với CAT (chuỗi cố định cụt lủn). Dự đoán: DeRTa sẽ ăn điểm **JustEval** cao hơn CAT.

**Fix đã áp dụng:** base đổi từ `meta-llama/Meta-Llama-3-8B-Instruct` (gated, server không có HF token) sang mirror **`NousResearch/Meta-Llama-3-8B-Instruct`**. LoRA `Youliang/llama3-8b-instruct-lora-derta-100step` nạp chồng lên bình thường, không cần chỉnh gì.

## 🔴 TRAIN LẠI: repo upstream THIẾU FILE DATA

Chạy `python train_smoke.py` lần đầu:
```
FileNotFoundError: [Errno 2] No such file or directory:
'safety_beaver_safe_and_unsafe_response.json'
```

Đó là **dòng đầu tiên** mà `repo/data/train/generate_training_data.py` đọc. File này **không có trong repo** — họ chỉ ship 6 shard `helpfulness_unsencor_wizardlm_60k_part_*.json` (nửa helpfulness), còn **nửa safety thì thiếu**, và README không nói cách lấy. Đã `find` toàn repo để chắc chắn.

### Giải pháp: `rebuild_safety_data.py` — tái tạo từ PKU-SafeRLHF

Đọc code consumer thì thấy mỗi record chỉ cần 3 trường:

| Trường | Nội dung |
|---|---|
| `instruction` | câu hỏi có hại |
| `output` | câu trả lời **ĐỘC** (dùng làm harmful prefix + target cho refusal-token prediction) |
| `safe_response` | câu trả lời **AN TOÀN** (thứ model phải bẻ lái sang) |

Paper nói data lấy từ BeaverTails, và **`PKU-Alignment/PKU-SafeRLHF`** đúng hình dạng đó: mỗi dòng có 1 prompt + 2 response kèm nhãn an toàn từng cái. Lấy các dòng có **đúng một** response safe → ra cặp (unsafe, safe) cho cùng prompt.

```bash
python rebuild_safety_data.py            # 6000 cặp (script chỉ tiêu thụ 6000)
cd repo/data/train && python generate_training_data.py
```
→ Đã chạy thật: **73,907 dòng nguồn → 6,000 cặp** (bỏ 35,422 dòng vì cả hai response cùng nhãn), rồi `generate_training_data.py` sinh ra `llama_derta.json` / `llama_recaug.json` / `llama_vanilla.json`, mỗi file ~116MB. ✅

Lưu ý kỹ thuật: file phải là **JSON Lines** (mỗi dòng một object) dù đuôi là `.json`, vì `read_from_json` của họ duyệt từng dòng.

### ⚠️ ĐÂY LÀ BẢN TÁI TẠO, KHÔNG PHẢI FILE GỐC

Cùng dataset nguồn, cùng schema, nhưng **các cặp cụ thể khác** với của tác giả → câu từ chối cũng khác. Model train ra là **"DeRTa-style", không phải tái hiện bit-exact**. Bắt buộc ghi rõ trong báo cáo nếu dùng số từ model tự train.

→ Vì vậy vẫn **khuyến nghị dùng checkpoint có sẵn** cho bảng kết quả chính.

### Bốn thứ phải xử lý để training chạy

**1. `transformers` v5 không dùng được — phải venv legacy.**
```
ImportError: cannot import name 'is_torch_tpu_available' from 'transformers'
```
`run_clm_lora_derta_llama.py` là bản fork `run_clm.py` đời cũ, còn nhiều chỗ v4-only. Upstream pin `transformers==4.40.0`. Thay vì vá từng dòng, **dùng chung venv legacy với CAT**:
```bash
source ../CAT/.venv_cat/bin/activate     # transformers 4.41.2, torch 2.6.0+cu124
```

**2. Thiếu `deepspeed`** — script import ở top level dù chạy 1 GPU không cần.

**3. `cpu_adam` bị JIT-compile vô điều kiện.** Dòng 319 gọi thẳng:
```python
deepspeed.ops.op_builder.CPUAdamBuilder().load()
```
→ trên máy không có ninja + CUDA build toolchain thì chết:
```
RuntimeError: Error building extension 'cpu_adam'
Command '['ninja', '-v']' returned non-zero exit status 1
```
Kernel này **chỉ cần cho DeepSpeed CPU-offload**, single-GPU LoRA không bao giờ đụng tới. `train_smoke.py` sinh **bản copy đã vá** (`run_files/run_clm_lora_derta_llama_compat.py`) bọc dòng đó sau cờ `DERTA_BUILD_CPU_ADAM=1`, `repo/` giữ nguyên.

**4. Thiếu file data** — xem phần trên.

### ✅ Kết quả train (smoke)

```bash
source ../CAT/.venv_cat/bin/activate
python train_smoke.py --base NousResearch/Meta-Llama-3-8B-Instruct --steps 5
```
→ **exit=0**, `train_loss = 3.1743`, `train_runtime = 3.41s`, ra `train_out/lora_llama_derta/` với `adapter_model.safetensors` + `adapter_config.json`.

### Ghi chú cấu hình quan sát được

LoRA của họ khá nặng: **r=96**, target 10 module (`q/k/v/o_proj`, `gate/up/down_proj`, `w1/w2/w3`) — lớn hơn nhiều so với r=16 của Circuit Breakers và r=64 của CAT. Tức là nếu train thật thì DeRTa tốn VRAM nhất trong 3 bài intra.

### ✅ Adapter tự train nạp lại được — nhưng phải vá thêm 2 chỗ

```bash
DERTA_BASE=NousResearch/Meta-Llama-3-8B-Instruct DERTA_LORA=train_out/lora_llama_derta \
  python method.py response --task harmbench --limit 2
```
→ chạy được, **local_sec 1.006 ± 0.431**. Hai vấn đề gặp phải:

**1. `target_modules` chứa module Llama-3 không có.**
Config LoRA của họ liệt kê cả `w1`, `w2`, `w3` (tên module kiểu Mixtral) bên cạnh các module Llama. Llama-3 không có chúng nên **không có weight nào được train cho chúng** — soi file safetensors thấy đúng 450 tensor, chỉ thuộc `q/k/v/o_proj` + `gate/up/down_proj`. peft 0.9 (venv train) im lặng bỏ qua, **peft 0.19 (venv chạy) thì từ chối nạp**:
```
RuntimeError: Error(s) in loading state_dict for PeftModelForCausalLM
```
→ `method.py::sanitize_local_adapter()` tự lọc bỏ 3 tên chết khỏi `adapter_config.json` (có backup `.orig`). Chỉ áp dụng cho adapter **tự train**, checkpoint trên Hub không bị đụng.

**2. Script train thêm 1 special token và resize embedding.**
```
size mismatch for base_model.model.model.embed_tokens.weight:
  copying a param with shape [128257, 4096], current model is [128256, 4096]
```
Trainer thêm pad token (128256 → **128257**) rồi lưu luôn `embed_tokens`/`lm_head` vào adapter. → `method.py` lấy tokenizer **từ chính thư mục adapter** và bật `resize_to_tokenizer=True` (tuỳ chọn mới trong `core/local_client.py`) để resize base trước khi gắn LoRA.

> Cả hai chỉ xảy ra với adapter **tự train**. Dùng checkpoint `Youliang/llama3-8b-instruct-lora-derta-100step` thì không gặp.
