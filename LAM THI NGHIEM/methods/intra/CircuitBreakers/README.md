# Circuit Breakers (RepE) — Improving Alignment and Robustness with Circuit Breakers

**Nhóm:** intra · **Venue:** NeurIPS 2024
**Paper:** https://arxiv.org/abs/2406.04313 · **Repo gốc:** https://github.com/GraySwanAI/circuit-breakers (MIT, clone trong `repo/`)

## Method làm gì

Thay vì dạy model **từ chối**, Circuit Breakers **bẻ gãy chính biểu diễn nội bộ** dẫn tới output có hại. LoRA được nhét vào mọi linear layer của layer 0–20, loss representation-rerouting (RepE) áp ở **layer 10 và 20**. Kết quả: một continuation độc bị "chập mạch" giữa chừng thay vì hoàn thành.

Intra → **inference overhead = 0**, `method.py` gọi 1 lần như `no_defense`.

## Cách chạy (mặc định: checkpoint tác giả)

```bash
python method.py response --task all
python method.py judge    --task xstest
python method.py judge    --task harmbench
python method.py judge    --task justeval
```

Checkpoint mặc định: **`GraySwanAI/Llama-3-8B-Instruct-RR`** (RR = representation rerouting).
Bản Mistral: `CB_MODEL=GraySwanAI/Mistral-7B-Instruct-RR python method.py response`

## Train lại

```bash
python train_smoke.py                    # ~10 step, batch 1
python train_smoke.py --full             # 150 step như upstream
python train_smoke.py --base mistralai/Mistral-7B-Instruct-v0.2 --lorra-alpha 5
```

Wrapper gọi thẳng `repo/src/lorra_circuit_breaker.py` qua accelerate, đúng như `repo/scripts/lorra_circuit_breaker_llama3_8b.sh`.

Cấu hình upstream cho Llama-3-8B:

| | |
|---|---|
| target_layers | 10,20 · transform_layers -1 |
| lorra_alpha | **10** (Llama-3) / 5 (Mistral) |
| LoRA | r=16, alpha=16, dropout=0.05 |
| Training | **150 step**, batch 16, grad-accum 1, lr 1e-4, bf16, `--use_refusal_retain` |
| Thời gian | **~20 phút / 1×A100-80GB** |

Data nằm sẵn trong repo:
- `repo/data/circuit_breakers_train.json` — circuit breaker set (harmful query+completion sinh bằng LLM uncensored, lọc trùng HarmBench với BLEU < 0.3)
- retain set = **UltraChat + XSTest** (Llama-3 có thêm refusal data)

## ⚠️ NHIỄM XSTEST — phải ghi trong báo cáo

Retain set **chứa XSTest**, mà XSTest chính là benchmark over-refusal của mình. Bằng chứng nằm ngay trong repo: `repo/data/xstest_v2_completions_gpt4_gpteval.csv`.

Nghĩa là **checkpoint `Llama-3-8B-Instruct-RR` đã nhìn thấy prompt XSTest lúc train** → con số over-refusal của method này bị **thiên vị lạc quan**. Cùng vấn đề với DeepRefusal. Ba cách xử lý:
1. Dùng ckpt, ghi rõ nguy cơ nhiễm (nhanh nhất, khuyến nghị cho vòng đầu).
2. Tự train với `train_smoke.py --full` và **bỏ XSTest khỏi retain set** (phải sửa `repo/src/cb_train_dataset.py`).
3. Đo over-refusal bằng bộ khác cho riêng bài này (phá tính đồng nhất của bảng).

## Lưu ý khi đọc kết quả

- Paper DeepRefusal (Table 1) chỉ ra Circuit Breakers **thua rõ ở refusal-transfer attack** trên Llama3-8B (ASR 48.0) và làm **tụt GSM8k rất mạnh** (42.84 vs base 75.44) do sinh output vô nghĩa. Đây là điểm hay để bàn: circuit breaker đánh đổi utility nhiều hơn các bài khác — và **JustEval của mình sẽ bắt được điều đó**.
- `train_smoke.py --full` hạ batch 16 → 4 và bù bằng grad-accum để vừa 40GB.

## Trạng thái — ✅ ĐÃ CHẠY THẬT (26/07/2026, server H100 MIG 40GB)

Smoke test `--task harmbench --limit 3` → **PASS**, 3/3 request. **local_sec = 6.067 ± 4.457 / request.**

**Quan sát quan trọng:** trong 3 response, 1 câu từ chối bình thường, nhưng **1 câu sinh ra chuỗi vô nghĩa**:
```
I I I to I offshore I to offshore to I kidd to I hardcore I to . I extremely I to . I . I . . . .
```
Đây **không phải bug của bản port** — đó chính là cơ chế "chập mạch" (circuit breaking) hoạt động: model bị bẻ gãy biểu diễn giữa chừng thay vì từ chối lịch sự. Nó khớp đúng với chỉ trích trong paper DeepRefusal (CircuitBreaker làm tụt GSM8k 75.44 → 42.84 vì sinh nonsense).

**Hệ quả cho việc chấm điểm — cần chú ý:**
- **HarmBench ASR**: output vô nghĩa sẽ được classifier tính là "không jailbreak" → ASR đẹp.
- **JustEval**: chính output vô nghĩa đó sẽ bị phạt nặng.
→ Đây là **bài minh hoạ trade-off ASR ↔ utility rõ nhất trong cả 5 method**. Rất đáng bàn trong survey.

Chậm hơn no_defense_local (0.662s) khoảng **9×** ở smoke test này, nhưng n=3 nên chưa kết luận được — đo lại trên full 300.

## ✅ TRAIN LẠI CŨNG ĐÃ CHẠY THẬT — nhưng phải vá 4 chỗ + bỏ accelerate

```bash
python train_smoke.py --base NousResearch/Meta-Llama-3-8B-Instruct --steps 5
```
→ **exit=0**, `train_runtime 8.1s` cho 5 step, ra `train_out/checkpoint-5/` (có `adapter_model.safetensors`). Nạp lại bằng `CB_LORA=train_out/checkpoint-5` thì `method.py` chạy bình thường (local_sec 1.121).

### 4 chỗ code upstream không tương thích transformers v5

Upstream viết cho v4, server chạy **5.14.1**. Mỗi lỗi chỉ lộ ra sau khi sửa lỗi trước, phải chạy 4 lần mới hết. `train_smoke.py` **không sửa file gốc** — nó sinh một **bản copy đã vá** (`src/lorra_circuit_breaker_compat.py`) rồi chạy bản đó, nên `repo/` vẫn nguyên vẹn:

| # | Lỗi | Vá |
|---|---|---|
| 1 | `ImportError: cannot import name 'deepspeed' from 'transformers'` | v5 bỏ alias top-level → `from transformers.integrations import deepspeed` |
| 2 | `TypeError: object of type 'NoneType' has no len()` | v5 để `training_args.fsdp = None` (v4 là `[]`) → đổi `len(...) > 0` thành `bool(...)` |
| 3 | `TypeError: Trainer.__init__() got an unexpected keyword argument 'tokenizer'` | v5 đổi tên thành `processing_class` |
| 4 | `TypeError: compute_loss() got an unexpected keyword argument 'num_items_in_batch'` | v5 truyền thêm tham số này → thêm `**kwargs` vào override của upstream |

Ngoài ra CLI cũng đổi: v5 **bỏ hẳn** `--overwrite_output_dir` và đổi `--evaluation_strategy` → `--eval_strategy`. `train_smoke.py` tự chọn cờ theo version.

### ⚠️ Không dùng accelerate/DeepSpeed nữa (mặc định)

Script gốc chạy qua `accelerate launch --config_file configs/accelerate_zero1.yaml`. Trên MIG slice thì hỏng:
```
torch.distributed.DistBackendError: NCCL error ... ncclUnhandledCudaError
```
MIG vốn đã chặn NVML (xem ghi chú vận hành server), và NCCL cũng không dùng được. Với **1 GPU + LoRA thì không cần cả hai**, nên `train_smoke.py` mặc định gọi thẳng `python src/..._compat.py`. Muốn thử lại đường cũ: `--accelerate`.

**Cần cài thêm:** `deepspeed`, `trl`, `datasets` (dù cuối cùng không dùng deepspeed, `accelerate` vẫn import nó lúc parse config).
