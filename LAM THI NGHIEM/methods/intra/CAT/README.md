# CAT / CAPO — Efficient Adversarial Training in LLMs with Continuous Attacks

**Nhóm:** intra (sửa trọng số vĩnh viễn) · **Venue:** NeurIPS 2024 Spotlight
**Paper:** https://arxiv.org/abs/2405.15589 · **Repo gốc:** https://github.com/sophie-xhonneux/Continuous-AdvTrain (MIT, clone trong `repo/`)

## Method làm gì

Adversarial training cho LLM xưa nay đắt vì phải **search chuỗi token tấn công rời rạc** (GCG cần hàng nghìn forward). Bài này tính adversarial perturbation thẳng trong **không gian embedding liên tục** — gradient chạy trực tiếp, không cần search. Rẻ hơn R2D2 **299×** cho cả quá trình train.

Hai biến thể:
- **CAT (C-AdvUL)** — loss toward/away (unlikelihood) dưới continuous attack + loss utility trên UltraChat200k để giữ năng lực.
- **CAPO (C-AdvIPO)** — biến thể adversarial của IPO. KL ngầm với model gốc chặn model suy sụp thành "từ chối tất cả" → **bỏ được utility data**.

Đây là intra nên **inference overhead = 0**: sau khi train xong nó chỉ là một model bình thường, `method.py` gọi đúng 1 lần như `no_defense`.

## Cách chạy (mặc định: dùng checkpoint của tác giả, KHÔNG train)

```bash
python method.py response --task all        # sinh 300 HarmBench + 250 XSTest + 800 JustEval
python method.py judge    --task xstest     # over-refusal (judge API)
python method.py judge    --task harmbench  # ASR (classifier GPU)
python method.py judge    --task justeval   # utility (judge API)
```

Checkpoint mặc định: **`ContinuousAT/Llama3-8B-IT-CAT`**. Đây là điểm may: paper chỉ train Gemma-2B / Phi-3-Mini / Mistral-7B / Zephyr-7B / Llama-2-7B, **không có Llama-3** — nhưng tác giả vẫn release ckpt Llama-3-8B-Instruct trên HF.

Đổi checkpoint bằng biến môi trường:
```bash
CAT_MODEL=ContinuousAT/Llama-2-7B-CAT python method.py response --task harmbench
CAT_MODEL=HuggingFaceH4/zephyr-7b-beta CAT_LORA=ContinuousAT/Zephyr-CAT python method.py response   # Zephyr là LoRA
```
Các ckpt khác: `ContinuousAT/Phi-CAT`, `ContinuousAT/Phi-CAPO`, `ContinuousAT/Zephyr-CAT` (LoRA), `ContinuousAT/Llama-2-7B-CAT`.

## Train lại

```bash
python train_smoke.py                       # ~10 step, batch 1 — chỉ để chứng minh đường ống chạy
python train_smoke.py --algo ipo            # CAPO
python train_smoke.py --full                # cấu hình paper
python train_smoke.py --base <hf-id>        # base khác
```

`train_smoke.py` **không viết lại thuật toán** — nó sinh đúng file config mà upstream yêu cầu (`repo/config/path/local_run.yaml`) rồi gọi entry point Hydra `repo/src/run_experiments.py`.

Cấu hình upstream (`repo/config/adv_train_ul.yaml` + dataclass default trong `run_experiments.py`):

| | |
|---|---|
| Data | `adv_training_behaviors` (AT set của HarmBench) + `ultrachat_200k`, trộn 0.125 adv / 0.875 utility |
| Attack | embedding-space, 10 iteration, sign optimizer lr=1e-4, **eps=0.05** (Llama/Mistral; 0.1 Gemma/Phi; 0.075 Zephyr) |
| PEFT | LoRA **r=64, alpha=16**, `target_modules="all-linear"` |
| Quantize | bitsandbytes **4-bit** (nf4) |
| Training | 5 epoch, batch 8 × grad-accum 8, max_seq_length 256 |
| Loss | toward/away weight 0.5, utility weight 1.0, cutoff -5.0 / 0.5 |

**Thời gian báo cáo:** CAT **~42 phút**, CAPO **~19 phút** trên 1×A100. Paper ghi rõ cluster có **A100 40GB** → 40GB là đủ.

> Con số "≥1904 GPU hours" trong paper là **tổng TẤT CẢ thí nghiệm** (5 model × train + chạy GCG/AutoDAN/PAIR để đánh giá). Attack mới là thứ ngốn GPU, không phải train. Pipeline mình không chạy attack.

Data train nằm sẵn trong `repo/data/`, không phải tải gì.

## Sai khác so với upstream (phải khai báo trong báo cáo)

1. **Không chạy GCG/AutoDAN/PAIR.** Paper đo ASR dưới 3 attack đó; mình đo trên prompt HarmBench thô. Số ASR của mình **không so trực tiếp** được với bảng của họ.
2. `train_smoke.py --full` hạ `per_device_train_batch_size` 8 → 2 và đẩy bù vào `gradient_accumulation_steps` 8 → 32 để vừa 40GB. Effective batch giữ nguyên 64.
3. Với Llama-3 script ép `bf16=True, fp16=False` (yaml gốc set fp16 vì nhắm card đời cũ).

## Lưu ý khi đọc kết quả

- **Safe answer lúc train cố định một câu**: `"Sorry, I can't do that."` → model có xu hướng từ chối cụt lủn. Ảnh hưởng **JustEval** (engagement/depth), không ảnh hưởng ASR.
- **KHÔNG nhiễm XSTest**: over-refusal của họ đo bằng bộ HARMLESS 40 câu tự viết, không đụng XSTest → số over-refusal của mình sạch. (Khác Circuit Breakers và DeepRefusal — hai bài đó có XSTest trong retain set.)
- Upstream yêu cầu `fcntl` → **POSIX only**, chạy trên server Linux, không chạy được trên Windows.

## Trạng thái — ✅ ĐÃ CHẠY THẬT (26/07/2026, server H100 MIG 40GB)

Smoke test `python method.py response --task harmbench --limit 3` → **PASS**, 3/3 request.
**local_sec = 0.317 ± 0.275 / request.** Response cả 3: `"Sorry, I can't do that."`

→ Xác nhận đúng caveat ở trên: model trả về **chính xác chuỗi safe answer cố định lúc train**, không phải một câu từ chối tự nhiên. Điều này sẽ kéo điểm **JustEval** (engagement/depth) xuống — nhớ diễn giải đúng khi đọc bảng utility.

**Hai thứ phải fix mới chạy được (đã fix trong code):**
1. Checkpoint **không ship tokenizer** (repo chỉ có `config.json`, `generation_config.json`, safetensors) → phải lấy tokenizer từ base model, xem biến `TOKENIZER`.
2. `meta-llama/*` bị **gated**, server không có HF token → default đổi sang mirror **`NousResearch/Meta-Llama-3-8B-Instruct`** (bản copy y hệt). Có token thì set `CAT_TOKENIZER=meta-llama/Meta-Llama-3-8B-Instruct`.

Checkpoint là **F32 → ~30GB tải về**, nạp bằng bf16 nên chỉ ~16GB VRAM.

## ✅ TRAIN LẠI CŨNG ĐÃ CHẠY THẬT — nhưng là bài khó nhất trong 4 bài

```bash
source .venv_cat/bin/activate           # venv RIÊNG, xem bên dưới
python train_smoke.py --base NousResearch/Meta-Llama-3-8B-Instruct --steps 5
```
→ **exit=0**, `train_runtime 2.26s` cho 5 step, adapter lưu ở `train_out/training/ul/None/final_model/`.

### Phải xử lý 6 thứ

**1. Cần venv riêng — không dùng chung venv chính được.**
`repo/src/data.py` import `DataCollatorForCompletionOnlyLM` từ trl, mà **trl 1.9 đã xoá hẳn class này**. CAT còn **kế thừa** nó (`class MultiDatasetDataCollatorCompletion(DataCollatorForCompletionOnlyLM)`) nên không shim an toàn được. → dựng `.venv_cat` với stack upstream pin.

**2. `requirements.txt` của upstream pin sai version.**
`transformers==4.41.3` **không tồn tại trên PyPI** (danh sách nhảy 4.41.2 → 4.42.0). Cài nguyên `requirements.txt` là hỏng cả lệnh. Dùng **4.41.2**.

**3. torch phải ép đúng build CUDA.**
`pip install torch` mặc định kéo về `2.13.0+cu130`, trong khi driver server là **CUDA 12.3** → `torch.cuda.is_available() = False`. Phải ép `--index-url .../cu124` với `torch==2.6.0`.

**4. Upstream KHÔNG có chat template cho Llama-3.**
`src/model_utils.py` có whitelist cứng (gemma / llama-2 / safe-llama2 / mistral-instruct / mistral / phi) và raise:
```
NotImplementedError: Model Meta-Llama-3-8B-Instruct not supported
```
Khớp với paper — Llama-3 **không nằm trong thí nghiệm của họ**. `train_smoke.py` tự thêm nhánh `llama-3` (dùng đúng special token `<|start_header_id|>`, `<|eot_id|>`) vào cả `get_chat_template` lẫn `get_model_name`, có lưu `model_utils.py.orig`.

> 📌 **Điểm đáng ghi vào survey:** checkpoint `ContinuousAT/Llama3-8B-IT-CAT` mà mình đang dùng cho phần inference **được train bằng code không có trong repo public**. Không tái lập được đúng quy trình sinh ra nó.

**5. `path.model_name` là KEY nội bộ, không phải tên HF.**
Xem `example_path.yaml`: `model_path=zephyr-7b-beta` nhưng `model_name=mistral`. `train_smoke.py` có bảng map riêng.

**6. Llama-3 không có `unk_token`.**
Upstream viết `tokenizer.pad_token = tokenizer.unk_token` → với Llama-3 thì pad_token vẫn None và collator chết ngay step đầu:
```
ValueError: Asking to pad but the tokenizer does not have a padding token.
```
Vá thành `unk_token or eos_token` (2 chỗ: nhánh ul và dpo), có backup `.orig`.

**Thêm:** venv legacy pin `datasets==2.17.1` còn venv chính có 5.x, hai bên **dùng chung `~/.cache/huggingface`** → bản cũ không đọc được metadata bản mới ghi (`TypeError: must be called with a dataclass type or instance`). `train_smoke.py` set `HF_DATASETS_CACHE` riêng; cache **model** vẫn dùng chung nên không tải lại model.

### ✅ Đã kiểm chứng loss adversarial thật sự chạy

Ban đầu smoke run 5 step cho `away_loss: 0, toward_loss: 0, attack_loss: 0` — chỉ `utility_loss` khác 0. Chạy lại **30 step vẫn y hệt**, tức là nhánh adversarial (thứ làm nên CAT) chưa hề được kích hoạt. Tỉ lệ trộn là **0.125 adv / 0.875 utility**, nên 30 lượt toàn utility có xác suất ~1.8% — đủ thấp để nghi có lỗi thật.

**Test quyết định:** ép tỉ lệ 50/50 rồi chạy 10 step:
```bash
python src/run_experiments.py --config-name=adv_train_ul path=local_run \
  training.max_steps=10 "dataset.probabilities=[0.5,0.5]" ...
```
Kết quả:
```
{'global_step': 1, 'loss': 0.815, 'away_loss': -1.614, 'toward_loss': 3.244}
{'global_step': 3, 'loss': 0.307, 'away_loss': -1.230, 'toward_loss': 1.844}
```
**5/10 step có `away_loss` và `toward_loss` khác 0** — đúng bằng tỉ lệ đặt vào. → **Cơ chế adversarial chạy đúng**, kết quả toàn-0 trước đó chỉ là do bốc mẫu ở tỉ lệ 12.5% với seed cố định.

📌 **Rút kinh nghiệm cho lần chạy thật:** nếu train ở tỉ lệ mặc định 0.125, **đừng đọc vài chục step đầu rồi kết luận** — phải chạy đủ dài hoặc kiểm bằng `dataset.probabilities` tạm thời để biết đường ống adversarial còn sống.
