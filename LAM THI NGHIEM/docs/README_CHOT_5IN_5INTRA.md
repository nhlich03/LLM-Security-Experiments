# Hồ sơ kỹ thuật 5 IN + 5 INTRA (để quyết định chạy bài nào)

> ## ✅ ĐÃ TRIỂN KHAI 5/10 (26/07/2026)
>
> **SafeDecoding · JBShield · CAT · Circuit Breakers · DeRTa** đã có code, chạy bằng checkpoint tác giả.
> Chi tiết từng bài: README trong chính folder method. Hạ tầng mới: `core/local_client.py`.
>
> | Method | Folder | Lệnh |
> |---|---|---|
> | no_defense (local) | `methods/no_defense_local/` | `python method.py response --task all` |
> | SafeDecoding | `methods/in/SafeDecoding/` | `python method.py response --task all` |
> | JBShield | `methods/in/JBShield/` | `python calibrate.py --model llama-3` **trước**, rồi `method.py` |
> | CAT | `methods/intra/CAT/` | `python method.py response --task all` |
> | Circuit Breakers | `methods/intra/CircuitBreakers/` | `python method.py response --task all` |
> | DeRTa | `methods/intra/DeRTa/` | `python method.py response --task all` |
>
> Train lại: `train_smoke.py` (3 bài intra) và `train_expert.py` (SafeDecoding). Mặc định smoke-size để chứng minh đường ống chạy; `--full` cho cấu hình paper.
>
> ### ✅ ĐÃ SMOKE TEST THẬT trên server H100 MIG 40GB (26/07/2026)
>
> `python method.py response --task harmbench --limit 3` — **4/5 PASS**, JBShield đang chạy calibration.
>
> | Method | Kết quả | local_sec/request | Ghi chú |
> |---|---|---|---|
> | no_defense_local | ✅ PASS | **0.662 ± 0.384** | mốc quy chiếu |
> | CAT | ✅ PASS | **0.317 ± 0.275** | trả về đúng chuỗi cố định `"Sorry, I can't do that."` |
> | DeRTa | ✅ PASS | **2.567 ± 1.444** | từ chối có giải thích, tự nhiên nhất |
> | SafeDecoding | ✅ PASS | **5.490 ± 2.449** | `disable_adapter()` thay mixed-adapter-batch chạy đúng |
> | Circuit Breakers | ✅ PASS | **6.067 ± 4.457** | 1/3 response là **chuỗi vô nghĩa** — đúng cơ chế, xem README |
> | JBShield | ✅ PASS | **0.839 ± 0.506** | sau khi vá 2 bug + tinh chỉnh; verbatim thì 9.6s và ra rác |
>
> **JBShield tái hiện đúng paper ở phần detection:** accuracy trung bình **0.958** trên 9 attack (paper báo 0.95). Chi tiết từng attack trong README của method.
>
> ⚠️ n=3 nên **các con số local_sec chưa kết luận được gì** — chỉ chứng minh đường ống chạy. CAT nhanh hơn no_defense vì nó từ chối bằng 1 câu ngắn (ít token sinh ra), không phải vì model nhanh hơn.
>
> **6 thứ phải fix mới chạy được** (không đoán trước được, chỉ lộ ra khi chạy thật):
> 1. Server có **transformers 5.14.1** → `torch_dtype=` đã đổi tên thành `dtype=`. `local_client.py` giờ tự dò version.
> 2. **`meta-llama/*` bị gated**, server không có HF token → toàn bộ default chuyển sang mirror **`NousResearch/*`**.
> 3. **Checkpoint CAT không ship tokenizer** (chỉ có config + safetensors) → phải mượn tokenizer từ base model.
> 4. Thiếu package: `peft`, `tiktoken`, `sentencepiece`, `scikit-learn`, `nltk`, `fschat` — đã cài, đã ghi vào `requirements.txt` từng method.
> 5. **JBShield hook viết cho transformers v4**: v5 đổi `LlamaDecoderLayer.forward` trả về **tensor** thay vì tuple → `AttributeError: 'tuple' object has no attribute 'dtype'`. Đã vá bằng subclass, giữ nguyên phần toán.
> 6. **JBShield `detection()` không thực sự gate** (`if <list không rỗng>` luôn True) + không giới hạn số token can thiệp → sinh ra `"I cannot illegal illegal illegal..."`. Đã thêm 2 knob `JBS_GATE` / `JBS_FIRST_M`; mặc định mới cho output sạch và **nhanh hơn 12×**. Xem A/B trong README của method.
>
> ### 🔧 TRAIN LẠI trên Llama-3 (smoke-size) — 26/07/2026
>
**4/4 PASS.** Mỗi bài đều: train → nạp lại adapter tự train → sinh response thành công.
>
> | Method | Train | Thời gian | Adapter tự train nạp lại | Số thứ phải xử lý |
> |---|---|---|---|---|
> | **SafeDecoding expert** | ✅ | **16.2 s** | ✅ 1.932 s/req | **0** — chạy ngay lần đầu |
> | **Circuit Breakers** | ✅ | 8.1 s / 5 step | ✅ 1.121 s/req | **5** |
> | **CAT** | ✅ | 2.26 s / 5 step | ✅ 1.022 s/req | **6** |
> | **DeRTa** | ✅ | 3.41 s / 5 step | ✅ 1.006 s/req | **6** |
>
> Chi tiết từng lỗi + cách vá: README trong folder từng method. Tóm tắt các nhóm lỗi:
>
> | Nhóm | Ví dụ |
> |---|---|
> | **transformers v4 → v5** | `from transformers import deepspeed` bị bỏ · `fsdp=None` thay vì `[]` · `Trainer(tokenizer=)` → `processing_class` · `compute_loss(num_items_in_batch=)` · `is_torch_tpu_available` bị bỏ · `--overwrite_output_dir` bị bỏ, `--evaluation_strategy` → `--eval_strategy` |
> | **Thư viện xoá API** | `DataCollatorForCompletionOnlyLM` biến mất khỏi trl 1.9 (CAT *kế thừa* nó → không shim được, phải venv riêng) |
> | **Hạ tầng MIG** | NCCL chết (`ncclUnhandledCudaError`) → bỏ accelerate/DeepSpeed · `cpu_adam` JIT-compile cần ninja → tắt |
> | **Repo lỗi/thiếu** | CAT pin `transformers==4.41.3` **không tồn tại trên PyPI** · DeRTa **thiếu hẳn file data safety** · CAT **không có chat template Llama-3** |
> | **Đặc thù Llama-3** | không có `unk_token` → `pad_token=None` · adapter DeRTa resize vocab 128256→128257 · `target_modules` chứa `w1/w2/w3` của Mixtral |
>
> **Hai cách vá đã dùng:** (a) sinh **bản copy đã vá** (`*_compat.py`) rồi chạy bản đó, giữ `repo/` nguyên vẹn — dùng cho Circuit Breakers, DeRTa; (b) **venv riêng với version upstream pin** — dùng cho CAT, DeRTa (chung một `.venv_cat`: torch 2.6.0+cu124, transformers 4.41.2, trl 0.8.3, peft 0.9.0).
>
> ⚠️ **Bẫy khi đọc log CAT:** tỉ lệ trộn mặc định là 0.125 adv / 0.875 utility, nên 30 step đầu **toàn utility**, `away_loss`/`toward_loss` đều bằng 0 — dễ tưởng hỏng. Ép `dataset.probabilities=[0.5,0.5]` thì 5/10 step có loss adversarial khác 0 → cơ chế **vẫn đúng**.
>
> **Bài học chung:** đây đều là repo 2024 viết cho **transformers v4**, server chạy **v5.14.1**. Phần *inference* thì chỉ cần vá nhẹ, nhưng phần *training* đụng sâu vào `Trainer` nên vỡ hàng loạt. Hai chiến lược đã dùng:
> - **Vá bản copy** (Circuit Breakers): sinh `*_compat.py` từ file gốc, `repo/` giữ nguyên. Hợp khi chỉ vài chỗ.
> - **Venv riêng với version upstream pin** (CAT, DeRTa): hợp khi API bị **xoá hẳn** (không shim được) hoặc file là fork `run_clm.py` đời cũ.
>
> Hai phát hiện về chất lượng repo, đáng ghi trong survey:
> - **CAT pin sai version**: `transformers==4.41.3` **không tồn tại trên PyPI** (nhảy 4.41.2 → 4.42.0) → `requirements.txt` của họ không cài được nguyên trạng.
> - **DeRTa thiếu data**: `safety_beaver_safe_and_unsafe_response.json` — dòng đầu tiên script data đọc — **không có trong repo**, README không nhắc. Đã viết `rebuild_safety_data.py` tái tạo từ PKU-SafeRLHF (6000 cặp) và sinh lại được 3 file train, nhưng **không bit-exact với bản gốc**.
>
> **Đính chính con số trong file này:** ước lượng **JBShield ~1.1×T là SAI, quá lạc quan**. Đọc code mới thấy hook chạy một **SVD mỗi forward pass** (đó là lý do upstream đánh giá với `max_new_tokens=50` còn mình sinh 512). Phải đo thực tế.


Ngày lập: 26/07/2026. Nguồn: đọc trực tiếp repo + paper từng bài.
File liên quan: `README_PHUONG_PHAP_IN.md` (7 bài nhóm tìm) · `README_PHUONG_PHAP_INTRA.md` (19 bài nhóm tìm) · `PHUONG_PHAP_MOI.md` (nguồn gốc của ROSE/SafeInfer/DRO/CircuitBreakers/DeRTa/LAT).

> ⚠️ **Chưa chốt base model.** Toàn bộ ước lượng dưới đây giả định target **8B, chạy local trên 1×MIG 40GB, batch ≤ 4**. Nếu chốt base khác (Qwen 1.5B chẳng hạn) thì cột "dùng ckpt" của nhóm INTRA **sụp hoàn toàn** — xem §Ảnh hưởng của lựa chọn base ở cuối.

> 📏 **Mốc quy chiếu:** một lượt `response` đầy đủ = 300 HarmBench + 250 XSTest + 800 JustEval = **1350 prompt × 512 max_token**. Gọi thời gian của `no_defense` là **T**. Ước lượng thô cho 8B trên MIG 40GB: **T ≈ 1.5–2.5 giờ**. Con số nhân (×T) lấy từ paper nên đáng tin; con số giờ tuyệt đối là **ước lượng của tôi, chưa đo**.

---

## Bảng tổng hợp

### Nhóm IN

| Method | Code | Đủ phần? | Code train | Data train | Model repo hỗ trợ | Chạy lại (ckpt/no-train) | Train thêm | Độ khó |
|---|---|---|---|---|---|---|---|---|
| **ROSE** | ✅ | ⚠️ thiếu | — training-free | — | Baichuan2-7B (qua lmdeploy) | **~2×T** (2 forward/token) | 0 | 🟢🟡 |
| **SafeDecoding** | ✅ | ✅ đủ | ✅ có | 36 query × 2 = **≤72 cặp** | vicuna, llama2, guanaco, falcon, dolphin | **~1.05×T** (ATGR 1.03–1.07) | **<1 phút** | 🟢 |
| **JBShield** | ✅ | ✅ đủ | — không train | calibration set **có sẵn trong repo** | Mistral-7B-v0.2, Llama-2-7b, **Llama-3-8B**, vicuna-7b/13b | **~1.1×T** + calibration vài phút | 0 | 🟡 |
| **SafeInfer** | ✅ | ⚠️ thiếu | — training-free | cần **demonstration examples** tự chuẩn bị | không nêu rõ (dựa trên lib Model Arithmetic) | **~2×T** + **2 model trong VRAM** | 0 | 🟡🔴 |
| **DRO** | ✅ | ✅ đủ | ✅ có | data trong repo (`data/`, `data_harmless/`) | **chỉ Mistral-v1** có script sẵn | ~1×T (chỉ prepend soft prompt) | ⚠️ **chưa xác nhận** (~vài chục phút) | 🟡 |

### Nhóm INTRA

| Method | Code | Đủ phần? | Code train | Data train | Model repo hỗ trợ | Chạy lại (dùng ckpt) | Train lại | Độ khó |
|---|---|---|---|---|---|---|---|---|
| **CAT / CAPO** | ✅ | ⚠️ thiếu code eval | ✅ có | HarmBench AT set + UltraChat200k | Gemma2B, Phi3, Mistral7B, Zephyr7B, Llama2-7B (+ **ckpt Llama3-8B**) | **1×T** | **~42 phút** (CAT) / **~19 phút** (CAPO) | 🟢 |
| **Circuit Breakers** | ✅ | ✅ đủ | ✅ notebook | CB set (tự sinh) + retain = **UltraChat + XSTest** | Mistral-7B-v2, **Llama-3-8B** | **1×T** | **~20 phút** (150 step, 1×A100-80GB) | 🟢 |
| **DeRTa** | ✅ | ✅ đủ | ✅ full + LoRA | `llama_derta`/`vanilla`/`recaug`, có `generate_training_data.py` | **Llama-3-8B/70B** (+Instruct) | **1×T** | ⚠️ repo cấu hình **8 GPU DeepSpeed**, 2 epoch | 🟢 (ckpt) / 🔴 (train) |
| **DeepRefusal** | ⚠️ mỏng | ❌ thiếu lệnh train | ⚠️ có `src/`, không có lệnh | CB 2k + UltraChat 4k + XSTest/Or-bench 500 | 4 họ (Llama3-8B, Llama2-7B, Mistral, Gemma) | **1×T** | **~45 phút** (1×A100-80GB, LoRA) | 🟢 (ckpt) / 🟡 (train) |
| **Targeted LAT** | ✅ | ⚠️ notebook, không phải CLI | ✅ notebook | không nêu rõ trong README | Llama-3-8B (qua ckpt LLM-LAT) | **1×T** | ~36× rẻ hơn R2D2 (chưa có số tuyệt đối) | 🟢 (ckpt) / 🟡 |

**Đọc bảng:** nhóm INTRA thắng tuyệt đối ở cột "chạy lại" — **đúng 1×T**, vì sau khi train xong nó chỉ là một model bình thường, **không có overhead lúc infer**. Nhóm IN thì mỗi request đều phải trả thêm chi phí, ROSE và SafeInfer đắt gấp đôi.

---

## NHÓM IN — chi tiết

### 1. ROSE — Findings ACL 2024

**Repo:** https://github.com/WHU-ZQH/ROSE · **Paper:** https://aclanthology.org/2024.findings-acl.814/

| Câu hỏi | Trả lời |
|---|---|
| Có code? | ✅ Có. Thư mục `datasets/` + `src/` |
| Đủ phần? | ⚠️ **Thiếu**. `src/` có script inference (vd `dangerousqa_inference.py`) nhưng viết quanh **lmdeploy**, không phải HF `generate()` thuần. Không có CLI tổng quát |
| Code train? | Không cần — **training-free hoàn toàn** |
| Data train? | Không có |
| Model nào? | Repo demo **Baichuan2-7B**; model HF khác phải convert qua lmdeploy. Paper đánh **5 loại instruction-tuned LLM**, 6 task safety + 2 task general |
| Chạy lại tốn bao lâu? | **~2×T** — mỗi token cần 2 forward (normal + reverse-prompt) |
| Train thêm? | **0** |
| Độ khó | 🟢🟡 **Thấp về khái niệm, trung bình về kỹ thuật** |

**Cơ chế:** contrastive decoding. `logit_cuối = logit(prompt bình thường) − w · logit(reverse prompt độc)`. Reverse prompt là prompt được thiết kế để **cố tình dụ model trả lời độc** — trừ nó đi thì phần "xu hướng độc" bị triệt tiêu. Paper báo tăng tới **+13.8% safety score** mà không mất năng lực chung.

**Việc thật sự phải làm:** repo bám lmdeploy, còn pipeline mình dùng HF `transformers`. Cơ chế thì đơn giản (chỉ là 2 lần forward rồi trừ logit), nên **tự viết lại bằng HF khoảng 50–80 dòng** có khi nhanh hơn là vật lộn với lmdeploy. Cái phải lấy verbatim từ repo là **nội dung reverse prompt** — đó mới là thứ quyết định con số.

**Vì sao nên chạy đầu tiên trong nhóm IN:** không train, không model phụ, không calibration. Dựng xong là có ngay một điểm dữ liệu, và đường ống local (load model → hook decoding → sinh → chấm) chạy thông thì 4 bài sau chỉ là thay phần can thiệp.

---

### 2. SafeDecoding — ACL 2024

**Repo:** https://github.com/uw-nsl/SafeDecoding

| Câu hỏi | Trả lời |
|---|---|
| Có code? | ✅ Có, và **đầy đủ nhất nhóm IN** |
| Đủ phần? | ✅ `defense.py` + cài sẵn **6 baseline** (PPL, Self-Examination, Paraphrase, Retokenization, Self-Reminder, ICD) + thư mục `lora_modules/` chứa expert |
| Code train? | ✅ Có (train expert), tuy README không mô tả kỹ data gốc |
| Data train? | **Cực nhỏ**: 36 harmful query (18 category, từ Ganguli et al. 2022) × 2 response = **≤72 cặp**. Response từ chối do **chính model sinh** (top-p 0.9, temp 0.7), GPT-4 verify |
| Model nào? | **vicuna, llama2, guanaco, falcon, dolphin** — cả 5 đều **có LoRA expert sẵn trong repo** |
| Chạy lại tốn bao lâu? | **~1.05×T** — ATGR đo được 1.03× (Llama2) và 1.07× (Vicuna) |
| Train thêm? | **< 1 phút/model** (LoRA r=16, α=64, 2 epoch, bs=1, lr=2e-3) |
| Độ khó | 🟢 **Thấp — dễ nhất nhóm IN xét tổng thể** |

**Cơ chế:** mỗi bước decode, lấy top-k của base và top-k của expert → **giao nhau** làm sample space → `P_n(x) = p_θ(x) + α(p_expert(x) − p_θ(x))` với **α=3**, chỉ áp cho **m=2 token đầu**, sau đó greedy bình thường. Lý do chỉ 2 token đầu: "safety disclaimer" nằm ngay đầu response, quyết định xong là xong.

**Điểm cần lưu ý:** không có expert cho Llama-3. Nếu chốt base Llama-3-8B thì **phải tự train expert** — nhưng đây là việc nhẹ nhất trong toàn bộ survey: sinh 72 câu từ chối rồi LoRA dưới 1 phút. Repo có `--GPT_API` để chấm harmful, mình **bỏ**, dùng `metrics/harmbench.py`.

---

### 3. JBShield — USENIX Security 2025

**Repo:** https://github.com/NISPLab/JBShield

| Câu hỏi | Trả lời |
|---|---|
| Có code? | ✅ Có, kèm script shell chạy sẵn |
| Đủ phần? | ✅ `interpret.sh` (phân tích concept) + `evaluate_detection.sh` + `evaluate_mitigation.sh` + `mitigation.py` |
| Code train? | Không cần — **không train model nào** |
| Data train? | Không train, nhưng cần **calibration set** — **đã có sẵn trong repo** `./data/jailbreak/`: jailbreak prompt của **9 loại attack × 5 LLM**, benign từ **Alpaca**, harmful từ **AdvBench + Hex-PHI**, chia sẵn calibration/test |
| Model nào? | Mistral-7B-Instruct-v0.2, Llama-2-7b-chat, **Meta-Llama-3-8B-Instruct**, vicuna-7b-v1.5, vicuna-13b-v1.5 |
| Chạy lại tốn bao lâu? | **~1.1×T** + **calibration một lần vài phút**. Detection = 1 forward trên prompt (rẻ), mitigation = hook cộng/trừ vector lúc sinh (gần như miễn phí) |
| Train thêm? | **0** |
| Độ khó | 🟡 **Trung bình** |

**Cơ chế:** dựa trên Linear Representation Hypothesis. Tách **toxic concept** (có ở cả harmful lẫn jailbreak prompt) và **jailbreak concept** (chỉ có ở jailbreak prompt, chính nó lật model từ *từ chối* sang *tuân theo*). **JBShield-D**: prompt kích hoạt cả hai → gắn cờ (accuracy 0.95). **JBShield-M**: cộng anchor vector của toxic subspace, trừ anchor vector của jailbreak subspace → **ASR 61% → 2%**.

**Hardware:** repo yêu cầu tối thiểu **2 GPU ≥24GB**, khuyến nghị 4×RTX4090 hoặc 1×A100-80GB. Với MIG 40GB: 7B/8B chạy được, **vicuna-13b thì chật**.

**⚠️ Caveat lớn nhất:** JBShield học jailbreak concept **theo từng loại attack** từ calibration set. Nhưng `harmbench_300.csv` của mình là **harmful prompt thô, không bọc jailbreak template** → jailbreak concept gần như không kích hoạt, method chỉ còn hoạt động qua toxic concept. Phải quyết trước: (a) chấp nhận và ghi rõ, hay (b) thêm một tập HarmBench có bọc template tấn công.

---

### 4. SafeInfer — AAAI 2025

**Repo:** https://github.com/NeuralSentinel/SafeInfer

| Câu hỏi | Trả lời |
|---|---|
| Có code? | ✅ Có: `MA_Inference.py` (SafeInfer chính), `FV_Inference.py` (baseline function vector), `Vanilla_Output.py` (baseline), notebook `Function_Vector_Creation.ipynb` |
| Đủ phần? | ⚠️ **Thiếu** — README không liệt kê model hỗ trợ, phải đọc code. Dựa trên thư viện **Model Arithmetic** |
| Code train? | Không cần — **training-free** |
| Data train? | Không train, **nhưng phải tự chuẩn bị "safe demonstration examples"** để sinh function vector (bước Safety Amplification). Repo không đóng gói sẵn bộ này. Benchmark của họ là **HarmEval** (~550 harmful query, 11 category) |
| Model nào? | Không nêu rõ; kế thừa từ lib Model Arithmetic (transformer chuẩn) |
| Chạy lại tốn bao lâu? | **~2×T**, và quan trọng hơn: **cần 2 model cùng lúc trong VRAM** (target + `M_unsafe`) |
| Train thêm? | **0** |
| Độ khó | 🟡🔴 **Trung bình–cao** |

**Cơ chế:** 2 pha lúc decode. (1) **Safety Amplification** — dịch hidden state theo hướng an toàn, hướng này trích offline từ demonstration examples. (2) **Bias mitigation** — dùng model arithmetic phối logit với một con `M_unsafe`.

**Nút thắt thật sự = VRAM.** Hai model 8B bf16 = ~32GB, cộng KV cache và activation trên **MIG 40GB thì rất chật**. Cách gỡ: quantize 4-bit con `M_unsafe`, hoặc chọn `M_unsafe` nhỏ hơn (nhưng lệch so với paper → phải khai báo). Cộng thêm việc phải tự dựng bộ demonstration → đây là **bài tốn công nhất nhóm IN**.

---

### 5. DRO — ICML 2024 *(On Prompt-Driven Safeguarding for LLMs)*

**Repo:** https://github.com/chujiezheng/LLM-Safeguard · **Paper:** https://arxiv.org/abs/2401.18018

| Câu hỏi | Trả lời |
|---|---|
| Có code? | ✅ Có, cấu trúc gọn gàng nhất |
| Đủ phần? | ✅ `train.py`, `train_unlikelihood.py` (baseline prompt-tuning), `forward.py`, `generate.py`, `evaluate.py`, `estimate.py` + `chat_templates/` + `scripts/` bash sẵn |
| Code train? | ✅ Có. Quy trình: `bash scripts/forward.sh` → `bash scripts/train_mistral-v1.sh` |
| Data train? | ✅ Có sẵn trong repo: `data/` + `data_harmless/`. Data thí nghiệm đầy đủ để ở repo companion riêng |
| Model nào? | ⚠️ script sẵn **chỉ cho Mistral-v1**. Model khác phải tự thêm chat template + config (`HF_MODELS` env var) |
| Chạy lại tốn bao lâu? | **~1×T** — deploy chỉ là prepend một soft prompt đã train, không có overhead decoding |
| Train thêm? | ⚠️ **README không ghi runtime.** Ước lượng: prompt tuning trên data nhỏ → **vài chục phút**, nhưng **chưa xác nhận** |
| Độ khó | 🟡 **Trung bình** |

**Cơ chế:** quan sát rằng model đã "biết" phân biệt harmful/harmless trong không gian biểu diễn, chỉ là ranh giới chưa đủ tách. **DRO tối ưu một soft prompt liên tục** sao cho: biểu diễn của prompt harmful bị đẩy **theo** hướng từ chối, prompt harmless bị đẩy **ngược lại**. Deploy = prepend soft prompt đó.

**⚠️ Vấn đề phân loại:** soft prompt = embedding được train, prepend vào input → xét chặt thì đây là **pre (optimized-prompt, cùng nhóm RPO / Prompt-Tuning)** hơn là in. `PHUONG_PHAP_MOI.md` L119 cũng đã tự ghi *"Borderline in/pre"*. Hai lựa chọn: giữ ở IN và định nghĩa "in = can thiệp tầng biểu diễn/decoding" ngay đầu survey, hoặc chuyển sang PRE rồi tìm bài IN khác thay. Nếu chuyển thì ứng viên thay thế yếu — **Self-CD** chỉ giảm over-refusal chứ không chống jailbreak; **Jailbreak Antidote / InferAligner / SafeInt** đều **không có repo**.

---

## NHÓM INTRA — chi tiết

Đặc điểm chung khiến cả nhóm này **dễ hơn nhóm IN nếu dùng checkpoint**: sau khi train xong nó chỉ là một model bình thường → **inference overhead = 0**, chạy đúng như `no_defense`, không phải viết hook decoding nào cả. Toàn bộ độ khó dồn vào bước train, mà bước đó thì **bỏ qua được nếu có ckpt**.

### 1. CAT / CAPO — NeurIPS 2024 Spotlight

**Repo:** https://github.com/sophie-xhonneux/Continuous-AdvTrain (MIT)

| Câu hỏi | Trả lời |
|---|---|
| Có code? | ✅ `config/` · `data/` · `src/` · `requirements.txt` |
| Đủ phần? | ⚠️ **thiếu code eval** (mình có sẵn rồi nên không sao); config path phải tự tạo từ `example_path.yaml` |
| Code train? | ✅ `python src/run_experiments.py --config-name=adv_train_ul path=example_path` (đổi `adv_train_ipo` cho CAPO), Hydra, override kiểu `adversarial.eps=0.075` |
| Data train? | ✅ **AT dataset của HarmBench** (safe answer cố định `"Sorry, I can't do that."`) + **UltraChat200k** làm utility. CAPO **không cần** utility data. Nguồn gốc data đã nằm trong `data/` |
| Model nào? | Paper: Gemma 2B, Phi-3-Mini 3.8B, Mistral-7B, Zephyr-7B, Llama2-7B. **Ckpt HF (org `ContinuousAT`)**: Phi-CAT, Phi-CAPO, Zephyr-CAT (LoRA), Llama-2-7B-CAT, ⭐ **Llama3-8B-IT-CAT** (8B params, safetensors F32 — có vẻ full weight merged) |
| Chạy lại (ckpt)? | **1×T** |
| Train lại? | **CAT ~42 phút** (780 iter × 3.2 s/step) · **CAPO ~19 phút** (360 iter). LoRA toàn bộ linear layer + **quantize 4-bit**, 10 attack iteration/step, ε=0.05–0.1 |
| Độ khó | 🟢 **Thấp cả hai chiều** |

⚠️ Con số **"≥1904 GPU hours"** trong paper là **tổng TẤT CẢ thí nghiệm** (5 model × train + chạy GCG/AutoDAN/PAIR để đánh giá — attack mới là thứ ngốn GPU). Một lần train chỉ 42 phút. Pipeline mình **không chạy attack** nên né hẳn phần đó.
✅ **Hardware đã chứng minh:** paper ghi rõ cluster gồm **A100 40GB** → 40GB đủ.
✅ **Không nhiễm XSTest** — over-refusal họ đo bằng bộ HARMLESS 40 câu tự viết.
⚠️ Safe answer khi train cố định một câu → model có xu hướng từ chối cụt lủn, ảnh hưởng **JustEval** (engagement/depth) chứ không ảnh hưởng ASR.

**Đây là bài duy nhất trong 10 bài mà mình vừa có ckpt vừa train lại được thoải mái trên đúng phần cứng đang có.**

---

### 2. Circuit Breakers (RepE) — NeurIPS 2024

**Repo:** https://github.com/GraySwanAI/circuit-breakers (MIT) · **Paper:** https://arxiv.org/abs/2406.04313

| Câu hỏi | Trả lời |
|---|---|
| Có code? | ✅ |
| Đủ phần? | ✅ notebook train + thư mục `evaluation/` |
| Code train? | ✅ `train_cb_llama3_8b.ipynb` và `train_cb_mistral_7b.ipynb` — **có sẵn notebook cho đúng Llama-3-8B** |
| Data train? | **Circuit breaker set**: sinh bằng cách prompt một LLM uncensored ra harmful query+completion nhiều category, lọc trùng HarmBench (BLEU < 0.3). **Retain set**: **UltraChat + XSTest** (Llama-3 có thêm refusal data) |
| Model nào? | **Mistral-7B-Instruct-v2**, **Llama-3-8B-Instruct**, LLaVA-NeXT-Mistral-7B (multimodal). Ckpt: `GraySwanAI/Llama-3-8B-Instruct-RR` |
| Chạy lại (ckpt)? | **1×T** |
| Train lại? | **~20 phút**, **150 step, batch 16, 1×A100-80GB**. LoRA vào tất cả linear layer của layer 0–20, loss circuit-breaking áp ở layer 10 và 20. α=5 (Mistral), α=10 (Llama-3) |
| Độ khó | 🟢 **Thấp** |

**Cơ chế:** thay vì dạy model từ chối, nó **bẻ gãy chính biểu diễn nội bộ** dẫn tới output có hại — model "chập mạch" giữa chừng thay vì hoàn thành câu trả lời độc. Đánh giá cực rộng: GCG, PAIR, TAP-Transfer, AutoDAN, multilingual, prefilling, input embedding optimization, RepE manipulation, human jailbreak của HarmBench.

⚠️ **Retain set có XSTest** → dùng ckpt rồi đo over-refusal bằng XSTest thì **có nguy cơ nhiễm**, y hệt DeepRefusal. Ghi chú lại khi báo cáo.
⚠️ Paper DeepRefusal chỉ ra CircuitBreaker **thua rõ ở refusal-transfer attack** trên Llama3-8B (ASR 48.0) và làm **tụt GSM8k mạnh** (42.84 vs base 75.44, do sinh ra output vô nghĩa). Đây là điểm hay để bàn trong survey: circuit breaker đánh đổi utility nhiều hơn các bài khác.

---

### 3. DeRTa — ACL 2025

**Repo:** https://github.com/RobustNLP/DeRTa

| Câu hỏi | Trả lời |
|---|---|
| Có code? | ✅ |
| Đủ phần? | ✅ train + eval + script sinh data |
| Code train? | ✅ **cả 2 đường**: `run_clm_llms_derta_llama_drop_5_percent.py` (full-param) và `run_clm_lora_derta_llama.py` (LoRA). Chạy `bash train.sh` → `bash evaluation.sh` |
| Data train? | ✅ `generate_training_data.py` **tự sinh tại chỗ**, ra JSON trong `data/train/`: 3 bộ `llama_derta` / `llama_vanilla` / `llama_recaug`, gồm helpfulness examples, safety instruction pairs, harmful response prefixes, transition optimization samples |
| Model nào? | **Meta-Llama-3-8B / 70B** + bản Instruct — đúng họ mình cần |
| Chạy lại (ckpt)? | **1×T**. Ckpt: **5 model trên HF (tài khoản `Youliang`)**, cả full-param lẫn LoRA, cho 8B và 70B |
| Train lại? | ⚠️ repo cấu hình cho **8 GPU + DeepSpeed**, 2 epoch, batch 16 (full) / batch 8 (LoRA). **Không ghi thời gian.** Trên 1×MIG 40GB phải hạ batch + grad accumulation, chưa ước được |
| Độ khó | 🟢 **Thấp nếu dùng ckpt LoRA** / 🔴 **cao nếu tự train** (cấu hình 8 GPU) |

**Cơ chế:** dạy model **từ chối giữa chừng**. Bình thường alignment chỉ dạy từ chối ở token đầu; DeRTa prepend sẵn một đoạn **harmful prefix** rồi dạy model bẻ lái sang từ chối từ giữa câu (RTO — Reinforced Transition Optimization). Nhờ vậy **chống prefilling attack** rất tốt — đúng cái điểm yếu mà alignment nông hay bị khai thác.

---

### 4. DeepRefusal — Findings EMNLP 2025

**Repo:** https://github.com/YuanBoXie/DeepRefusal (MIT) — *(chi tiết đầy đủ ở `README_PHUONG_PHAP_INTRA.md` §1)*

| Câu hỏi | Trả lời |
|---|---|
| Có code? | ⚠️ Có `src/` nhưng **README rất mỏng** |
| Đủ phần? | ❌ **Không có lệnh train**, không nêu dataset trong README. Tác giả trỏ sang [refusal_direction](https://github.com/andyrdt/refusal_direction) và circuit-breakers để lấy **code** trích refusal direction + code eval |
| Code train? | ⚠️ một phần — phải ghép từ 3 repo |
| Data train? | ✅ **paper ghi đủ, đều public**: 2,000 harmful từ CircuitBreaker (**bắt buộc áp prefill augmentation**) + 4,000 benign từ UltraChat + 500 từ XSTest/Or-bench |
| Model nào? | Llama3-8B-instruct, Llama2-7B-instruct, Mistral-7B-v0.2, Gemma-7B-it. Ckpt: `skysys00/Meta-Llama-3-8B-Instruct-DeepRefusal` |
| Chạy lại (ckpt)? | **1×T** |
| Train lại? | **~45 phút**, 1 epoch, **1×A100-80GB**, LoRA α=16 r=16, batch 16, PAA p=0.5 |
| Độ khó | 🟢 (ckpt) / 🟡 (train — data dễ, nhưng **tốn công ghép code từ 3 repo**) |

⚠️ **Over-refusal 28.5%** ở p=0.5 (chính họ thừa nhận điểm này yếu) và **ckpt đã train trên 500 sample XSTest** → nguy cơ nhiễm khi đo XSTest.
✅ Bù lại: ASR đẹp nhất bảng (CodeAttack 87.1% → 0.2%) và **Table 1 của họ có sẵn số cho LAT, CAT, CircuitBreaker trên đúng Llama3-8B** → dùng để **kiểm tra chéo pipeline của mình**.

---

### 5. Targeted LAT — arXiv 2407.15549 (2024)

**Repo:** https://github.com/aengusl/latent-adversarial-training · **Ckpt:** org HF [`LLM-LAT`](https://huggingface.co/LLM-LAT)

| Câu hỏi | Trả lời |
|---|---|
| Có code? | ✅ |
| Đủ phần? | ⚠️ **toàn bộ là notebook**, không có CLI. `/notebooks` gồm latent space attack, jailbreak robustness, backdoor removal, harry potter unlearning, wmdp unlearning |
| Code train? | ✅ trong notebook |
| Data train? | ⚠️ **README không nêu rõ** — phải mở notebook đọc |
| Model nào? | Ckpt `LLM-LAT/robust-llama3-8b-instruct`, `LLM-LAT/llama3-8b-instruct-rt-jailbreak-robust2/3` *(cần verify khi tải)* |
| Chạy lại (ckpt)? | **1×T** |
| Train lại? | RT-EAT-LAT dùng **~36× ít GPU hour hơn R2D2** — nhưng **chưa có số tuyệt đối** |
| Độ khó | 🟢 (ckpt) / 🟡 (train — phải bóc từ notebook) |

**Cơ chế:** adversarial training nhưng perturbation đặt ở **không gian ẩn (residual stream activation)** thay vì token rời rạc — cùng tinh thần "né search rời rạc cho rẻ" với CAT, khác chỗ CAT perturb ở **embedding đầu vào** còn LAT perturb ở **activation giữa các layer**. Cài đặt: `pip install -r requirements.txt` rồi `bash install_tasks_from_github.sh`.

**Vai trò trong bảng:** nó là **baseline trong Table 1 của DeepRefusal** (LAT trên Llama3-8B: GCG 2.0, Prefilling 0.0, nhưng Refusal-Transfer 87.5 — thua hẳn). Có nó thì bảng INTRA phủ đủ 3 biến thể adversarial training: **rời rạc (R2D2, không chạy) → embedding (CAT) → latent (LAT)**.

---

## Tổng hợp chi phí

### Nếu KHÔNG train gì (dùng ckpt / training-free)

| Nhóm | Tổng chi phí sinh response |
|---|---|
| **INTRA (5 bài)** | 5 × 1×T = **~5T** ≈ 8–13 giờ |
| **IN (5 bài)** | ROSE 2T + SafeDecoding 1.05T + JBShield 1.1T + SafeInfer 2T + DRO 1T = **~7.15T** ≈ 11–18 giờ |
| **Cộng** | **~12T** ≈ **19–31 giờ GPU** |

Cộng thêm: chấm HarmBench bằng classifier Llama-2-13b (~26GB, chạy riêng), chấm XSTest + JustEval qua API Groq.

### Nếu train lại (chỉ những bài có số xác nhận)

| Bài | Thời gian train | Ghi chú |
|---|---|---|
| SafeDecoding expert | **< 1 phút** | 72 sample |
| Circuit Breakers | **~20 phút** | 150 step, A100-80GB |
| CAPO | **~19 phút** | 360 iter |
| CAT | **~42 phút** | 780 iter |
| DeepRefusal | **~45 phút** | 1 epoch, A100-80GB |
| DRO soft prompt | ⚠️ chưa xác nhận | ước vài chục phút |
| DeRTa | ⚠️ chưa xác nhận | repo cấu hình 8 GPU |
| Targeted LAT | ⚠️ chưa xác nhận | ~36× rẻ hơn R2D2 |

→ **Tổng phần đã xác nhận ≈ 2 giờ 7 phút.** Train lại rẻ đến bất ngờ, **so với ~20–30 giờ chỉ để sinh response**. Nút thắt thật sự của survey **không phải train, mà là inference + chấm điểm**.

⚠️ Nhưng số này đo trên **A100-80GB**. Trên MIG 40GB phải hạ batch (16 → 4) + gradient accumulation → nhân lên khoảng **2–4×**. Vẫn nằm trong tầm nửa buổi.

---

## Ảnh hưởng của lựa chọn base model

| Nếu chốt base | Nhóm INTRA | Nhóm IN |
|---|---|---|
| **Meta-Llama-3-8B-Instruct** | ✅ **cả 5 ckpt dùng được ngay**, khỏi train | ✅ JBShield có sẵn Llama-3-8B · ⚠️ SafeDecoding phải tự train expert (<1 phút) · ROSE/SafeInfer/DRO phải tự port |
| **Llama-3.1-8B-Instruct** | ❌ **mất cả 5 ckpt**, phải train lại từ đầu (nhưng CAT/CB/DeepRefusal đều có code train + data public → khả thi, ~2–8 giờ tổng) | tương tự cột trên |
| **Qwen2.5-1.5B-Instruct** (đang ghi trong `02_QUY_UOC_MODEL.md`) | ❌ **mất cả 5 ckpt** VÀ **lệch khỏi mọi số liệu paper** (không paper nào chạy 1.5B) → không đối chiếu được với ai | ✅ nhẹ VRAM, SafeInfer 2 model dễ thở |

**Nhận xét:** Qwen 1.5B rẻ nhất về tính toán nhưng khiến toàn bộ bảng **không so được với bất kỳ con số công bố nào**. Llama-3-8B-Instruct đắt hơn nhưng cho **5 checkpoint miễn phí + một bảng đối chiếu công bố sẵn (DeepRefusal Table 1)**. Nếu VRAM là lo ngại chính thì nhớ: 8B bf16 ≈ 16GB, vừa MIG 40GB thoải mái — chỉ **SafeInfer (2 model)** là chật, và nó xử lý được bằng cách quantize 4-bit con phụ.
