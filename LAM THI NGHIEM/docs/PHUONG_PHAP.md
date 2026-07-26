# Phương pháp phòng thủ
**Mục tiêu:** 5 phương pháp mỗi nhóm — **5 pre · 5 post · 5 in · 5 intra**, cộng `no_defense` làm mốc.

---

## Cách đọc bảng

| Cột | Nghĩa |
|---|---|
| **Ưu tiên** | `✅` đã có kết quả · `🔧` đã code, chưa chạy full · `1` `2` `3` thứ tự nên làm tiếp · `✕` đã loại |
| **GitHub** | repo chính thức. `✕` = không có repo → phải tự viết từ paper |
| **Code** | `✅` đủ chạy · `⚠️` thiếu phần nào đó (ghi rõ) · `✕` repo rỗng |
| **Data** | `✅` có sẵn trong repo · `⚠️` phải tự lo · `—` không cần (training-free) |
| **Flow** | các bước một request đi qua, và **tốn mấy call / mấy forward** |

Còn thiếu: pre **1** · post **1** · in **3** · intra **2**.

---

## 1. PRE — can thiệp ở INPUT

| Ưu tiên | Method | Venue | GitHub | Code | Data | Flow chạy cơ bản |
|:--:|---|---|---|:--:|:--:|---|
| ✅ | **SAGE** | Findings ACL 2025 | [NJUNLP/SAGE](https://github.com/NJUNLP/SAGE) | ✅ | — | Ghép 2 instruction (analysis + response) vào input → sinh **1 call** |
| ✅ | **IA** | COLING 2025 | [alphadl/SafeLLM_with_IntentionAnalysis](https://github.com/alphadl/SafeLLM_with_IntentionAnalysis) | ✅ | — | Lượt 1 phân tích ý định → ghép đoạn phân tích vào hội thoại → lượt 2 trả lời. **2 call** |
| ✅ | **G4D** | Findings NAACL 2025 | [IDEA-XL/G4D](https://github.com/IDEA-XL/G4D) | ✅ | — | intent detect → paraphrase (chỉ khi nghi) → safety analyze → ghép cả 3 thành "guidance" chèn vào prompt → target trả lời. **3–4 call**, retrieval TẮT |
| ✅ | **erase-and-check** | arXiv 2023 | [aounon/certified-llm-safety](https://github.com/aounon/certified-llm-safety) | ✅ | ⚠️ weight DistilBERT tải Dropbox ~256MB | Xoá 1..20 token cuối → **DistilBERT (66M, local)** chấm từng biến thể → chỉ cần **một** biến thể harmful là chặn thẳng, không gọi target. **≤21 forward + 1 call nếu pass** |
| **1** | **Self-Reminder** | Nature MI 2023 | [yjw1029/Self-Reminder](https://github.com/yjw1029/Self-Reminder) | ✅ prompt-only | — | Bọc query bằng system prompt an toàn + câu nhắc ở cuối → **1 call**. Rẻ nhất, copy là chạy |
| **2** | **Goal Prioritization** | ACL 2024 | [thu-coai/JailbreakDefense_GoalPriority](https://github.com/thu-coai/JailbreakDefense_GoalPriority) | ✅ | — | Chèn chỉ dẫn "ưu tiên an toàn hơn hữu ích" + template internal-thought → **1 call**. Paper: ASR 66% → 3.6% |
| 3 | **Paraphrase** | arXiv 2023 | [neelsjain/baseline-defenses](https://github.com/neelsjain/baseline-defenses) | ⚠️ retokenization không chạy qua API | — | Viết lại prompt (phá suffix GCG) rồi mới đưa target. **2 call** |
| 4 | **SmoothLLM** | arXiv 2023 | [arobey1/smooth-llm](https://github.com/arobey1/smooth-llm) | ✅ | — | Tạo N bản nhiễu ký tự → chạy cả N → vote refusal. **N=6–10 call**, đắt |
| 4 | **FJD** | Findings EMNLP 2025 | [GuoruiC/FJD](https://github.com/GuoruiC/FJD) | ✅ | ⚠️ bản FJD-LI cần train nhỏ | Prepend affirmative instruction → chạy **đúng 1 token** → xét confidence token đầu để flag → benign mới sinh full |
| 4 | **Perplexity filter** | arXiv 2023 | ✕ không có repo | tự viết (~30 dòng) | — | Tính PPL của prompt, cao bất thường (suffix GCG) → chặn. **1 forward + 1 call nếu pass** |
| 5 | **RPO** | NeurIPS 2024 | [lapisrocks/rpo](https://github.com/lapisrocks/rpo) | ✅ | ✅ advbench trong repo | **Offline:** GCG chạy ngược, tối ưu 1 defensive suffix ~20 token (2000 step, ~vài giờ GPU). **Deploy:** gắn suffix vào cuối prompt → 1 call. Cần **white-box local** |
| 5 | **ICAG** | EMNLP 2024 | [YujunZhou/In-Context-Adversarial-Game](https://github.com/YujunZhou/In-Context-Adversarial-Game) | ⚠️ thiếu `val_inputs.pkl` / `test_set.pkl` | ✅ | **Offline:** chạy game attack↔defense 10 vòng → chưng cất ra 1 safety system prompt (~800 token). **Deploy:** prepend rồi 1 call. Có sẵn prompt trong `data/ICAG_prompts_iter.json` → **train = 0** |
| ✕ | ICD | arXiv 2023 | ✕ repo `PKU-ML/adv-icl` **404** | tự viết ~5 dòng | — | Prepend vài cặp (harmful→refusal) làm few-shot. **Loại:** yếu với many-shot jailbreak |
| ✕ | Proxy Barrier | Findings EMNLP 2025 | ✕ chưa tìm được repo | — | — | **Loại:** không có code |
| ✕ | Prompt-Tuning | ACL 2023 short | [amazon-science/controlling-llm-memorization](https://github.com/amazon-science/controlling-llm-memorization) | ✅ | ⚠️ tự convert từ Pile | **Loại:** chủ đề **privacy** (chống trích xuất dữ liệu đã nhớ), đo bằng reconstruction rate + perplexity, **không có khái niệm ASR/từ chối** → chạy trên HarmBench là sai phạm trù |

**Ghi chú RPO / ICAG** (2 bài duy nhất nhóm pre cần pha offline):
- Cả hai **không đổi trọng số target**, chỉ tạo ra một *artifact* (suffix / system prompt) rồi gắn vào lúc infer → cost infer vẫn ≈ 1 call.
- RPO **bắt buộc GPU** (cần gradient). Việc phải sửa: `SuffixManager`/`string_utils.py` hardcode template `llama-2`/`vicuna` → thêm nhánh chat template Llama-3 để định vị đúng span token của suffix.
- ICAG **chạy được thuần API**, nhưng code gốc hardcode judge = Llama-3-8B local + AutoDAN white-box + OpenAI embedding → phải thay judge sang Groq, bỏ AutoDAN, đổi embedding. Hoặc đi **đường tắt**: lấy luôn prompt có sẵn, train = 0.

---

## 2. POST — can thiệp ở OUTPUT

| Ưu tiên | Method | Venue | GitHub | Code | Data | Flow chạy cơ bản |
|:--:|---|---|---|:--:|:--:|---|
| ✅ | **LLM Self Defense** | arXiv 2023 | [poloclub/llm-self-defense](https://github.com/poloclub/llm-self-defense) | ✅ | — | Target sinh → 1 call hỏi "response này có hại không" → nếu có thì **vứt, thay bằng câu từ chối cố định**. **2 call** |
| ✅ | **Self-Refine** | NeurIPS 2023 | [madaan/self-refine](https://github.com/madaan/self-refine) | ✅ | — | Target sinh → call feedback phê bình → call refine viết lại → bản mới **thay** bản cũ, lặp k vòng. **1 + 2k call** |
| ✅ | **Backtranslation** | Findings ACL 2024 | [YihanWang617/LLM-Jailbreaking-Defense-Backtranslation](https://github.com/YihanWang617/LLM-Jailbreaking-Defense-Backtranslation) | ✅ | — | Target sinh → **suy ngược** từ response ra prompt gốc (lộ intent thật) → hỏi lại target bằng prompt suy ngược → nếu target từ chối thì vứt response ban đầu. **~2–3 call** |
| ✅ | **AutoDefense** | arXiv 2024 | [XHMY/AutoDefense](https://github.com/XHMY/AutoDefense) | ✅ | — | Target sinh → nhiều agent cùng đọc response → bỏ phiếu → có hại thì thay bằng từ chối. **4 call**, đắt nhất nhóm |
| **1** | **WildGuard** | NeurIPS 2024 | [allenai/wildguard](https://github.com/allenai/wildguard) · `pip install wildguard` | ✅ | ✅ | Target sinh → 1 model đọc (prompt, response) → ra **3 nhãn**: prompt-harm · response-harm · **refusal** → đo được **cả 2 trục cùng lúc**. LOCAL 7B |
| **2** | **Llama Guard 3** | Meta 2024 | ⚠️ `PurpleLlama` không có code, chỉ weight | 📦 dùng ckpt | ✅ taxonomy có sẵn | Classifier chấm (prompt, response) theo taxonomy 13 mục → gắn cờ. ⚠️ **Groq đã bỏ `llama-guard-3-8b`** → dùng `openai/gpt-oss-safeguard-20b` hoặc host local |
| 3 | **Aligner** | NeurIPS 2024 Oral | [PKU-Alignment/aligner](https://github.com/PKU-Alignment/aligner) | ⚠️ repo chỉ có code **train** | 📦 ckpt `aligner/aligner-7b-v1.0` | Model nhỏ đứng SAU target, đọc (query, answer) rồi **viết lại** answer an toàn (copy-and-correct). **Transformer rewriter duy nhất** — giữ benign tốt |
| 3 | **ShieldGemma** | Google 2024 | 📦 weight `google/shieldgemma-2b` | 📦 | ✅ | Chấm output ra **xác suất** (threshold chỉnh được → vẽ được đường trade-off). Bản 2B rẻ |
| ✕ | RC-RAG | Findings EMNLP 2024 | [ict-bigdatalab/RC-RAG](https://github.com/ict-bigdatalab/RC-RAG) | ✅ | ✅ | **Loại:** phải dựng RAG/retrieval, lệch hạ tầng hiện có |
| ✕ | SelfCheckGPT | EMNLP 2023 | [potsawee/selfcheckgpt](https://github.com/potsawee/selfcheckgpt) | ✅ | ✅ | **Loại:** chủ đề **hallucination detection**, không phải jailbreak |

---

## 3. IN — can thiệp lúc decoding (tạm thời)

Toàn bộ **bắt buộc chạy local** — phải đọc/sửa logits hoặc hidden state, API không xen vào decode được. Cột cuối ghi **overhead**: T = thời gian của `no_defense`.

| Ưu tiên | Method | Venue | GitHub | Code | Data | Flow chạy cơ bản · overhead |
|:--:|---|---|---|:--:|:--:|---|
| 🔧 | **SafeDecoding** | ACL 2024 | [uw-nsl/SafeDecoding](https://github.com/uw-nsl/SafeDecoding) | ✅ đủ nhất nhóm | ✅ 72 cặp + expert LoRA sẵn 5 model | Mỗi bước decode: lấy top-k của base ∩ top-k của expert → `p = p_base + α(p_expert − p_base)`, α=3, **chỉ áp 2 token đầu** rồi greedy bình thường. **2 forward/token** ở 2 bước đầu. Paper: **ATGR 1.03–1.07×** |
| 🔧 | **JBShield** | USENIX Sec 2025 | [NISPLab/JBShield](https://github.com/NISPLab/JBShield) | ✅ + script shell | ✅ calibration set trong repo (9 attack × 5 LLM) | Calibrate 1 lần ra concept vector → lúc sinh, hook cộng anchor vector của toxic subspace và trừ của jailbreak subspace vào hidden state. **Hook chạy 1 SVD mỗi forward** → phải đo thực tế |
| **1** | **ROSE** | Findings ACL 2024 | [WHU-ZQH/ROSE](https://github.com/WHU-ZQH/ROSE) | ⚠️ viết quanh **lmdeploy**, không phải HF `generate()` | — training-free | Contrastive decoding: `logit = logit(prompt thường) − w · logit(reverse prompt độc)`. **2 forward/token → ~2×T**. Train = 0. Tự viết lại bằng HF ~50–80 dòng có khi nhanh hơn vật lộn với lmdeploy; thứ phải lấy verbatim là **nội dung reverse prompt** |
| **2** | **DRO** | ICML 2024 | [chujiezheng/LLM-Safeguard](https://github.com/chujiezheng/LLM-Safeguard) | ✅ gọn nhất nhóm | ✅ `data/` + `data_harmless/` | Offline tối ưu **soft prompt** sao cho biểu diễn prompt harmful bị đẩy *theo* hướng từ chối, harmless bị đẩy ngược. Deploy = prepend soft prompt → **~1×T**, không overhead decoding. ⚠️ script sẵn **chỉ cho Mistral-v1** |
| **3** | **SafeInfer** | AAAI 2025 | [NeuralSentinel/SafeInfer](https://github.com/NeuralSentinel/SafeInfer) | ⚠️ README không nêu model hỗ trợ, phải đọc code | ⚠️ phải tự chuẩn bị demonstration examples | 2 pha lúc decode: (1) dịch hidden state theo hướng an toàn trích từ demonstration, (2) phối logit với một con `M_unsafe`. **~2×T + cần 2 model cùng lúc trong VRAM** (~32GB, chật trên MIG 40GB → phải quantize 4-bit con phụ) |
| 4 | **GeDi** | Findings EMNLP 2021 | [salesforce/GeDi](https://github.com/salesforce/GeDi) | ✅ | ⚠️ | Mỗi bước decode, một discriminator LM reweight xác suất token kế tiếp qua Bayes. Cần **train discriminator** |
| 4 | **TaskTracker** | IEEE SaTML 2025 | [microsoft/TaskTracker](https://github.com/microsoft/TaskTracker) | ✅ | ⚠️ **phải xin qua form** | Trích activation delta → đưa vào probe đã train → phát hiện task drift. Detector, **không sửa output** |
| ✕ | Self-CD | ACL 2024 | — | — | — | **Loại:** chỉ **giảm over-refusal**, không chống jailbreak |
| ✕ | InferAligner | EMNLP 2024 | [Jihuai-wpy/InferAligner](https://github.com/Jihuai-wpy/InferAligner) | ✕ **0 file .py**, "coming soon" từ 1/2024 | ✕ | **Loại:** repo rỗng, phải tự implement từ paper |
| ✕ | SafeInt | Findings EMNLP 2025 | ✕ không có repo | ✕ | ✕ | **Loại:** repo rỗng |
| ✕ | Jailbreak Antidote | ICLR 2025 | ✕ không có repo | ✕ | ✕ | **Loại:** repo rỗng |

> ⚠️ **DRO là borderline in/pre.** Soft prompt = embedding đã train rồi prepend vào input → xét chặt thì cùng họ với RPO (pre). Giữ ở IN thì phải định nghĩa "in = can thiệp tầng biểu diễn/decoding" ngay đầu survey. Nếu chuyển sang PRE thì **không có bài thay thế** — 3 ứng viên còn lại đều repo rỗng.

---

## 4. INTRA — sửa trọng số vĩnh viễn

Đặc điểm chung: train xong nó chỉ là một model bình thường → **overhead infer = 0, đúng 1×T**, không phải viết hook decoding nào. Toàn bộ độ khó dồn vào bước train, mà bước đó **bỏ qua được nếu có checkpoint**.

| Ưu tiên | Method | Venue | GitHub | Code | Data | Flow chạy cơ bản · checkpoint |
|:--:|---|---|---|:--:|:--:|---|
| 🔧 | **CAT / CAPO** | NeurIPS 2024 Spotlight | [sophie-xhonneux/Continuous-AdvTrain](https://github.com/sophie-xhonneux/Continuous-AdvTrain) | ⚠️ thiếu code eval | ✅ HarmBench AT set + UltraChat200k | Adversarial training với nhiễu đặt ở **embedding đầu vào** (liên tục, né search rời rạc của GCG). Sinh response = 1 call thường. Ckpt ✅ `ContinuousAT/Llama3-8B-IT-CAT`. Train lại **~42 phút** (CAPO ~19 phút) |
| 🔧 | **Circuit Breakers** | NeurIPS 2024 | [GraySwanAI/circuit-breakers](https://github.com/GraySwanAI/circuit-breakers) | ✅ + notebook train sẵn cho Llama-3-8B | ✅ CB set tự sinh + retain = UltraChat + XSTest | LoRA + loss RepE **bẻ gãy chính biểu diễn nội bộ** dẫn tới output có hại → model "chập mạch" giữa chừng thay vì hoàn thành câu độc. Ckpt ✅ `GraySwanAI/Llama-3-8B-Instruct-RR`. Train **~20 phút** |
| 🔧 | **DeRTa** | ACL 2025 | [RobustNLP/DeRTa](https://github.com/RobustNLP/DeRTa) | ✅ full + LoRA | ⚠️ **thiếu hẳn file safety**, đã tự tái tạo | Dạy model **từ chối giữa chừng**: prepend sẵn harmful prefix rồi dạy bẻ lái sang từ chối (MLE) + RTO reinforce điểm chuyển ở **mọi vị trí**. Ckpt ✅ `Youliang/llama3-8b-instruct-lora-derta-100step`. Train: repo cấu hình **8 GPU DeepSpeed** |
| **1** | **DeepRefusal** | Findings EMNLP 2025 | [YuanBoXie/DeepRefusal](https://github.com/YuanBoXie/DeepRefusal) | ⚠️ **không có lệnh train**, phải ghép code từ 3 repo | ✅ paper ghi đủ, đều public (CB 2k + UltraChat 4k + XSTest/Or-bench 500) | Phá rồi dựng lại **refusal direction** ở nhiều tầng, có prefill augmentation → từ chối không còn nông ở token đầu. Ckpt ✅ `skysys00/Meta-Llama-3-8B-Instruct-DeepRefusal`. Train **~45 phút** |
| **2** | **Targeted LAT** | arXiv 2407.15549 | [aengusl/latent-adversarial-training](https://github.com/aengusl/latent-adversarial-training) | ⚠️ **toàn notebook, không có CLI** | ⚠️ README không nêu rõ | Adversarial training với nhiễu đặt ở **activation giữa các layer** (latent) thay vì embedding như CAT. Ckpt ✅ org HF `LLM-LAT`. Train ~36× rẻ hơn R2D2 (chưa có số tuyệt đối) |
| 3 | **Safe RLHF** | ICLR 2024 | [PKU-Alignment/safe-rlhf](https://github.com/PKU-Alignment/safe-rlhf) | ✅ | ✅ | Tách reward (hữu ích) + cost (hại), PPO ràng buộc Lagrangian. Ckpt ✅ `PKU-Alignment/beaver-7b-v1.0`. ⚠️ **full RLHF không vừa 40GB** → chỉ dùng được ckpt |
| 4 | **SecAlign** | ACM CCS 2025 | [facebookresearch/SecAlign](https://github.com/facebookresearch/SecAlign) | ✅ | ✅ | DPO trên cặp preference → ra model đã align. ⚠️ threat model = **prompt injection** chứ không hẳn jailbreak |
| ✕ | R2D2 | ICML 2024 | trong [HarmBench](https://github.com/centerforaisafety/HarmBench) | ✅ | ✅ | **Loại:** adversarial training vs GCG, **rất đắt** (chạy GCG mỗi step), base Zephyr |
| ✕ | TAR | ICLR 2025 | [rishub-tamirisa/tamper-resistance](https://github.com/rishub-tamirisa/tamper-resistance) | ✅ | ✅ | **Loại:** threat model = **weight tampering**, không phải prompt jailbreak. Rất đắt |
| ✕ | RMU / WMDP | ICML 2024 | ✅ | ✅ | ✅ | **Loại:** đo bằng **WMDP multiple-choice**, lệch metric ASR/over-refusal |

**5 bài chốt cho INTRA** (CAT · Circuit Breakers · DeRTa · DeepRefusal · Targeted LAT) — **cả 5 đều có checkpoint Llama-3-8B sẵn** → chạy được toàn bộ mà không cần train lần nào.

---

## 5. Kết quả đã chạy

### 5.1 Nhóm API — target `llama-3.1-8b-instant` (Groq)

ASR chấm bằng classifier chính thức `cais/HarmBench-Llama-2-13b-cls`. Over-refusal = XSTest judge2 (`gpt-oss-20b`).

| Method | Nhóm | ASR ↓ | Over-refusal ↓ | Utility ↑ | Call/req |
|---|---|---:|---:|---|---:|
| **no_defense** *(mốc)* | — | 30.7% | 8.0% | chưa chạy | 1.0 |
| SAGE | pre | **0.7%** | 34.8% | chưa chạy | 1.0 |
| IA | pre | 2.0% | 12.4% | chưa chạy | 2.0 |
| G4D | pre | 7.0% | 10.8% | chưa chạy | 4.0 |
| erase-and-check | pre | 14.7% | **8.4%** | chưa chạy | 1.6 + filter |
| Self_Defense | post | 9.7% | 35.6% | chưa chạy | 2.0 |
| Self_Refine | post | 6.3% | 12.0% | chưa chạy | 3.3 |
| Backtranslation | post | 17.0% | 9.6% | chưa chạy | 2.5 |
| AutoDefense | post | 18.7% | 9.2% | chưa chạy | 4.0 |

**Đọc bảng:** ASR thấp nhất chưa chắc tốt nhất — SAGE hạ ASR còn 0.7% nhưng từ chối oan **34.8%**, gấp hơn 4 lần mốc; Self_Defense cũng vậy. Cặp **cả hai cột đều tốt** hiện là IA và Self_Refine.

> ⚠️ **Utility (JustEval) chưa chạy cho bất kỳ method nào.** Không có cột này thì chưa kết luận được trade-off.

Bảng đầy đủ kèm token/cost: **`tools/comparison.md`** — tự sinh bằng `python tools/compare_methods.py`, đây mới là số chính thức. Bảng trên chép tay nên có thể cũ.

### 5.2 Nhóm LOCAL — target `Meta-Llama-3-8B-Instruct` trên GPU

Smoke test `--task harmbench --limit 3` → **6/6 PASS**.

| Method | Nhóm | Checkpoint | s/req | Ghi chú |
|---|---|---|---:|---|
| no_defense_local | — | base thuần | 0.662 | mốc quy chiếu |
| CAT | intra | `ContinuousAT/Llama3-8B-IT-CAT` | 0.317 | trả đúng chuỗi cố định `"Sorry, I can't do that."` |
| JBShield | in | không có ckpt — `calibrate.py` tự sinh | 0.839 | phải chạy `calibrate.py` **trước** |
| DeRTa | intra | base + LoRA `Youliang/...-derta-100step` | 2.567 | từ chối có giải thích, tự nhiên nhất |
| SafeDecoding | in | Llama-2-7b-chat + expert LoRA trong repo | 5.490 | `disable_adapter()` thay được mixed-adapter-batch |
| Circuit Breakers | intra | `GraySwanAI/Llama-3-8B-Instruct-RR` | 6.067 | 1/3 response là **chuỗi vô nghĩa** — đúng cơ chế, không phải bug |

⚠️ **n=3, chưa kết luận được gì.** CAT nhanh hơn baseline vì nó sinh ít token (từ chối một câu), không phải vì model nhanh hơn.

✅ **JBShield tái hiện đúng paper**: detection accuracy trung bình **0.958** trên 9 loại attack (paper báo 0.95).

### 5.3 Train lại trên Llama-3 (smoke-size) — 4/4 PASS

| Method | Script | Thời gian | Adapter tự train nạp lại | Số thứ phải vá |
|---|---|---:|---|:--:|
| SafeDecoding expert | `train_expert.py` | **16.2 s** | ✅ 1.932 s/req | **0** |
| Circuit Breakers | `train_smoke.py` | 8.1 s / 5 step | ✅ 1.121 s/req | 5 |
| CAT | `train_smoke.py` | 2.26 s / 5 step | ✅ 1.022 s/req | 6 |
| DeRTa | `train_smoke.py` | 3.41 s / 5 step | ✅ 1.006 s/req | 6 |

Bốn nhóm lỗi: **transformers v4→v5** (`deepspeed` alias bị bỏ · `fsdp=None` · `tokenizer=`→`processing_class` · `num_items_in_batch` · `is_torch_tpu_available` · `--eval_strategy`) · **trl 1.9 xoá `DataCollatorForCompletionOnlyLM`** (CAT *kế thừa* nó → phải venv riêng `.venv_cat`) · **MIG** (NCCL chết → bỏ accelerate · `cpu_adam` JIT cần ninja → tắt) · **Llama-3** (không có `unk_token` · DeRTa resize vocab 128256→128257 · `target_modules` chứa `w1/w2/w3` của Mixtral).

Chi tiết vá lỗi từng bài: **README trong chính folder method**.

---

## 6. Caveat bắt buộc ghi vào báo cáo

1. **Circuit Breakers có XSTest trong retain set** → số over-refusal của nó **thiên vị lạc quan**. CAT thì sạch (đo over-refusal bằng bộ 40 câu tự viết). DeepRefusal cũng train trên 500 sample XSTest → cùng vấn đề.
2. **CAT pin `transformers==4.41.3` — version này không tồn tại trên PyPI** (nhảy 4.41.2 → 4.42.0). Thêm nữa checkpoint `Llama3-8B-IT-CAT` được train bằng code **không có trong repo public** (`model_utils.py` không có nhánh Llama-3, chạy vào là `NotImplementedError`) → không tái lập được đúng quy trình sinh ra nó.
3. **DeRTa thiếu hẳn file data safety** (`safety_beaver_safe_and_unsafe_response.json` — dòng đầu tiên script data đọc). Đã viết `rebuild_safety_data.py` tái tạo 6000 cặp từ `PKU-Alignment/PKU-SafeRLHF`, nhưng **không bit-exact** → model tự train chỉ là "DeRTa-style".
4. **JBShield** học jailbreak concept theo **từng loại attack** từ calibration set, nhưng `harmbench.csv` là prompt thô **không bọc jailbreak template** → jailbreak concept gần như không kích hoạt, method chỉ còn chạy bằng toxic concept.
5. **DeRTa** nhắm thẳng **prefilling attack**, mà HarmBench thô không có prefilling → điểm mạnh nhất của nó **không hiện ra trong bảng**. Đừng kết luận "DeRTa yếu".
6. **SafeDecoding đang chạy Llama-2-7b** (dùng expert có sẵn) còn baseline chạy Llama-3-8B → bội số 8.3× lẫn cả chênh lệch model. Paper báo ATGR 1.03–1.07×.

---

## 7. Chi phí

**Mốc:** một lượt `response` đầy đủ = 300 HarmBench + 250 XSTest + 800 JustEval = **1350 prompt × 512 max_token**. Gọi thời gian `no_defense` local là **T**.

Cost đo ở **đơn vị thô, không quy ra tiền**: API đo bằng **token**, local đo bằng **cả token lẫn giây**, train tách riêng (một lần, khác đơn vị).

**Vì sao local ghi cả hai đơn vị:** *giây* bắt được overhead decoding (SafeDecoding 2 forward/token, JBShield SVD mỗi forward) nhưng lẫn chênh lệch model và thưởng nhầm method từ chối cụt lủn (ca CAT 0.5×); *token* thì tái lập được và cùng đơn vị với nhóm API nhưng **mù** với overhead đó. Đo hết, chọn sau. Ghi bằng:

```python
with meter.local("target") as rec:
    text, resp = client.chat(raw)
    rec.from_usage(resp)
```

**Ước lượng sinh response nếu không train gì:** INTRA 5 bài × 1×T = ~5T · IN 5 bài (SafeDecoding 1.05T + JBShield ~1.1T + DRO 1T + ROSE 2T + SafeInfer 2T) ≈ 7.15T → **~12T tổng**.

**Train lại** (số từ paper, A100-80GB): SafeDecoding 16 giây *(đã đo thật)* · CAPO ~19 phút · Circuit Breakers ~20 phút · CAT ~42 phút · DeepRefusal ~45 phút → **≈ 2 giờ 7 phút**. Trên MIG 40GB phải hạ batch → nhân **2–4×**.

→ **Nút thắt không phải train mà là inference.** Train lại cả nhóm intra rẻ hơn sinh response cho chúng cả chục lần.

---

## 8. Việc tiếp theo

1. Chạy `no_defense_local` full → mốc cho bảng thứ hai.
2. Thêm **Self-Reminder** (pre) + **WildGuard** (post) → đủ 5/5 cho nhóm API.
3. Chạy `judge --task justeval` cho 9 method API đã có response — **đang thiếu hẳn một metric**.
4. Chạy full `--task all` cho 5 bài local đã code.
5. Quét `JBS_FIRST_M` trên full 300 (hiện đặt 2, **chưa kiểm còn tác dụng phòng thủ không**).
6. Thêm **ROSE · DRO · SafeInfer** (in) + **DeepRefusal · Targeted LAT** (intra).

> Nhóm target-API và nhóm target-local **không cùng thang** — tách 2 bảng, mỗi bảng có `no_defense` của chính nó.
