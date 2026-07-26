# Phương pháp phòng thủ
**Mục tiêu:** 5 phương pháp mỗi nhóm — **5 pre · 5 post · 5 in · 5 intra**, cộng `no_defense` làm mốc.

---

## Cách đọc bảng

| Cột | Nghĩa |
|---|---|
| **Ưu tiên** | `✅` đã có kết quả · `🔧` đã code, chưa chạy full · `⏸️` **tạm gác, có lý do** · `1` `2` `3` thứ tự nên làm tiếp · `✕` đã loại |
| **GitHub** | repo chính thức. `✕` = không có repo → phải tự viết từ paper |
| **Code** | `✅` đủ chạy · `⚠️` thiếu phần nào đó (ghi rõ) · `✕` repo rỗng |
| **Data** | `✅` có sẵn trong repo · `⚠️` phải tự lo · `—` không cần (training-free) |
| **Flow** | các bước một request đi qua, và **tốn mấy call / mấy forward** |

**Đã đủ 5/5 cho pre và post.** Còn thiếu: in **2** (ROSE · SafeInfer) · intra **0**.

Cập nhật 27/07/2026: thêm 6 bài — Self-Reminder (pre) · SelfDefend + WildGuard (post) · DRO (in) · DeepRefusal + Targeted LAT (intra).

---

## 1. PRE — can thiệp ở INPUT

| Ưu tiên | Method | Venue | GitHub | Code | Data | Flow chạy cơ bản |
|:--:|---|---|---|:--:|:--:|---|
| ✅ | **SAGE** | Findings ACL 2025 | [NJUNLP/SAGE](https://github.com/NJUNLP/SAGE) | ✅ | — | Ghép 2 instruction (analysis + response) vào input → sinh **1 call** |
| ✅ | **IA** | COLING 2025 | [alphadl/SafeLLM_with_IntentionAnalysis](https://github.com/alphadl/SafeLLM_with_IntentionAnalysis) | ✅ | — | Lượt 1 phân tích ý định → ghép đoạn phân tích vào hội thoại → lượt 2 trả lời. **2 call** |
| ✅ | **G4D** | Findings NAACL 2025 | [IDEA-XL/G4D](https://github.com/IDEA-XL/G4D) | ✅ | — | intent detect → paraphrase (chỉ khi nghi) → safety analyze → ghép cả 3 thành "guidance" chèn vào prompt → target trả lời. **3–4 call**, retrieval TẮT |
| ✅ | **erase-and-check** | arXiv 2023 | [aounon/certified-llm-safety](https://github.com/aounon/certified-llm-safety) | ✅ | ⚠️ weight DistilBERT tải Dropbox ~256MB | Xoá 1..20 token cuối → **DistilBERT (66M, local)** chấm từng biến thể → chỉ cần **một** biến thể harmful là chặn thẳng, không gọi target. **≤21 forward + 1 call nếu pass** |
| ✅ | **Self-Reminder** | Nature MI 2023 | [yjw1029/Self-Reminder](https://github.com/yjw1029/Self-Reminder) | ✅ prompt-only | — | Bọc query bằng system prompt an toàn + câu nhắc ở cuối → **1 call**. Rẻ nhất, copy là chạy |
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

✅ **Nhóm pre đã đủ 5/5** (SAGE · IA · G4D · erase-and-check · Self-Reminder). Goal-Prioritization để dự phòng — nó cùng họ *prompt augmentation* với Self-Reminder nên làm cả hai thì bảng trùng hướng.

⚠️ **SelfDefend từng nằm ở bảng này, đã chuyển sang POST** — xem §5.1.

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
| ✅ | **SelfDefend** | USENIX Sec 2025 | [selfdefend/Code](https://github.com/selfdefend/Code) | ✅ + **checkpoint LoRA sẵn** | ✅ trong repo (AlpacaEval, JailbreakHub, JailbreakBench, MultiJail, Anthropic red-team) | Query đi **đồng thời** vào target (sinh bình thường nhưng **cache lại**) và một **shadow LLM** (đọc *prompt*, bọc bằng P_direct hoặc P_intent) → shadow trả "No" thì thả cache ra, ngược lại vứt đi và trả template từ chối. Chạy song song nên latency thêm rất ít. Bản prompt-only = **2 call thuần API** |
| ⏸️ | **WildGuard** | NeurIPS 2024 | [allenai/wildguard](https://github.com/allenai/wildguard) | ✅ ⚠️ **KHÔNG cài bằng pip** — package pin `vllm` là hard dep, đã thử và nó nâng torch lên cu130 làm hỏng CUDA của venv chung. Nạp thẳng bằng transformers, prompt+parser verbatim | ✅ | Target sinh → 1 model đọc (prompt, response) → ra **3 nhãn**: prompt-harm · response-harm · **refusal** → đo được **cả 2 trục cùng lúc**. LOCAL 7B + target Groq → tốn cả token lẫn giây GPU |
| 3 | **Llama Guard 3** | Meta 2024 | ⚠️ `PurpleLlama` không có code, chỉ weight | 📦 dùng ckpt | ✅ taxonomy có sẵn | Classifier chấm (prompt, response) theo taxonomy 13 mục → gắn cờ. ⚠️ **Groq đã bỏ `llama-guard-3-8b`** → dùng `openai/gpt-oss-safeguard-20b` hoặc host local |
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
| ⏸️ | **DRO** | ICML 2024 | [chujiezheng/LLM-Safeguard](https://github.com/chujiezheng/LLM-Safeguard) | ✅ gọn nhất nhóm | ✅ `data/` + `data_harmless/` | Offline tối ưu **soft prompt** sao cho biểu diễn prompt harmful bị đẩy *theo* hướng từ chối, harmless bị đẩy ngược. Deploy = prepend soft prompt → **~1×T**, không overhead decoding. ⚠️ script sẵn **chỉ cho Mistral-v1** |
| **3** | **SafeInfer** | AAAI 2025 | [NeuralSentinel/SafeInfer](https://github.com/NeuralSentinel/SafeInfer) | ⚠️ README không nêu model hỗ trợ, phải đọc code | ⚠️ phải tự chuẩn bị demonstration examples | 2 pha lúc decode: (1) dịch hidden state theo hướng an toàn trích từ demonstration, (2) phối logit với một con `M_unsafe`. **~2×T + cần 2 model cùng lúc trong VRAM** (~32GB, chật trên MIG 40GB → phải quantize 4-bit con phụ) |
| 4 | **GeDi** | Findings EMNLP 2021 | [salesforce/GeDi](https://github.com/salesforce/GeDi) | ✅ | ⚠️ | Mỗi bước decode, một discriminator LM reweight xác suất token kế tiếp qua Bayes. Cần **train discriminator** |
| 4 | **TaskTracker** | IEEE SaTML 2025 | [microsoft/TaskTracker](https://github.com/microsoft/TaskTracker) | ✅ | ⚠️ **phải xin qua form** | Trích activation delta → đưa vào probe đã train → phát hiện task drift. Detector, **không sửa output** |
| 5 | **DeAL** | ACL 2025 | ✕ **không có repo** (ACL Anthology không có mục Software, arXiv không nêu URL, không có trên `amazon-science`/`amzn`) | tự viết 100% | ✅ HH-RLHF public | Coi sinh văn bản là **A\* search**: giữ beam top-k (k=5–10), mỗi ứng viên **lookahead 32 token** greedy để chấm heuristic sớm, heuristic = reward model **OPT-125M**. ⚠️ **~k×l = 160 forward mỗi bước decode** → **tốn giờ GPU nhất toàn survey**; chính tác giả thừa nhận "generality makes decoding slower" |
| ✕ | Self-CD | ACL 2024 | — | — | — | **Loại:** chỉ **giảm over-refusal**, không chống jailbreak |
| ✕ | AdapT (Hot or Cold?) | AAAI 2024 | ✕ không tìm thấy | ✕ | ✕ | **Loại — lệch chủ đề:** đúng là decoding-time (temperature cao cho token khó, thấp cho token dễ) nhưng chủ đề là **sinh CODE**, đo bằng pass@k. Không có yếu tố an toàn nào, không có ASR/over-refusal để điền |
| ✕ | Hybrid Uncertainty Quantification | ACL 2023 | [AIRI-Institute/hybrid_uncertainty_estimation](https://github.com/AIRI-Institute/hybrid_uncertainty_estimation) | ✅ nhưng là pipeline train classifier | ✅ | **Loại — sai đối tượng:** selective classification cho **encoder classifier**, không đụng decoding của LLM sinh. "Từ chối" ở đây là *abstention vì không chắc*, khác hẳn *refusal vì có hại*. Không có threat model tấn công. *(Vẫn hữu ích làm reference nếu sau này cần hiệu chỉnh ngưỡng cho detector — Perplexity, FJD, JBShield-D)* |
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
| 🔧 | **DeepRefusal** | Findings EMNLP 2025 | [YuanBoXie/DeepRefusal](https://github.com/YuanBoXie/DeepRefusal) | ⚠️ **không có lệnh train**, phải ghép code từ 3 repo | ✅ paper ghi đủ, đều public (CB 2k + UltraChat 4k + XSTest/Or-bench 500) | Phá rồi dựng lại **refusal direction** ở nhiều tầng, có prefill augmentation → từ chối không còn nông ở token đầu. Ckpt ✅ `skysys00/Meta-Llama-3-8B-Instruct-DeepRefusal`. Train **~45 phút** |
| 🔧 | **Targeted LAT** | arXiv 2407.15549 | [aengusl/latent-adversarial-training](https://github.com/aengusl/latent-adversarial-training) | ⚠️ **toàn notebook, không có CLI** | ⚠️ README không nêu rõ | Adversarial training với nhiễu đặt ở **activation giữa các layer** (latent) thay vì embedding như CAT. Ckpt ✅ org HF `LLM-LAT`. Train ~36× rẻ hơn R2D2 (chưa có số tuyệt đối) |
| 3 | **Safe RLHF** | ICLR 2024 | [PKU-Alignment/safe-rlhf](https://github.com/PKU-Alignment/safe-rlhf) | ✅ | ✅ | Tách reward (hữu ích) + cost (hại), PPO ràng buộc Lagrangian. Ckpt ✅ `PKU-Alignment/beaver-7b-v1.0`. ⚠️ **full RLHF không vừa 40GB** → chỉ dùng được ckpt |
| 4 | **SecAlign** | ACM CCS 2025 | [facebookresearch/SecAlign](https://github.com/facebookresearch/SecAlign) | ✅ | ✅ | DPO trên cặp preference → ra model đã align. ⚠️ threat model = **prompt injection** chứ không hẳn jailbreak |
| 5 | **ReFAT** | ⚠️ arXiv 2409.20089, **chưa xác nhận venue** (chỉ thấy OpenReview forum) | ✕ không có repo public | tự viết (thuật toán **ngắn**) | ✅ public | Quan sát: **mọi** attack đều đi qua một cơ chế chung — ablate **refusal feature** trong residual stream. Vậy mô phỏng thẳng hiệu ứng cuối: xoá refusal feature với xác suất p qua các layer lúc forward, rồi SFT để model **vẫn từ chối**. Rẻ hơn R2D2 ~1700×, hơn CAT ~10×. ⚠️ **giá trị biên thấp** — cùng hướng refusal-direction với DeepRefusal + Circuit Breakers, làm cả 3 là trùng |
| ✕ | R2D2 | ICML 2024 | trong [HarmBench](https://github.com/centerforaisafety/HarmBench) | ✅ | ✅ | **Loại:** adversarial training vs GCG, **rất đắt** (chạy GCG mỗi step), base Zephyr |
| ✕ | TAR | ICLR 2025 | [rishub-tamirisa/tamper-resistance](https://github.com/rishub-tamirisa/tamper-resistance) | ✅ | ✅ | **Loại — threat model khác:** = **weight tampering** (safety phải sống sót qua fine-tune tấn công), không phải prompt jailbreak. Rất đắt |
| ✕ | Booster | ICLR 2025 Oral | [git-disl/Booster](https://github.com/git-disl/Booster) | ✅ | ✅ BeaverTails | **Loại — threat model khác, cùng nhóm TAR:** chống **harmful fine-tuning attack** (kẻ tấn công cầm model đã align rồi fine-tune lại bằng data độc). 3 gradient/step trên 7B, **không có ckpt**, hạ tầng viết cho Slurm. HarmBench ASR trên prompt **không đo được** thứ nó bảo vệ |
| ✕ | RMU / WMDP | ICML 2024 | ✅ | ✅ | ✅ | **Loại:** đo bằng **WMDP multiple-choice**, lệch metric ASR/over-refusal |

**5 bài chốt cho INTRA** (CAT · Circuit Breakers · DeRTa · DeepRefusal · Targeted LAT) — **cả 5 đều có checkpoint Llama-3-8B sẵn** → chạy được toàn bộ mà không cần train lần nào.

---

## 4b. ⏸️ Hai bài đang tạm gác — lý do và việc cần làm để chạy tiếp

### WildGuard (post) — chặn bởi HF token

`allenai/wildguard` là **gated repo**, mức `auto` (chỉ cần một tài khoản HF bấm đồng ý, **không** phải chờ duyệt tay). Server không có `HF_TOKEN` nên tải về báo:

```
401 Client Error — Access to model allenai/wildguard is restricted.
```

Đã tìm mirror mở: **không có**. Các repo trùng tên trên HF đều là model khác fine-tune trên *dữ liệu* WildGuardMix, không phải model của AI2.

**Để chạy tiếp:** đặt `HF_TOKEN` trên server rồi chạy `methods/post/WildGuard/method.py` như bình thường. Code đã sẵn sàng (không dùng package `wildguard` vì nó pin vllm — xem README của method).

**Không gấp:** nhóm post **đã đủ 5/5** (Self_Defense · Self_Refine · Backtranslation · AutoDefense · SelfDefend). WildGuard là bài thứ 6.

### DRO (in) — pipeline 5 stage, thiếu một model đánh giá

Ban đầu tưởng 3 stage, chạy thật mới lộ ra **5**:

| # | Stage | Trạng thái |
|:--:|---|---|
| 1 | `forward.py` — hidden state tập **harmful** | ✅ 17.6s |
| 2 | `forward.py --use_harmless` — hidden state tập **harmless** | ✅ 19.5s |
| 3 | `generate.py` — sinh response cho 100+100 query | ❌ chưa chạy |
| 4 | `evaluate.py` — gán nhãn refusal/compliance cho từng response | ❌ **chặn** |
| 5 | `estimate.py` → `train.py` — ước lượng hướng từ chối rồi tối ưu soft prompt | ⏸️ chờ stage 4 |

**Chỗ chặn ở stage 4:** `evaluate.py:88` cho nhánh *harmful* dùng `get_matching_scores × get_eval_scores(evaluator)`, mà evaluator upstream chỉ định là **`meta-llama/LlamaGuard-7b`** — lại một gated repo nữa. *(Nhánh harmless thì chỉ string-matching, không cần model.)*

**Ba hướng đi tiếp:**
1. **Có `HF_TOKEN`** → chạy đúng như paper, sạch nhất.
2. **Thay evaluator** bằng `cais/HarmBench-Llama-2-13b-cls` (đã có sẵn trên server, không gated) → phải sửa `get_eval_scores` cho khớp prompt format, và **khai báo là sai khác**.
3. **Chỉ dùng string-matching** cho cả hai nhánh → nhãn kém chính xác hơn → hướng từ chối ước lượng kém → soft prompt yếu đi. Rẻ nhất nhưng lệch xa paper nhất.

**Bốn thứ đã vá được để tới bước này** (đều nằm trong `train_smoke.py`, `repo/` giữ nguyên): whitelist model không có Llama-3 · không có chat template Llama-3 · bật cứng FlashAttention2 (server không có) · `utils.py` gọi `pynvml` mà **MIG chặn NVML**.

⚠️ Nhắc lại: **upstream không phát hành soft prompt đã train**, nên không train xong thì không có gì để chạy inference. Khác hẳn 3 bài intra (tải checkpoint là chạy).

---

## 5. Bài xếp nhầm nhóm / lệch chủ đề — đọc trước khi chạy

### 5.1 Đã đổi nhóm so với lúc đầu

| Bài | Xếp lúc đầu | Xếp đúng | Vì sao |
|---|---|---|---|
| **SelfDefend** | in | **post** | Không phải in vì shadow LLM không đọc logits, không đụng nội tại target. Xếp **post** chứ không phải pre vì **lúc ra quyết định thì target đã sinh xong response rồi** (chỉ đang nằm trong cache) — xem phân tích ở dưới |
| **ICAG** | in | **pre** | Sản phẩm cuối là một chuỗi **safety system prompt** prepend vào input, không can thiệp decoding |
| **FJD** | post | **pre** | Chạy 1 token rồi xét confidence để **chặn trước khi sinh full** — quyết định nằm ở cổng input |
| **RC-RAG** | in | **post** | Sinh answer xong mới tạo counterfactual prompt để thách thức → thao tác trên **output** |

> ⚠️ **ĐỪNG NHẦM `SelfDefend` với `LLM Self Defense`** — hai bài khác hẳn, chỉ trùng tên:
>
> | | **LLM Self Defense** (đã chạy) | **SelfDefend** (chưa làm) |
> |---|---|---|
> | Tác giả · venue | Phute et al., arXiv 2023 | Wang et al., USENIX Sec 2025 |
> | Repo | `poloclub/llm-self-defense` | `selfdefend/Code` |
> | **Soi cái gì** | **RESPONSE** — sinh xong mới hỏi "có hại không?" | **PROMPT** — shadow LLM đọc câu hỏi |
> | Nhóm | POST | POST |
> | Trong repo | `methods/post/Self_Defense/` | chưa có |
>
> Cùng nhóm post nhưng **khác tín hiệu**: một bên phán dựa trên câu trả lời, một bên phán dựa trên câu hỏi. Đây là chỗ trục phụ **detector-đọc-input vs detector-đọc-output** có ích.

#### Vì sao SelfDefend là post chứ không phải pre

Từng có lúc xếp nó vào pre với lý lẽ: shadow LLM chỉ đọc *prompt*, còn chạy song song chỉ là mẹo giảm latency — cài tuần tự (kiểm prompt trước, an toàn mới sinh) thì **output giống hệt**, nên "về bản chất" nó là bộ lọc input.

Lý lẽ đó **sai ở chỗ nó phân loại theo cách *có thể* cài, chứ không phải cách paper *thực sự* chạy.** Trong thiết kế của paper, target **sinh xong response** rồi mới có tín hiệu checkpoint; response nằm trong cache chờ được thả hoặc bị vứt. Hai hệ quả quan sát được:

- **Chi phí:** đã tiêu trọn một lượt sinh của target **kể cả khi bị chặn**. Bộ lọc pre thật (erase-and-check) thì chặn xong là **khỏi gọi target**, tiết kiệm thật.
- **Cách nó hỏng:** over-refusal xảy ra sau khi đã có sẵn một câu trả lời tốt rồi vứt đi — đúng kiểu hỏng của nhóm post.

Vì survey này **bám sát paper**, không được sửa thiết kế cho gọn hơn (xem ghi chú cuối §5), nên phân loại phải theo cách chạy thật → **post**.

**DRO** thì vẫn đang treo: soft prompt = embedding đã train rồi prepend vào input → xét chặt là **pre** (cùng họ RPO). Giữ ở IN thì phải định nghĩa "in = can thiệp tầng biểu diễn/decoding" ngay đầu survey. Chuyển sang PRE thì **không có bài thay thế** vì 3 ứng viên IN còn lại đều repo rỗng.

> **Tiêu chí phân loại pre/post đang dùng:** *lúc defense ra quyết định, đã tồn tại một response hoàn chỉnh chưa?* Chưa → **pre**; rồi → **post**. Tiêu chí này xử được cả 4 ca khó: FJD chỉ sinh 1 token để đọc confidence, chưa có response → pre. SelfDefend có response đầy đủ trong cache → post.
>
> **Nguyên tắc bao trùm:** phân loại theo **cách paper thực sự chạy**, không theo cách "về lý thuyết cài kiểu khác cũng ra kết quả đó".

### 5.2 Lệch chủ đề — KHÔNG chạy trên HarmBench/XSTest

Chạy mấy bài này trên pipeline hiện tại là **sai phạm trù**, không phải "chạy ra số xấu":

| Bài | Nhóm cơ chế | Vì sao không đo được |
|---|---|---|
| **Prompt-Tuning** (memorization) | pre | Chủ đề **privacy**. Đo bằng reconstruction rate + perplexity. Không có khái niệm ASR / từ chối |
| **SelfCheckGPT** | post | **Hallucination detection**. Đo tính nhất quán giữa N mẫu, không liên quan nội dung có hại |
| **AdapT** (Hot or Cold?) | in | Chủ đề **sinh code**, đo pass@k |
| **Hybrid UQ** | in | Đối tượng là **encoder classifier**, không phải LLM sinh. "Từ chối" = *abstention vì không chắc*, khác *refusal vì có hại* |
| **Self-CD** | in | Chỉ **giảm over-refusal**, không có cơ chế chống jailbreak → chỉ cải thiện được 1 trong 2 trục |
| **RMU / WMDP** | intra | Đo bằng **multiple-choice**, không phải sinh văn bản |
| **TAR** · **Booster** | intra | Threat model = **kẻ tấn công cầm được trọng số rồi fine-tune lại**, không phải prompt jailbreak. HarmBench đo prompt nên không chạm tới thứ chúng bảo vệ |
| **SecAlign** | intra | Threat model = **prompt injection**. Gần hơn 2 bài trên nhưng vẫn không hẳn jailbreak |

Nếu muốn giữ TAR + Booster thì mở **mục riêng "defense chống harmful fine-tuning"** với metric riêng (harmful score sau khi bị fine-tune ở poison ratio p), đừng nhét vào bảng ASR chung.

### 5.3 Không phải defense đứng riêng

- **Refusal-direction orthogonalization** (Arditi, NeurIPS 2024) — chủ yếu là **tấn công/ablation** (gỡ refusal đi). Nó là **nền cơ học** cho cả họ steering (Circuit Breakers, DeepRefusal, ReFAT, InferAligner) → nên trích trong phần related work, không phải một dòng trong bảng.
- **Token Highlighter · RA-LLM** — hay bị xếp vào post, nhưng thực chất là **input-side**.

### 5.4 Đúng INTRA nhưng lệch chủ đề (14 bài, chỉ làm related work)

Đã rà nhưng không lấy — ghi lại để khỏi tìm lại:

| Hướng | Bài | Vì sao loại |
|---|---|---|
| **Detox / RLHF chung** | Quark · DAPT Detox · Fine-Grained RLHF · DRLC · FIGA · InstructGPT · Okapi · Self-Criticism | Detox hoặc alignment chất lượng chung trên GPT-2 / T5 / Megatron. Không có khái niệm jailbreak, base sai stack |
| **RAG noise robustness** | ATM · RAAT | Threat model = tài liệu nhiễu/bịa trong RAG, đo F1/EM |
| **Classifier robustness** | FLAT · Veiled Toxicity · Impact of Adv. Training | Adversarial training cho **classifier** (LSTM/CNN/BERT), tấn công thay từ đồng nghĩa. Không phải LLM sinh |
| **Backdoor / poisoning** | Moderate-fitting | Threat model = data poisoning khi fine-tune PLM |

### 5.5 Nguyên tắc bám sát paper — chi phối cả việc phân loại lẫn việc cài

Thứ tự ưu tiên khi triển khai bất kỳ bài nào:

1. **Repo chạy được** → chạy code của họ, chỉ thêm hàm phụ trợ tối thiểu để cắm vào `core/runner.py`. **Không viết lại.**
2. **Không có repo / repo rỗng** → tự viết nhưng cố làm giống nhất, và lấy **verbatim** những thứ quyết định con số (prompt, hằng số, thuật toán).

Đây là lý do cột **GitHub / Code / Data** trong mọi bảng ở trên phải kiểm trước khi chọn bài — repo chỉ có kết quả mà không có code thì bài đó đắt hơn nhiều so với vẻ ngoài.

**Hệ quả 1 — không "tối ưu" thiết kế của paper cho gọn hơn.** Ví dụ SelfDefend chạy target song song với shadow, tốn trọn một lượt sinh kể cả khi bị chặn. Cài lại thành kiểm-prompt-trước-rồi-mới-sinh thì rẻ hơn và **output giống hệt**, nhưng đó không còn là SelfDefend nữa → không làm.

**Hệ quả 2 — phân loại theo cách paper *thực sự* chạy**, không theo cách "về lý thuyết cài kiểu khác cũng ra kết quả đó". Chính hệ quả này quyết định SelfDefend là post.

**Mọi sai khác bắt buộc phải khai báo rõ trong báo cáo** — đổi model phụ trợ (G4D thay GPT-4o-mini bằng `llama-3.1-8b`, Backtranslation thay Vicuna-13B), hạ batch vì VRAM, dùng venv version khác, quantize 4-bit.

---

## 6. Chi tiết từng bài (bài trong kế hoạch)

Mỗi bài: cơ chế · trạng thái · thứ phải để ý. Bài đã loại xem §5.

### 6.1 PRE

**SAGE** — Findings ACL 2025 · ✅ ASR 0.7% / over-refusal 34.8%
Training-free, rẻ nhất bảng: ghép 2 instruction (một bảo model tự phân tích, một bảo trả lời) vào cùng input, sinh 1 lần. ⚠️ **Đánh đổi lộ liễu nhất trong 9 bài** — chặn gần như tuyệt đối nhưng từ chối oan gấp hơn 4 lần mốc. Đừng trích ASR mà bỏ cột kia.

**IA** (Intention Analysis) — COLING 2025 · ✅ ASR 2.0% / over-refusal 12.4%
Hai lượt: lượt 1 bắt model phân tích ý định thật của câu hỏi, đoạn phân tích đó **ghép vào hội thoại** làm ngữ cảnh, lượt 2 mới trả lời. Không chặn request nào. **Cân bằng tốt nhất hiện có** — cả hai cột đều đẹp với chi phí 2 call.

**G4D** — Findings NAACL 2025 · ✅ ASR 7.0% / over-refusal 10.8%
Ba agent riêng: phát hiện ý định → viết lại câu hỏi (chỉ khi bị nghi) → phân tích an toàn; ba kết quả ghép thành đoạn "guidance" chèn vào prompt cuối. Retrieval **tắt** đúng theo `main.py` upstream. Paper dùng GPT-4o-mini cho cả 3 agent, mình thay bằng `llama-3.1-8b` → phải khai báo trong báo cáo.

**erase-and-check** — arXiv 2023 · ✅ ASR 14.7% / over-refusal 8.4%
Bài duy nhất **chặn thẳng không gọi target**: xoá 1..20 token cuối tạo ra ≤21 biến thể, DistilBERT (66M, chạy local) chấm từng cái, chỉ cần một biến thể bị gắn harmful là trả câu từ chối cố định. **Over-refusal thấp nhất nhóm** (8.4%, gần mốc 8.0%) nhưng ASR cao — bảo thủ vừa phải. Cũng là bài duy nhất tốn **cả token API lẫn giây GPU**.

**Self-Reminder** — Nature MI 2023 · ưu tiên **1**
Bọc query bằng system prompt an toàn + thêm một câu nhắc ở cuối. Prompt-only, 1 call, copy prompt là chạy. Paper: ASR 67% → 19%. ⚠️ coi chừng over-refusal — cùng họ "nhắc nhở mạnh" với SAGE.

### 6.2 POST

**SelfDefend** — USENIX Sec 2025 · ưu tiên **1**
Shadow LLM chạy **song song** target (mượn ý tưởng shadow stack): target vẫn sinh nhưng **cache lại**, shadow đọc *prompt* bằng P_direct hoặc P_intent rồi ra tín hiệu; "No" thì thả cache, ngược lại vứt và trả từ chối. Bản prompt-only = **2 call thuần API**, cấu trúc gần y hệt LLM Self Defense đã làm → dựng trong một buổi. Bản tuned có **checkpoint LoRA sẵn** (`llama-2-7b-lora-direct` / `-intent`), khỏi train. Xếp post vì lúc quyết định thì response đã sinh xong — xem §5.1. **Giá trị riêng:** là detector post duy nhất **đọc input**, 4 bài post hiện có đều đọc output.

**LLM Self Defense** — arXiv 2023 · ✅ ASR 9.7% / over-refusal 35.6%
Target sinh xong, 1 call hỏi "response này có hại không", có thì vứt và thay bằng từ chối cố định. ⚠️ **Over-refusal cao nhất bảng (35.6%)** — con judge quá nhạy, gạt nhầm nhiều câu vô hại.

**Self-Refine** — NeurIPS 2023 · ✅ ASR 6.3% / over-refusal 12.0%
Target sinh → call feedback phê bình → call refine viết lại, lặp k vòng, bản mới thay bản cũ. **Vốn không phải method an toàn** (paper là cải thiện chất lượng chung) nhưng ra số tốt cả hai cột. Đắt: 1 + 2k call.

**Backtranslation** — Findings ACL 2024 · ✅ ASR 17.0% / over-refusal 9.6%
Suy ngược từ response ra **prompt gốc** (lộ intent thật, bỏ hết lớp nhiễu jailbreak) → hỏi lại target bằng prompt suy ngược → target từ chối prompt đó thì vứt response ban đầu. Paper dùng Vicuna-13B để backtranslate, mình thay bằng target.

**AutoDefense** — arXiv 2024 · ✅ ASR 18.7% / over-refusal 9.2%
Nhiều agent cùng đọc response rồi bỏ phiếu. **Đắt nhất nhóm (4 call, 2821 token vào)** mà ASR lại cao nhất trong 4 bài post đã chạy → hiện là bài **kém hiệu quả trên chi phí** nhất bảng.

**WildGuard** — NeurIPS 2024 · ưu tiên **2**
Một model đọc (prompt, response) ra **3 nhãn cùng lúc**: prompt-harm · response-harm · **refusal**. Đây là điểm đặc biệt — nó tự cho luôn cả hai trục mình cần đo, không cần judge riêng. LOCAL 7B, `pip install wildguard` là có.

### 6.3 IN

**SafeDecoding** — ACL 2024 · 🔧 đã chạy smoke, 5.490 s/req
Mỗi bước decode lấy giao top-k của base và top-k của expert (chính target + LoRA train trên 72 cặp), rồi `p = p_base + α(p_expert − p_base)` với α=3, **chỉ áp 2 token đầu** — vì "safety disclaimer" nằm ngay đầu response, quyết xong là xong. Đã xác nhận `disable_adapter()` thay được mixed-adapter-batch → **khỏi cần peft fork** của upstream. Train expert Llama-3 mất **16 giây**, 0 lỗi. ⚠️ đang chạy Llama-2-7b nên bội số 8.3× lẫn cả chênh lệch model; paper báo ATGR chỉ 1.03–1.07×.

**JBShield** — USENIX Sec 2025 · 🔧 đã chạy smoke, 0.839 s/req
Dựa trên Linear Representation Hypothesis: tách **toxic concept** (có ở cả harmful lẫn jailbreak prompt) và **jailbreak concept** (chỉ có ở jailbreak prompt, chính nó lật model từ từ chối sang tuân theo). Mitigation = cộng anchor vector của toxic subspace, trừ của jailbreak subspace. ✅ **Tái hiện đúng paper phần detection: 0.958 vs 0.95**. Phải vá 2 bug upstream mới dùng được (hook viết cho transformers v4; `detection()` không thực gate). ⚠️ hook chạy **1 SVD mỗi forward** nên đắt hơn tôi ước ban đầu.

**ROSE** — Findings ACL 2024 · ưu tiên **1**
Contrastive decoding: `logit = logit(prompt thường) − w · logit(reverse prompt độc)`. Reverse prompt là prompt **cố tình dụ model trả lời độc** — trừ nó đi thì xu hướng độc bị triệt tiêu. Training-free hoàn toàn, không model phụ, không calibration → **dựng xong là có ngay một điểm dữ liệu**, và đường ống local chạy thông thì 4 bài IN sau chỉ là thay phần can thiệp.

⚠️ **Vì sao gắn cảnh báo ở cột Code** — repo *có* code thật, nhưng viết quanh **lmdeploy** (engine suy luận C++/CUDA của nhóm InternLM, dùng TurboMind) chứ không phải HF `generate()`. Bốn hệ quả:

1. **Không hook được:** contrastive decoding cần logits của **2 lần forward** mỗi bước rồi trừ nhau. HF đưa thẳng logits; lmdeploy giấu vòng decode trong C++.
2. **Phá cột cost — nặng nhất:** `no_defense_local` và 5 bài local kia đều chạy HF. ROSE chạy TurboMind (nhanh hơn hẳn) thì **giây/request không so được với baseline** — thậm chí tốn 2 forward/token mà vẫn ra nhanh hơn no_defense, đọc bảng sẽ kết luận ngược. Vỡ nguyên tắc "cùng GPU, cùng điều kiện" của `cost_meter.py`.
3. Phải **convert trọng số** sang định dạng TurboMind; repo demo trên **Baichuan2-7B**, không phải Llama-3.
4. Thêm dependency nặng phải compile CUDA trên server MIG.

**→ Nên tự viết lại bằng HF (~50–80 dòng).** Việc này **không vi phạm §5.5** vì `CLAUDE.md` §8 cho phép đổi *engine* miễn khai báo; thứ **phải lấy verbatim** là **nội dung reverse prompt** (cả cơ chế nằm ở đó) và **hệ số `w`**. Cố chạy nguyên lmdeploy mới là bám paper sai chỗ: giống code nhưng hỏng cột cost của cả bảng.

**DRO** — ICML 2024 · ưu tiên **2**
Quan sát: model đã "biết" phân biệt harmful/harmless trong không gian biểu diễn, chỉ là ranh giới chưa đủ tách. DRO tối ưu một **soft prompt liên tục** đẩy biểu diễn harmful *theo* hướng từ chối và harmless *ngược lại*. Deploy chỉ là prepend → **không overhead decoding**. Repo gọn nhất nhóm, data có sẵn. ⚠️ script sẵn **chỉ cho Mistral-v1**, model khác phải tự thêm chat template. Xem thêm vụ borderline in/pre ở §5.1.

**SafeInfer** — AAAI 2025 · ưu tiên **3**
Hai pha lúc decode: (1) dịch hidden state theo hướng an toàn trích offline từ demonstration examples, (2) phối logit với một con `M_unsafe`. **Nút thắt thật sự là VRAM** — hai model 8B bf16 ≈ 32GB, cộng KV cache thì rất chật trên MIG 40GB → phải quantize 4-bit con phụ (lệch paper, phải khai báo). Cộng thêm việc tự dựng bộ demonstration → **tốn công nhất nhóm IN**.

### 6.4 INTRA

**CAT / CAPO** — NeurIPS 2024 Spotlight · 🔧 đã chạy smoke, 0.317 s/req
Adversarial training với nhiễu đặt ở **embedding đầu vào** (liên tục) thay vì đi tìm chuỗi token rời rạc như GCG → rẻ hơn hẳn. ⚠️ **Trả về đúng một chuỗi cố định** `"Sorry, I can't do that."` vì safe answer lúc train là câu đó → sẽ ăn điểm thấp ở JustEval (depth/engagement), không ảnh hưởng ASR. Con số "≥1904 GPU hours" trong paper là **tổng mọi thí nghiệm** kể cả chạy GCG/AutoDAN để đánh giá; một lần train chỉ 42 phút. ✅ Không nhiễm XSTest.

**Circuit Breakers** — NeurIPS 2024 · 🔧 đã chạy smoke, 6.067 s/req
Thay vì dạy model từ chối, nó **bẻ gãy chính biểu diễn nội bộ** dẫn tới output có hại → model "chập mạch" giữa chừng. ⚠️ **1/3 response là chuỗi vô nghĩa — đó CHÍNH LÀ cơ chế, không phải bug.** Paper DeepRefusal chỉ ra nó làm tụt GSM8k mạnh (42.84 vs base 75.44) đúng vì lý do này → điểm hay để bàn: circuit breaker đánh đổi utility nhiều hơn các bài khác. ⚠️ **retain set có XSTest** → over-refusal thiên vị lạc quan.

**DeRTa** — ACL 2025 · 🔧 đã chạy smoke, 2.567 s/req
Alignment thường chỉ dạy từ chối **ở token đầu**, nên attacker ép sẵn prefix tuân thủ là model viết tiếp. DeRTa prepend sẵn harmful prefix rồi dạy model **bẻ lái sang từ chối giữa chừng** (MLE) + RTO reinforce điểm chuyển ở **mọi vị trí**. Response là câu từ chối **có giải thích lý do**, tự nhiên hơn hẳn CAT → dự đoán ăn điểm JustEval cao hơn. ⚠️ điểm mạnh nhất (chống prefilling) **không hiện ra** vì HarmBench của mình là prompt thô.

**DeepRefusal** — Findings EMNLP 2025 · ưu tiên **1**
Phá rồi dựng lại **refusal direction** ở nhiều tầng, kèm prefill augmentation → từ chối không còn nông. ASR đẹp nhất bảng (CodeAttack 87.1% → 0.2%). ✅ **Table 1 của họ có sẵn số cho LAT, CAT, CircuitBreaker trên đúng Llama3-8B** → dùng để **kiểm tra chéo pipeline của mình**. ⚠️ over-refusal 28.5% ở p=0.5 (chính họ thừa nhận yếu) và ckpt đã train trên 500 sample XSTest → cùng vấn đề nhiễm như Circuit Breakers. ⚠️ README **không có lệnh train**, phải ghép code từ 3 repo.

**Targeted LAT** — arXiv 2407.15549 · ưu tiên **2**
Adversarial training với nhiễu đặt ở **activation giữa các layer** (latent) — cùng tinh thần "né search rời rạc cho rẻ" với CAT, khác chỗ CAT perturb ở *embedding đầu vào* còn LAT perturb ở *giữa mạng*. Có nó thì bảng INTRA phủ đủ 3 biến thể adversarial training: **rời rạc (R2D2, không chạy) → embedding (CAT) → latent (LAT)**. ⚠️ repo **toàn notebook, không có CLI**.

---

## 7. Kết quả đã chạy

### 7.1 Nhóm API — target `llama-3.1-8b-instant` (Groq)

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

### 7.2 Nhóm LOCAL — target `Meta-Llama-3-8B-Instruct` trên GPU

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

### 7.3 Train lại trên Llama-3 (smoke-size) — 4/4 PASS

| Method | Script | Thời gian | Adapter tự train nạp lại | Số thứ phải vá |
|---|---|---:|---|:--:|
| SafeDecoding expert | `train_expert.py` | **16.2 s** | ✅ 1.932 s/req | **0** |
| Circuit Breakers | `train_smoke.py` | 8.1 s / 5 step | ✅ 1.121 s/req | 5 |
| CAT | `train_smoke.py` | 2.26 s / 5 step | ✅ 1.022 s/req | 6 |
| DeRTa | `train_smoke.py` | 3.41 s / 5 step | ✅ 1.006 s/req | 6 |

Bốn nhóm lỗi: **transformers v4→v5** (`deepspeed` alias bị bỏ · `fsdp=None` · `tokenizer=`→`processing_class` · `num_items_in_batch` · `is_torch_tpu_available` · `--eval_strategy`) · **trl 1.9 xoá `DataCollatorForCompletionOnlyLM`** (CAT *kế thừa* nó → phải venv riêng `.venv_cat`) · **MIG** (NCCL chết → bỏ accelerate · `cpu_adam` JIT cần ninja → tắt) · **Llama-3** (không có `unk_token` · DeRTa resize vocab 128256→128257 · `target_modules` chứa `w1/w2/w3` của Mixtral).

Chi tiết vá lỗi từng bài: **README trong chính folder method**.

---

## 8. Caveat bắt buộc ghi vào báo cáo

1. **Circuit Breakers có XSTest trong retain set** → số over-refusal của nó **thiên vị lạc quan**. CAT thì sạch (đo over-refusal bằng bộ 40 câu tự viết). DeepRefusal cũng train trên 500 sample XSTest → cùng vấn đề.
2. **CAT pin `transformers==4.41.3` — version này không tồn tại trên PyPI** (nhảy 4.41.2 → 4.42.0). Thêm nữa checkpoint `Llama3-8B-IT-CAT` được train bằng code **không có trong repo public** (`model_utils.py` không có nhánh Llama-3, chạy vào là `NotImplementedError`) → không tái lập được đúng quy trình sinh ra nó.
3. **DeRTa thiếu hẳn file data safety** (`safety_beaver_safe_and_unsafe_response.json` — dòng đầu tiên script data đọc). Đã viết `rebuild_safety_data.py` tái tạo 6000 cặp từ `PKU-Alignment/PKU-SafeRLHF`, nhưng **không bit-exact** → model tự train chỉ là "DeRTa-style".
4. **JBShield** học jailbreak concept theo **từng loại attack** từ calibration set, nhưng `harmbench.csv` là prompt thô **không bọc jailbreak template** → jailbreak concept gần như không kích hoạt, method chỉ còn chạy bằng toxic concept.
5. **DeRTa** nhắm thẳng **prefilling attack**, mà HarmBench thô không có prefilling → điểm mạnh nhất của nó **không hiện ra trong bảng**. Đừng kết luận "DeRTa yếu".
6. **SafeDecoding đang chạy Llama-2-7b** (dùng expert có sẵn) còn baseline chạy Llama-3-8B → bội số 8.3× lẫn cả chênh lệch model. Paper báo ATGR 1.03–1.07×.

---

## 9. Chi phí

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

## 10. Việc tiếp theo

1. Chạy `no_defense_local` full → mốc cho bảng thứ hai.
2. Thêm **Self-Reminder** (pre) + **SelfDefend** (post) → đủ 5/5 cho nhóm API. Cả hai chạy thuần API, không cần GPU.
3. Chạy `judge --task justeval` cho 9 method API đã có response — **đang thiếu hẳn một metric**.
4. Chạy full `--task all` cho 5 bài local đã code.
5. Quét `JBS_FIRST_M` trên full 300 (hiện đặt 2, **chưa kiểm còn tác dụng phòng thủ không**).
6. Thêm **ROSE · DRO · SafeInfer** (in) + **DeepRefusal · Targeted LAT** (intra).

> Nhóm target-API và nhóm target-local **không cùng thang** — tách 2 bảng, mỗi bảng có `no_defense` của chính nó.
