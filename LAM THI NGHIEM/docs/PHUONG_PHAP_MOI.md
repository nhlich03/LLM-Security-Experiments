# Phương pháp phòng thủ MỚI tìm thêm (chưa có trong `BANG_PHUONG_PHAP.md`)

Kết quả research (07/2026) cho cả 4 nhóm **pre / post / in / intra**, kèm phân tích + đề xuất.
Mục tiêu: lấp khoảng trống (đặc biệt **intra** đang chỉ có 1 bài, **in** chỉ 3), và bổ sung method 2023-2025 mạnh.

> Ký hiệu: **API** = chạy được qua Groq (không cần GPU) · **LOCAL** = cần white-box/GPU · **train** = cần huấn luyện offline.
> **Fit** = hợp đo HarmBench(ASR)+XSTest(over-refusal) hay lệch chủ đề.

---

## Tổng quan nhanh

| Nhóm | Method | Venue · Năm | Loại | Chạy | Fit jailbreak |
|---|---|---|---|---|---|
| pre | Self-Reminder | Nature MI 2023 | augmenter | API, P1 | ✅ |
| pre | In-Context Defense (ICD) | arXiv 2023 | augmenter (few-shot) | API, P1 | ✅ (yếu với many-shot) |
| pre | Goal Prioritization | ACL 2024 | augmenter | API, P1 | ✅ mạnh |
| pre | Paraphrase / Retokenization | arXiv 2023 | **transformer** (rewrite input) | API, P2 | ✅ (mạnh với GCG) |
| pre | SmoothLLM | arXiv 2023 | perturb+vote (detector-lai) | API, P3 (N call) | ✅ (mạnh với suffix) |
| pre | Llama Guard (input mode) | arXiv 2023 | **detector** (gate input) | API Groq / LOCAL | ✅ |
| post | **Llama Guard 3** | Meta 2024 | detector | **API Groq** | ✅ (over-flag XSTest) |
| post | **WildGuard** | NeurIPS 2024 | detector (có refusal head) | LOCAL 7B | ✅✅ (đo cả 2 trục) |
| post | ShieldGemma (2B/9B/27B) | Google 2024 | detector | LOCAL (2B rẻ) | ✅ |
| post | Aegis / Aegis 2.0 | 2024/2025 | detector | LOCAL 7-8B | ✅ (Permissive/Defensive) |
| post | **Aligner** | NeurIPS 2024 Oral | **transformer** (rewrite output) | upstream API + corrector LOCAL | ✅ (giữ benign tốt) |
| post | Self-Guard | NAACL 2024 | detector (self-tag, **train**) | LOCAL train | ✅ (breadth) |
| in | **SafeInfer** | AAAI 2025 | steer activation + logit | LOCAL | ✅ |
| in | **Jailbreak Antidote** | ICLR 2025 | sparse activation edit (5% dim) | LOCAL | ✅ (knob safety/utility) |
| in | **InferAligner** | EMNLP 2024 | safety steering vector | LOCAL | ✅ (chỉ steer khi harmful) |
| in | ROSE | Findings ACL 2024 | contrastive decoding (logit) | LOCAL (2× forward) | ✅ training-free |
| in | SafeInt | Findings EMNLP 2025 | intervention module (train nhỏ) | LOCAL | ✅ |
| in | DRO | ICML 2024 | soft-prompt theo refusal-direction | LOCAL (train soft prompt) | ✅ |
| in | Self-CD | ACL 2024 | contrastive decoding | LOCAL | ⚠️ giảm over-refusal (bổ XSTest, không chống jailbreak) |
| intra | **Circuit Breakers (RepE)** | NeurIPS 2024 | LoRA + RepE loss | LOCAL train (rẻ) | ✅✅ HarmBench-native |
| intra | **DeRTa** | ACL 2025 | SFT + RTO (refuse-mid-gen) | LOCAL train (rẻ) | ✅ mạnh với prefilling |
| intra | R2D2 | ICML 2024 | adversarial training vs GCG | LOCAL train (đắt) | ✅ (baseline HarmBench) |
| intra | Targeted LAT | 2024 | latent adversarial training | LOCAL train (vừa) | ✅ |
| intra | Safe RLHF (→ safety DPO) | ICLR 2024 | RLHF ràng buộc / DPO | LOCAL train | ✅ (baseline alignment) |
| intra | TAR | ICLR 2025 | meta-learn tamper-resistant | LOCAL train (rất đắt) | ⚠️ threat model = fine-tune attack, không phải prompt jailbreak |

---

## ✅ Tình trạng CODE (đã clone + soi repo 26/07/2026)

Ký hiệu: **✅ code thật** (import/chạy được) · **📦 weight/checkpoint** (repo ít code, dùng ckpt HF) · **⚠️ một phần** · **❌ không có code / không repo** (tự implement từ paper).

**PRE**
| Method | Code | Repo | Port ngay? |
|---|---|---|---|
| Self-Reminder | ✅ | `yjw1029/Self-Reminder` (`src/pia_defense.py`, `defense_templates`) | ✅ prompt-only, copy là chạy |
| Goal Prioritization | ✅ | `thu-coai/JailbreakDefense_GoalPriority` (`utils.py::add_defense('priority')`) | ✅ prompt-only |
| SmoothLLM | ✅ | `arobey1/smooth-llm` (`lib/perturbations.py`, `defenses.py`) | ✅ (nhưng tốn N call/prompt) |
| Paraphrase | ⚠️ | `neelsjain/baseline-defenses` — paraphrase ✅ (2 call); **retokenization ❌** (đụng tokenizer, API không được) | phần paraphrase ✅ |
| ICD | ❌ | repo `PKU-ML/adv-icl` **404** | tự viết few-shot refusal từ paper (~5 dòng) |

**POST**
| Method | Code | Nguồn | Chạy |
|---|---|---|---|
| WildGuard | ✅ | `pip install wildguard` (code import được) | LOCAL 7B — **ngon nhất** |
| Aligner-7B | 📦 | repo chỉ có code **train**; weight `aligner/aligner-7b-v1.0` | LOCAL 7B (rewriter) |
| ShieldGemma-2B | 📦 | weight `google/shieldgemma-2b` + snippet | LOCAL 2B (rẻ, ra xác suất) |
| Llama Guard | 📦 | repo `PurpleLlama` **không có code**; weight HF | ⚠️ Groq **đã bỏ `llama-guard-3-8b`** → dùng `openai/gpt-oss-safeguard-20b`; hoặc host local 8B |

**IN**
| Method | Code | Repo | Port |
|---|---|---|---|
| DRO | ✅ | `chujiezheng/LLM-Safeguard` (`train.py`, `estimate.py`) | **dễ nhất** — 1 model, thêm nhánh template |
| SafeInfer | ✅ | `NeuralSentinel/SafeInfer` (`FVPlug.py`, `MA_Inference.py`) | vừa — cần thêm 1 con `M_unsafe` (2 model) |
| ROSE | ✅ | `WHU-ZQH/ROSE` (**có repo**, trước tưởng không) | dễ — contrastive decoding training-free |
| InferAligner | ❌ | `Jihuai-wpy/InferAligner` — **chỉ result + benchmark, 0 file .py** ("coming soon" từ 1/2024) | tự implement từ paper |
| SafeInt | ❌ | không có repo | tự implement (dùng `pyreft`/LoReFT) |
| Jailbreak Antidote | ❌ | không có repo | tự implement từ paper |

**INTRA**
| Method | Code | Repo | Checkpoint sẵn | Port cho Llama-3.1-8B/40GB |
|---|---|---|---|---|
| **Circuit Breakers** | ✅ | `GraySwanAI/circuit-breakers` (`lorra_circuit_breaker.py`) | ✅ `GraySwanAI/Llama-3-8B-Instruct-RR` | **dễ** — LoRA, đã Llama-3-8B (hạ batch/ctx trên 40GB) |
| **DeRTa** | ✅ | `RobustNLP/DeRTa` (LoRA path) | ✅ `Youliang/llama3-8b-instruct-lora-derta-100step` | **dễ** — LoRA, đã Llama-3-8B |
| R2D2 | ✅ | trong `centerforaisafety/HarmBench` | ❌ | nặng (GCG mỗi step), base Zephyr |
| **Targeted LAT** | ✅ | `aengusl/latent-adversarial-training` (bản Targeted, arXiv 2407.15549). *Repo cũ `thestephencasper/latent_adversarial_training` = bản trước, fork kiến trúc Llama-2, khó port* | ✅ org HF **`LLM-LAT`**: `robust-llama3-8b-instruct`, `llama3-8b-instruct-rt-jailbreak-robust2/3` *(verify khi tải)* | **dễ** — đã Llama-3-8B, vừa 40GB (khớp L127) |
| Safe-RLHF | ✅ | `PKU-Alignment/safe-rlhf` (full-FT 4 model) | ✅ `PKU-Alignment/beaver-7b-v1.0` | không train nổi trên 40GB → **dùng ckpt Beaver** |

### 🎯 Nên làm tiếp (có code/ckpt, port được ngay)
- **PRE (API, nhanh nhất):** Self-Reminder, Goal-Prioritization (prompt-only) → thêm 2 dòng trong 1 buổi. + SmoothLLM (tốn N call), Paraphrase (2 call).
- **POST:** WildGuard (local, code sẵn, đo cả 2 trục) · Llama Guard qua Groq (`gpt-oss-safeguard-20b`) · Aligner/ShieldGemma (dùng ckpt).
- **IN (local GPU):** DRO ⭐ (dễ nhất) · ROSE (training-free) · SafeInfer (cần M_unsafe).
- **INTRA (local GPU, LoRA):** Circuit Breakers ⭐ · DeRTa ⭐ — **cả 2 có checkpoint HF sẵn** → đánh giá được mà **khỏi train**.

### ❌ Dead-end (không có code — phải tự viết từ paper)
InferAligner, SafeInt, Jailbreak Antidote (in) · ICD (pre, nhưng trivial). Còn lại Safe-RLHF/LAT/R2D2 có code nhưng nặng/khó port.

---

## PRE-PROCESSING (6)

- **Self-Reminder** — [Nature MI 2023](https://www.nature.com/articles/s42256-023-00765-8) · [repo](https://github.com/yjw1029/Self-Reminder). Bọc query bằng system prompt an toàn + câu nhắc cuối. Prompt-only, 1 call. ASR 67%→19%; coi chừng over-refusal. **P1, API.**
- **In-Context Defense (ICD)** — [arXiv 2310.06387](https://arxiv.org/abs/2310.06387). Prepend vài cặp (harmful→refusal) làm few-shot. 1 call, no train. Yếu với many-shot jailbreak. **P1, API.**
- **Goal Prioritization** — [ACL 2024](https://arxiv.org/abs/2311.09096) · [repo](https://github.com/thu-coai/JailbreakDefense_GoalPriority). Chèn chỉ dẫn "ưu tiên an toàn hơn hữu ích" + template internal-thought. Bản inference = prompt-only. ASR 66%→3.6%. **P1 (inference) / P4 (train), API.**
- **Paraphrase / Retokenization** — [arXiv 2309.00614](https://arxiv.org/abs/2309.00614). Paraphrase lại prompt rồi mới đưa target (phá suffix GCG). **Transformer rewrite input** — mẫu sạch nhất của loại này. 2 call. **P2, API.**
- **SmoothLLM** — [arXiv 2310.03684](https://arxiv.org/abs/2310.03684) · [repo](https://github.com/arobey1/smooth-llm). Tạo N bản nhiễu ký tự, vote refusal. Mạnh vs GCG/PAIR. Đắt (N=6-10 call). **P3, API.**
- **Llama Guard (input mode)** — xem POST; dùng ở cổng input để chặn trước.

## POST-PROCESSING (6)

- **Llama Guard 3** — [arXiv 2312.06674](https://arxiv.org/abs/2312.06674) · [PurpleLlama](https://github.com/meta-llama/PurpleLlama). Classifier an toàn (Llama-3.1-8B) chấm (prompt, response) theo taxonomy. **Groq host `llama-guard-3-8b` → chạy được API, không cần GPU.** Detector. **P1.** (over-flag XSTest — điểm đáng đo.)
- **WildGuard** — [NeurIPS 2024](https://arxiv.org/abs/2406.18495) · [repo](https://github.com/allenai/wildguard). 1 model ra **3 nhãn**: prompt-harm, response-harm, **refusal**. Hợp nhất với đo **cả 2 trục** (ASR + over-refusal). LOCAL 7B. **P1.**
- **ShieldGemma** — [arXiv 2407.21772](https://arxiv.org/abs/2407.21772). Gemma2 2B/9B/27B chấm output, ra xác suất (threshold tuỳ chỉnh). 2B rẻ. LOCAL. **P2.**
- **Aegis / Aegis 2.0** — [arXiv 2404.05993](https://arxiv.org/abs/2404.05993) / [2501.09004](https://arxiv.org/abs/2501.09004). Biến thể Llama-Guard, có cặp **Permissive/Defensive** → ablation trade-off ASR↔over-refusal. LOCAL. **P2/P3.**
- **Aligner** — [NeurIPS 2024 Oral](https://arxiv.org/abs/2402.02416) · [repo](https://github.com/PKU-Alignment/aligner). Model nhỏ đứng SAU target, đọc (query, answer) rồi **viết lại** answer an toàn (copy-and-correct). **Transformer rewriter** duy nhất — giữ benign tốt. Upstream = Groq, corrector = LOCAL 7B. **P2.**
- **Self-Guard** — [NAACL 2024](https://aclanthology.org/2024.naacl-long.92/). Fine-tune model tự gắn tag `[harmful]/[harmless]` cuối response. **Cần train**, ít checkpoint. **P3, LOCAL train.**

## IN-PROCESSING (6) — xu hướng chính = **activation steering / contrastive decoding**

- **SafeInfer** — [AAAI 2025](https://arxiv.org/abs/2406.12274) · [repo](https://github.com/NeuralSentinel/SafeInfer). 2 pha decode-time: shift hidden state theo hướng an toàn (từ demo, offline không train) + reshape phân phối token. LOCAL. **✅**
- **Jailbreak Antidote** — [ICLR 2025](https://arxiv.org/abs/2410.02298). Cộng `α·d_safe` vào residual stream, chỉ ~5% chiều (mask). `d_safe` = PCA benign-vs-harmful (offline, không train). `α` = knob safety/utility, **0 overhead**. Đã chạy trên Llama-3.1-8B. *(chưa có repo chính thức → tự implement.)* **✅**
- **InferAligner** — [EMNLP 2024](https://aclanthology.org/2024.emnlp-main.585/) · repo `Jihuai-wpy/InferAligner` **CHỈ có result, 0 file code** ("coming soon" từ 1/2024). Safety Steering Vector = hiệu activation harmful−harmless; chỉ cộng khi phát hiện input harmful. LOCAL. **⚠️ phải tự implement từ paper**
- **ROSE** — [Findings ACL 2024](https://aclanthology.org/2024.findings-acl.814/) · repo [`WHU-ZQH/ROSE`](https://github.com/WHU-ZQH/ROSE) (**có code**). Contrastive decoding **training-free**: logit = normal − w·logit(reverse-prompt độc). 2× forward. LOCAL. **✅**
- **SafeInt** — [Findings EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.450/). Module intervention ở lớp giữa, kéo representation jailbreak về "vùng từ chối". Train module nhỏ, base frozen. LOCAL. **✅**
- **DRO** — [ICML 2024](https://arxiv.org/abs/2401.18018). Tối ưu **soft prompt** theo refusal-direction (harmful đi theo hướng từ chối, benign ngược lại). Borderline in/pre (soft prompt = trained embedding). LOCAL. **✅**
- *(bổ sung)* **Self-CD** — [ACL 2024](https://arxiv.org/abs/2401.17633): contrastive decoding **giảm over-refusal** (bổ cho XSTest), KHÔNG chống jailbreak — dùng nếu survey có nhánh over-refusal.

## INTRA-PROCESSING (6) — lấp nhóm mỏng nhất

- **Circuit Breakers (RepE)** — [NeurIPS 2024](https://arxiv.org/abs/2406.04313) · [repo](https://github.com/GraySwanAI/circuit-breakers). LoRA + loss RepE: **phá biểu diễn nội bộ sinh nội dung hại** (đẩy hidden state harmful về hướng vô nghĩa) + retain loss giữ benign. **HarmBench-native**, LoRA rẻ (~vài giờ 1 GPU). **⭐ nên thêm đầu tiên.**
- **DeRTa** — [ACL 2025](https://arxiv.org/abs/2407.09121) · [repo](https://github.com/RobustNLP/DeRTa). SFT dạy model **từ chối giữa chừng** (prepend harmful prefix rồi refuse) + RTO. Chống prefilling mạnh. SFT + LoRA rẻ, dễ reproduce. **⭐**
- **R2D2** — [ICML 2024 / HarmBench](https://arxiv.org/abs/2402.04249) · [repo](https://github.com/centerforaisafety/HarmBench). Adversarial training vs GCG (pool động). Baseline chính thức của HarmBench → số so trực tiếp. **Đắt** (full-FT + GCG inner loop). Base Zephyr-7B.
- **Targeted LAT** — [arXiv 2407.15549](https://arxiv.org/abs/2407.15549) · [repo](https://github.com/thestephencasper/latent_adversarial_training) · [ckpt LLM-LAT](https://huggingface.co/LLM-LAT). Adversarial training trong **không gian latent** (nhiễu activation), rẻ hơn GCG. Base Llama-3-8B. Vừa 40GB với LoRA.
- **Safe RLHF** — [ICLR 2024](https://arxiv.org/abs/2310.12773) · [repo](https://github.com/PKU-Alignment/safe-rlhf). Tách reward (hữu ích) + cost (hại), PPO ràng buộc Lagrangian. **Full RLHF không vừa 40GB** → dùng **safety-DPO/LoRA** làm bản đại diện khả thi.
- **TAR** — [ICLR 2025](https://arxiv.org/abs/2408.00761) · [repo](https://github.com/rishub-tamirisa/tamper-resistance). Meta-learn để safety **sống sót qua fine-tune tấn công**. ⚠️ Threat model = weight-tampering, KHÔNG phải prompt jailbreak → để làm đại diện "tamper-resistance", đừng so ASR trực tiếp. Rất đắt.

---

## Ngoài phạm vi (ghi trong taxonomy, KHÔNG chạy trên HarmBench/XSTest)

- **RMU / WMDP unlearning** — [ICML 2024](https://arxiv.org/abs/2403.03218): xoá kiến thức nguy hiểm, đo bằng **WMDP multiple-choice**, không phải ASR/over-refusal. Là flavor "unlearning" của intra nhưng lệch metric.
- **Refusal-direction orthogonalization** (Arditi, [NeurIPS 2024](https://arxiv.org/abs/2406.11717)) — chủ yếu là **tấn công/ablation** (gỡ refusal). Là nền cơ học cho họ steering (Circuit Breakers, InferAligner), không phải defense đứng riêng.
- **Token Highlighter, RA-LLM** — thực chất là **input-side**, không phải post (agent post đã loại).
- **SelfCheckGPT** (đã có trong bảng) — hallucination detection, không phải jailbreak.

---

## Phân tích & đề xuất

**1. Lấp khoảng trống.** Trước đây intra=1, in=3. Sau research:
- **intra**: có ngay 5-6 ứng viên hợp jailbreak; **Circuit Breakers** là lựa chọn số 1 (HarmBench-native, LoRA rẻ, giữ utility), kế đó **DeRTa** (SFT rẻ, chống prefilling). R2D2 để làm baseline chuẩn nhưng đắt.
- **in**: xu hướng 2024-2025 rõ ràng là **activation steering** (SafeInfer/InferAligner/Jailbreak Antidote) + **contrastive decoding** (ROSE/Self-CD). Tất cả cần white-box → chạy trên server GPU với target local Llama-3.1-8B.

**2. Quick win chạy được NGAY qua API (không cần GPU), giống nhóm đã làm:**
- pre: **Self-Reminder, ICD, Goal-Prioritization** (P1, 1 call) → thêm 3 dòng rất nhanh.
- pre: **Paraphrase** (P2, 2 call), **SmoothLLM** (P3, N call).
- post: **Llama Guard 3 qua Groq** (`llama-guard-3-8b`) — detector chuẩn, API luôn, không cần host.
→ Đây là các bài nên làm tiếp trước khi động tới nhóm local-train.

**3. Cân bằng detector vs transformer.** Survey nên có đủ 2 kiểu mỗi nhóm:
- transformer (biến đổi): Paraphrase (pre-input), Aligner (post-output), steering (in).
- detector (chặn/flag): Llama Guard/WildGuard (gate), SmoothLLM-vote.

**4. Cần GPU local (target Llama-3.1-8B):** toàn bộ in-processing + intra + WildGuard/Aligner/ShieldGemma. Máy H100 MIG 40GB kham được (LoRA cho train, inference 8B fp16 ~16GB). Full-RLHF/TAR/R2D2-full thì quá tải → dùng bản LoRA/rút gọn.

**5. Shortlist đề xuất (ưu tiên reproducible + đúng chủ đề):**
- **pre:** Self-Reminder, Goal-Prioritization, Paraphrase (API, nhanh).
- **post:** Llama Guard 3 (API), WildGuard (local, đo cả 2 trục), Aligner (rewriter).
- **in:** SafeInfer, InferAligner (có repo), ROSE (training-free).
- **intra:** Circuit Breakers ⭐, DeRTa ⭐, + safety-DPO làm baseline alignment.

*(Nguồn đầy đủ ở link inline. Method chưa có repo chính thức: Jailbreak Antidote, ROSE, SafeInt — phải tự implement từ paper.)*
