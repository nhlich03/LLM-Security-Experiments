# Bảng phân loại phương pháp phòng thủ

Cập nhật: FJD post→pre, ICAG in→pre, RC-RAG in→post, thêm SAGE (pre); bỏ AdapT + Constitutional AI (lạc chủ đề); thêm 4 pre: IA, Perplexity, RPO, Proxy Barrier; loại SUPMER + TRAIT (lạc chủ đề).

Cột **Ưu tiên** = thứ tự nên triển khai (P1 dễ nhất → P5 nặng nhất). Xem "Lộ trình" cuối file.

---

## LỘ TRÌNH TRIỂN KHAI (chạy P1 trước, tăng dần)

- **P1 — Prompt-only, 1-2 API call, KHÔNG train, KHÔNG cần local.** Làm trước để dựng + kiểm thử toàn pipeline nhanh. → SAGE, IA
- **P2 — Detector rẻ hoặc multi-call API đơn giản, không train.** → Perplexity, FJD, LLM Self Defense, Self-Refine
- **P3 — Multi-call / nhiều model, cần orchestration hơn, vẫn không train.** → Proxy Barrier, G4D, AutoDefense, RC-RAG
- **P4 — Cần train model phụ / setup offline / can thiệp decoding / O(n) → bắt buộc local, nặng.** → erase-and-check, RPO, ICAG, Prompt-Tuning, SafeDecoding, GeDi, TaskTracker
- **P5 — Fine-tune target vĩnh viễn, train nặng nhất.** → SecAlign

Lý do xếp vậy: P1-P3 chạy được hoàn toàn qua API (target = llama-3.1-8b Groq), dựng nhanh, cost chỉ là token. P4-P5 cần GPU local (target = Qwen 1.5B) vì phải train phụ hoặc xen vào decoding. Làm nhóm API trước cho ra kết quả sớm, nhóm local sau.

---

## TIẾN ĐỘ (đã triển khai)

ASR chấm bằng **classifier chính thức `HarmBench-Llama-2-13b-cls`** (GPU H100 MIG 40GB, server thuê) — thay các số Mistral-7b cũ trên Kaggle.

| Method | Trạng thái | ASR ↓ | over-refusal ↓ (judge2) | utility ↑ | cost |
|---|---|---|---|---|---|
| **no_defense** (mốc) | ✅ xong | 30.7% | 8.0% | (chưa) | 1 call |
| **SAGE** (pre, P1) | ✅ xong | 0.7% | 34.8% | (chưa) | 1 call |
| **IA** (pre, P1) | ✅ xong | 2.0% | 12.4% | (chưa) | 2 call |
| **G4D** (pre, P3) | ✅ xong | 7.0% | 10.8% | (chưa) | 4 call (~1600 tok) |
| **erase-and-check** (pre, P4) | ✅ xong | 14.7% | 8.4% | (chưa) | 1.6 call + filter 0.056s |

> Số Mistral-7b (Kaggle) cũ để tham khảo: no_defense 22.7%, SAGE 16.3%. Llama-13b nghiêm hơn với no_defense và "tha" cho SAGE/IA nhiều hơn → dùng Llama-13b làm chuẩn, đã re-judge toàn bộ.

Còn lại: chưa triển khai. (utility JustEval: metric đã sẵn, chưa chạy cho các method.)

---

## PRE-PROCESSING (can thiệp ở input, trước khi target xử lý)

| Ưu tiên | # | Paper | Venue · Năm | Link paper | GitHub | Cách chạy & cách tính cost |
|---|---|---|---|---|---|---|
| P1 ✅ | 1 | Self-Aware Guard Enhancement (SAGE) | Findings of ACL 2025 | https://aclanthology.org/2025.findings-acl.325.pdf | https://github.com/NJUNLP/SAGE | **Chạy:** ghép 2 instruction (analysis+response) vào input, sinh 1 lần; training-free. **Cost:** single-call, rẻ nhất. **Đo:** 1 call `target` mỗi request |
| P1 ✅ | 2 | Intention Analysis (IA) | COLING 2025 | https://arxiv.org/pdf/2401.06561 | https://github.com/alphadl/SafeLLM_with_IntentionAnalysis | **Chạy:** 2 vòng — vòng 1 phân tích ý định của prompt, vòng 2 trả lời theo policy dựa trên vòng 1; inference-only. **Cost:** ~2 call API. **Đo:** `intention` + `target` |
| P2 | 3 | Detecting Language Model Attacks with Perplexity | arXiv preprint 2023 | https://arxiv.org/pdf/2308.14132 | (không có repo chính thức) | **Chạy:** tính perplexity của prompt; PPL cao (suffix GCG) → flag & chặn trước khi sinh. **Cost:** 1 forward pass tính PPL (không sinh), rẻ; detector. **Đo:** `local("ppl_forward")` + `target` nếu pass |
| P2 | 4 | LLM Jailbreak Detection for (Almost) Free! (FJD) | Findings of EMNLP 2025 | https://aclanthology.org/2025.findings-emnlp.309.pdf | https://github.com/GuoruiC/FJD | **Chạy:** prepend affirmative instruction, chạy 1 forward token, xét confidence token đầu để flag; benign mới sinh full. **Cost:** first-token forward (gần free) + có điều kiện sinh full. **Đo:** `local("first_token_forward")` + `target` nếu pass. FJD-LI có train nhỏ |
| P3 | 5 | Proxy Barrier (ProB) | Findings of EMNLP 2025 | https://aclanthology.org/2025.findings-emnlp.528.pdf | (chưa xác nhận repo) | **Chạy:** proxy LLM đứng trước target, chỉ lặp lại input; lặp "fail" → phát hiện tấn công, chặn trước khi tới target. **Cost:** 2 model (1 call proxy + 1 call target nếu pass). **Đo:** `proxy_repeat` + `target` |
| P3 ✅ | 6 | Dynamic Guided and Domain Applicable Safeguards (G4D) | Findings of NAACL 2025 | https://aclanthology.org/2025.findings-naacl.368.pdf | https://github.com/IDEA-XL/G4D | **Chạy:** intent detect → (paraphrase nếu unsafe) → safety analyze → target trả lời với guidance. Retrieval TẮT (đúng main.py). **Cost:** 3 call (safe) / 4 call (unsafe). **Đo:** `intent_detect`+`paraphrase`+`safety_analyze`+`target` |
| P4 ✅ | 7 | Certifying LLM Safety (erase-and-check) | arXiv preprint 2023 | https://arxiv.org/pdf/2309.02705 | https://github.com/aounon/certified-llm-safety | **Chạy:** suffix mode, xoá 1..20 token cuối, filter DistilBERT (local GPU) chấm từng biến thể; bất kỳ biến thể harmful → chặn (refusal), pass → target. **Cost:** filter O(n) local (≤21 forward, batch 1) + `target` nếu pass. **Đo:** `local("filter_check")` + `target` |
| P4 | 8 | Robust Prompt Optimization (RPO) | NeurIPS 2024 | https://arxiv.org/pdf/2401.17263 | https://github.com/lapisrocks/rpo | **Chạy:** tối ưu 1 "defensive suffix" (offline), deploy = gắn suffix vào prompt rồi gọi target. **Cost:** setup một lần tối ưu suffix + 1 call target lúc infer. **Đo:** `train()` quanh tối ưu suffix + `target` mỗi request |
| P4 | 9 | In-Context Adversarial Game (ICAG) | EMNLP 2024 (main) | https://aclanthology.org/2024.emnlp-main.1121.pdf | https://github.com/YujunZhou/In-Context-Adversarial-Game | **Chạy:** offline chạy "game" attack-vs-defense (nhiều call) tạo ra safety system prompt; deploy = prepend prompt đó + 1 call target. **Cost:** setup một lần đắt + infer rẻ. **Đo:** `train()` quanh game + `target` mỗi request |
| P4 | 10 | Controlling Extraction of Memorized Data via Prompt-Tuning | ACL 2023 (short) | https://aclanthology.org/2023.acl-short.129.pdf | https://github.com/amazon-science/controlling-llm-memorization | **Chạy:** prepend soft prompt (đã train) vào input rồi sinh. **Cost:** train soft prompt (một lần, nhỏ) + 1 call target. **Đo:** `train()` quanh train soft prompt + `target`. *(chủ đề privacy)* |

---

## POST-PROCESSING (can thiệp ở output, sau khi model đã sinh)

| Ưu tiên | # | Paper | Venue · Năm | Link paper | GitHub | Cách chạy & cách tính cost |
|---|---|---|---|---|---|---|
| P2 | 11 | LLM Self Defense (Self-Examination) | arXiv preprint 2023 | https://arxiv.org/pdf/2308.07308 | https://github.com/poloclub/llm-self-defense | **Chạy:** target sinh → 1 call phụ hỏi "response có hại không" → nếu có thì thay bằng từ chối. **Cost:** 2 call. **Đo:** `target` + `self_exam` |
| P2 | 12 | SELF-REFINE: Iterative Refinement with Self-Feedback | NeurIPS 2023 | https://arxiv.org/pdf/2303.17651 | https://github.com/madaan/self-refine | **Chạy:** target sinh → call feedback phê bình → call refine viết lại, lặp k vòng. **Cost:** 1 + 2k call. **Đo:** `target` + loop `feedback`+`refine`. *(method chất lượng chung)* |
| P3 | 13 | AutoDefense: Multi-Agent LLM Defense | arXiv preprint 2024 | https://arxiv.org/pdf/2403.04783 | https://github.com/XHMY/AutoDefense | **Chạy:** target sinh → pipeline nhiều agent đọc response → quyết định. **Cost:** multi-call (target + N agent) — đắt. **Đo:** `target` + từng agent |
| P3 | 14 | Controlling Risk of RAG (Counterfactual Prompting) | Findings of EMNLP 2024 | https://aclanthology.org/2024.findings-emnlp.133.pdf | https://github.com/ict-bigdatalab/RC-RAG | **Chạy:** sinh answer ban đầu → tạo CF prompt thách thức → hỏi lại → fusion quyết định giữ/bỏ. **Cost:** multi-call. **Đo:** `target` + từng `cf_prompt` + `fusion`. *(cần setup RAG/retrieval)* |

---

## IN-PROCESSING (can thiệp lúc decoding, thay đổi TẠM THỜI)

| Ưu tiên | # | Paper | Venue · Năm | Link paper | GitHub | Cách chạy & cách tính cost |
|---|---|---|---|---|---|---|
| P4 | 15 | SafeDecoding: Safety-Aware Decoding | ACL 2024 (long) | https://aclanthology.org/2024.acl-long.303.pdf | https://github.com/uw-nsl/SafeDecoding | **Chạy:** mỗi bước decode kết hợp logits target với expert model (đã fine-tune) khuếch đại token an toàn. **Cost:** overhead decoding-time/token + train expert. **Đo:** `train()` expert + `local("guided_decode")`; overhead = trừ no_defense. Bắt buộc local |
| P4 | 16 | GeDi: Generative Discriminator Guided Generation | Findings of EMNLP 2021 | https://aclanthology.org/2021.findings-emnlp.424.pdf | https://github.com/salesforce/GeDi | **Chạy:** mỗi bước decode một discriminator LM reweight xác suất token kế tiếp qua Bayes. **Cost:** overhead decoding-time/token + train discriminator. **Đo:** như SafeDecoding. Bắt buộc local |
| P4 | 17 | Get my drift? Task Drift with Activation Deltas (TaskTracker) | IEEE SaTML 2025 | https://arxiv.org/pdf/2406.00799 | https://github.com/microsoft/TaskTracker | **Chạy:** trích activation delta, đưa vào probe (đã train) phát hiện task drift; detector, không sửa output. **Cost:** trích activation (~1 forward) + probe rẻ + train probe. **Đo:** `local("activation_extract")` + `local("probe")` + `train()` probe |

---

## INTRA-PROCESSING (sửa trọng số VĨNH VIỄN)

| Ưu tiên | # | Paper | Venue · Năm | Link paper | GitHub | Cách chạy & cách tính cost |
|---|---|---|---|---|---|---|
| P5 | 18 | SecAlign: Defending Prompt Injection with Preference Optimization | ACM CCS 2025 | https://arxiv.org/pdf/2410.05451 | https://github.com/facebookresearch/SecAlign | **Chạy:** offline train DPO trên cặp preference → ra model đã align; infer sinh bình thường. **Cost:** train một lần lớn (DPO) + infer ≈ model thường. **Đo:** `train()` quanh DPO + `target` mỗi request |

---

## Tồn đọng

- **Link GitHub chưa chắc:** Proxy Barrier — cần tìm/xác nhận repo chính thức. RPO đã có repo (lapisrocks/rpo). Perplexity không có repo (tự implement PPL filter, dễ).
- **Venue:** một số bài chỉ là **arXiv preprint** (Perplexity, erase-and-check, LLM Self Defense, AutoDefense) — chưa có kỷ yếu chính thức, ghi rõ trong bảng.
- **G4D**: đã verify + triển khai xong (3-4 call, retrieval TẮT theo main.py). ✅
- **Cân bằng nhóm:** pre 10, post 4, in 3, intra 1. intra rất mỏng — bổ sung thêm (adversarial training như R2D2, safety fine-tuning, unlearning). Nhóm pre đang phình to; cân nhắc chia sub-category trong pre (instruction-based: SAGE/IA/G4D vs detector-gate: Perplexity/FJD/ProB vs optimized-prompt: RPO/ICAG/Prompt-Tuning).
