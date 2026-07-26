# Kiểm tra 19 paper nhóm tự tìm cho nhóm INTRA

Ngày check: 26/07/2026. Nguồn: paper gốc + đọc trực tiếp README/repo.
File song song: `README_PHUONG_PHAP_IN.md` (7 bài nhóm IN).

Nhắc lại định nghĩa INTRA của mình (`CLAUDE.md` §2):
> **intra-processing** = sửa trọng số **VĨNH VIỄN**, tạo target model mới (safety fine-tuning, DPO/RLHF, adversarial training).
> Ranh giới in vs intra = **tạm thời vs vĩnh viễn**.

Nhưng "có fine-tune" **chưa đủ** để nhận vào survey. Phải qua **2 cửa**:
1. **Cửa 1 — cơ chế:** có sửa weight vĩnh viễn không? → đúng INTRA.
2. **Cửa 2 — threat model:** có chống **prompt jailbreak** không? Nếu chống thứ khác (RAG noise, backdoor poisoning, synonym attack lên classifier, harmful fine-tuning) thì **không đo được bằng HarmBench + XSTest** của mình.

Kết quả: **19 bài → chỉ 5 bài qua cả 2 cửa**, trong đó **3 bài có checkpoint sẵn** nên chạy được ngay dù mình không đủ GPU để train.

---

## ⚠️ Ràng buộc quyết định tất cả: mình KHÔNG train nổi nhóm INTRA

Server hiện tại = **1 × H100 MIG 40GB**, batch ≤ 4 (xem `SERVER_SSH_HUONGDAN.md`). Trong khi:
- Safe RLHF: tác giả dùng **8 × A800-80GB** cho LLaMA-7B
- Fine-Grained RLHF: **80G A100**
- Booster: 3 lần forward/backward mỗi step (harmful + perturbed + alignment) trên Llama2-7B

→ **Chiến lược bắt buộc cho toàn bộ nhóm INTRA: dùng checkpoint tác giả đã public, chạy inference-only**, rồi đo ASR/over-refusal/utility đúng như `no_defense` chế độ local. Cột `train()` trong cost meter ghi theo **số liệu báo cáo trong paper**, ghi rõ "không tự train".

Cách này đã chốt từ trước cho **Circuit Breakers** (`GraySwanAI/Llama-3-8B-Instruct-RR`) và **DeRTa** — nay áp dụng tiếp cho 3 bài dưới đây.

---

## Kết luận nhanh — 19 bài

Ký hiệu: **✅ đạt** · **⚠️ có vấn đề** · **❌ không đạt**

### Nhóm A — Đúng INTRA + đúng chủ đề jailbreak (5 bài, NÊN LẤY)

| # | Paper | Venue · Năm | GitHub | Code thật? | Ckpt sẵn? | Độ khó | Đề xuất |
|---|---|---|---|---|---|---|---|
| 1 | **DeepRefusal** (Beyond Surface Alignment) | Findings EMNLP 2025 | [YuanBoXie/DeepRefusal](https://github.com/YuanBoXie/DeepRefusal) | ✅ (README mỏng) | ✅✅ **Llama-3-8B-Instruct-DeepRefusal** | 🟢 Thấp (inference-only) | **LÀM ĐẦU TIÊN** |
| 2 | **C-AdvUL / C-AdvIPO** (Efficient AdvTrain w/ Continuous Attacks) | **NeurIPS 2024 Spotlight** | [sophie-xhonneux/continuous-advtrain](https://github.com/sophie-xhonneux/continuous-advtrain) | ✅ đầy đủ | ✅✅ Phi-CAT, Phi-CAPO, Zephyr-CAT, Llama-2/3 | 🟢 Thấp (inference-only) | **LÀM** |
| 3 | **Safe RLHF** | **ICLR 2024 Spotlight** | [PKU-Alignment/safe-rlhf](https://github.com/PKU-Alignment/safe-rlhf) | ✅✅ pipeline đầy đủ | ✅✅ Beaver-7B v1/v2/v3 + reward/cost model | 🟢 dùng ckpt / 🔴 tự train (8×A800) | **LÀM** (đã có trong `PHUONG_PHAP_MOI.md`) |
| 4 | **ReFAT** (Refusal Feature Adversarial Training) | arXiv 2409.20089 · OpenReview (chưa xác nhận venue) | ❌ không tìm thấy repo | ❌ tự implement | ❌ | 🟡 Trung bình (ý tưởng rẻ, 1700× rẻ hơn R2D2) | Cân nhắc |
| 5 | **Booster** | **ICLR 2025 Oral** | [git-disl/Booster](https://github.com/git-disl/Booster) | ✅ trainer đầy đủ | ❌ | 🔴 Cao (phải tự train) | ⚠️ Threat model = **harmful fine-tuning**, không phải prompt jailbreak |

### Nhóm B — Đúng INTRA nhưng lệch chủ đề (8 bài, chỉ nên làm related work)

| # | Paper | Venue · Năm | GitHub | Code thật? | Vấn đề |
|---|---|---|---|---|---|
| 6 | **Quark** (Reinforced Unlearning) | NeurIPS 2022 | [GXimingLu/Quark](https://github.com/GXimingLu/Quark) | ⚠️ có nhưng mỏng (5 commit) | Detox **GPT-2**, reward = PerspectiveAPI. Không có khái niệm jailbreak |
| 7 | **DAPT Detox** (Exploring the Limits…) | NeurIPS 2022 | [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) `examples/detxoify_lm` | ✅ | Detox, chạy trên Megatron 126M–530B. Quá nặng, sai stack |
| 8 | **Fine-Grained RLHF** | NeurIPS 2023 | [allenai/FineGrainedRLHF](https://github.com/allenai/FineGrainedRLHF) | ✅✅ | Task = **QA + detox**, base **T5-large / GPT-2**, cần 80G A100. Không phải jailbreak |
| 9 | **DRLC / Dense Rewards from LM Critic** | EMNLP 2024 | ❌ không tìm thấy | ❌ | Task = sentiment control / detox / summarization |
| 10 | **FIGA** (Beyond Imitation) | ICLR 2024 | [RUCAIBox/FIGA](https://github.com/RUCAIBox/FIGA) + [SPA dataset](https://huggingface.co/datasets/RUCAIBox/SPA) | ✅ | Alignment **chất lượng chung**, không safety-specific |
| 11 | **InstructGPT** | NeurIPS 2022 (OpenAI) | ❌ | ❌ **không code, không data** | Bài nền RLHF. Labeler data private. Không phải defense — chỉ là background |
| 12 | **Okapi** | EMNLP 2023 (Demo) | [nlp-uoregon/Okapi](https://github.com/nlp-uoregon/Okapi) | ✅ | RLHF **đa ngôn ngữ 26 thứ tiếng**. Mục tiêu là instruction-following, không phải an toàn |
| 13 | **Self-Criticism** (HHH) | EMNLP 2023 (Industry) | ❌ không tìm thấy | ❌ | HHH alignment qua self-critique rồi fine-tune. Không code, industry track |

### Nhóm C — KHÔNG phải INTRA cho LLM / threat model khác hẳn (6 bài, LOẠI)

| # | Paper | Venue · Năm | GitHub | Vì sao loại |
|---|---|---|---|---|
| 14 | **ATM** (Adversarial Tuning Multi-agent) | EMNLP 2024 | ❌ không có | Threat model = **tài liệu nhiễu/bịa trong RAG**. Đo bằng QA F1/EM, không phải ASR |
| 15 | **RAAT** (Adaptive Adversarial Training cho RAG) | ACL 2024 | [calubkk/RAAT](https://github.com/calubkk/RAAT) | Cùng lý do với ATM — RAG noise robustness, LLaMA-2 7B, đo F1/EM |
| 16 | **Impact of Adversarial Training on Robustness and Generalizability** | Findings ACL 2023 | [m1k2zoo/RobustDG](https://github.com/m1k2zoo/RobustDG) (liên quan) | **Bài phân tích**, không đề xuất defense mới. Đối tượng là encoder LM |
| 17 | **FLAT** (Look at Both Prediction and Interpretation) | AAAI 2022 | [uva-nlp/FLAT](https://github.com/uva-nlp/FLAT) | Adversarial training cho **classifier** (LSTM/CNN/BERT/DeBERTa), tấn công thay từ đồng nghĩa. Không phải LLM sinh |
| 18 | **Fortifying Toxic Speech Detectors Against Veiled Toxicity** | EMNLP 2020 | [xhan77/veiled-toxicity-detection](https://github.com/xhan77/veiled-toxicity-detection) | Củng cố **detector** toxic speech thời BERT (pytorch 1.3.1, pytorch-pretrained-bert 0.6.1). Không phải LLM |
| 19 | **Moderate-fitting** | NeurIPS 2022 | [thunlp/Moderate-fitting](https://github.com/thunlp/Moderate-fitting) | Threat model = **backdoor / data poisoning** khi fine-tune PLM. Khác trục hoàn toàn |

**Tổng kết số học:** 19 bài → **5 qua cả 2 cửa** (nhưng 1 trong đó lệch threat model) → thực chất **3 bài dùng được ngay** (DeepRefusal, C-AdvUL/IPO, Safe RLHF) + **1 bài phải tự viết** (ReFAT) + **1 bài nhóm riêng** (Booster).

---

## NHÓM A — chi tiết

### 1. DeepRefusal (Beyond Surface Alignment) — Findings EMNLP 2025

**Paper:** https://arxiv.org/abs/2509.15202 · **ACL:** https://aclanthology.org/2025.findings-emnlp.956/ · **Repo:** https://github.com/YuanBoXie/DeepRefusal

**Phân loại:** INTRA chuẩn — fine-tune tạo model mới, weight đổi vĩnh viễn.

**Ý tưởng:** alignment hiện tại chỉ "nông" ở bề mặt — cơ chế từ chối nằm gọn trong **một refusal direction** trong residual stream, nên attacker chỉ cần **ablate** hướng đó là model xuôi theo. DeepRefusal lật ngược: trong lúc fine-tune, **chủ động ablate refusal direction theo xác suất**, trải **qua nhiều layer và nhiều độ sâu token** → ép model **tự dựng lại** cơ chế từ chối từ trạng thái đã bị jailbreak. Kết quả: không còn một hướng đơn lẻ để phá.

**Flow chạy:**
1. *Offline bước 1:* **tính refusal direction** — mean activation của harmful prompt trừ mean của benign prompt tại mọi cặp (layer `l`, token position `i`), ra `|I|×L` ứng viên, rồi **lọc heuristic chọn đúng MỘT hướng** `r(l*,i*)` thoả 2 ràng buộc: cộng vào → ép model từ chối cả prompt vô hại; xoá đi → model trả lời cả prompt độc hại. Theo Arditi et al. 2024.
   → Tác giả có thử **tính động trong lúc train** nhưng bỏ, vì hướng "rất không ổn định, khó thoả 2 ràng buộc theo thời gian thực". Chốt: **1 hướng tĩnh, tính offline**.
2. *Offline bước 2:* LoRA fine-tune với **PAA (Probabilistic Activation Ablation)** — mỗi layer `l` bốc `Q_l ~ Bernoulli(p)`, mỗi vị trí token `t` bốc `M_l,t ~ Bernoulli(p)`, rồi `h' ← h − Q·M·(r̂ r̂ᵀ h)`. Tức là **xoá ngẫu nhiên refusal direction theo cả chiều sâu layer lẫn chiều sâu token** → model bị đặt vào "trạng thái đã bị jailbreak" và buộc phải học cách từ chối lại từ trạng thái đó.
3. *Deploy:* **không có gì đặc biệt** — chỉ là một model đã align, sinh bình thường, cost = đúng 1 call như `no_defense`. Đây là ưu điểm lớn của cả nhóm INTRA: **inference overhead = 0**.

**Số LLM:** 1 (chính target).

**Model gốc (4 họ):** Llama3-8B-instruct, Llama2-7B-instruct, Mistral-7B-Instruct-v0.2, Gemma-7B-it.
**Attack (7, không phải 6):** No Attack, Manual (HumanJailbreaks của HarmBench), CodeAttack, GCG, Refusal, Refusal-Transfer, Prefilling.
**Baseline so sánh:** RT, RT-Augmented, LAT, **CAT** (= bài #2 dưới đây), **CircuitBreaker**.

**Train cái gì — paper §5.1 ghi rõ, RẺ:**

| Hạng mục | Con số |
|---|---|
| Hardware | **1 × NVIDIA A100 80GB** |
| Thời gian | **~45 phút**, đúng **1 epoch** |
| Phương pháp | **LoRA**, alpha=16, rank=16 (copy y hệt CircuitBreaker) |
| Batch size | 16 |
| Siêu tham số riêng | PAA probability **p = 0.5** |

**Data (~6,500 sample, TẤT CẢ public — không phải xin ai):**
| Nguồn | Số lượng | Vai trò |
|---|---|---|
| **CircuitBreaker** (Zou et al. 2024) | 2,000 | harmful, **có áp prefill augmentation** (Eq. 11) |
| **UltraChat** (HF) | 4,000 | benign, giữ năng lực ngôn ngữ |
| **XSTest + Or-bench** | 500 | ghì over-refusal xuống |

⚠️ **Prefill augmentation là bắt buộc, không bỏ qua được.** Ablation (Table 3): bỏ bước này thì ASR của GCG nhảy **2% → 50%**, Prefilling nhảy **0.4% → 83.7%**.

**Test set của họ:** 500 sample từ AdvBench + **HarmBench** + JailbreakBench; GCG chỉ 100 sample; over-refusal 200 prompt từ XSTest + Or-bench. → **Trùng đúng 2 metric của mình (HarmBench + XSTest)** nên số liệu đối chiếu trực tiếp được.

**Checkpoint:** ✅ [`skysys00/Meta-Llama-3-8B-Instruct-DeepRefusal`](https://huggingface.co/skysys00/Meta-Llama-3-8B-Instruct-DeepRefusal) (release 09/04/2026).

**Kết quả họ báo (Llama3-8B-instruct), ASR%:** No Attack 0.0 · Manual 0.0 · CodeAttack **0.2** (base 87.1) · GCG 2.0 · Refusal-Transfer 0.4 · Refusal 0.2 · Prefilling 0.4.
**Năng lực:** MMLU 63.61 (base 63.82), GSM8k 72.40 (base 75.44), MT-bench **5.94 (base 6.89 — tụt đáng kể)**.

**Về repo:** README **rất mỏng** — chỉ có `src/`, **không có lệnh train**, không nêu dataset. Tác giả ghi nguyên văn: *"This work is built on Refusal Direction and CircuitBreaker... Please refer to the corresponding repository for the **code** to obtain the refusal direction and output evaluation."* → trỏ sang lấy **CODE**, không phải lấy data (data đã nêu đủ trong paper).

**Độ khó:**
- **🟢 THẤP nếu dùng ckpt** — tải `Meta-Llama-3-8B-Instruct-DeepRefusal`, chạy inference. 8B fp16 ≈ 16GB, **vừa MIG 40GB**.
- **🟡 TRUNG BÌNH nếu tự train** — data public hết, LoRA 45 phút trên A100 80GB. Trên MIG 40GB: LoRA r=16 trên 8B gánh được, nhưng **batch 16 sẽ chật** (ghi chú vận hành: batch ≤ 4) → phải gradient accumulation, thời gian đội lên ~1.5–3h. Vẫn khả thi. Công phải bỏ ra chủ yếu là **ghép code** từ 3 repo (DeepRefusal `src/` + refusal_direction + circuit-breakers), vì repo chính không có lệnh train.

**⚠️ HAI CAVEAT PHẢI XỬ LÝ TRƯỚC KHI ĐƯA VÀO BẢNG:**

1. **Over-refusal cao.** Table 4: ở p=0.5 (setting chính thức) over-refusal = **28.5%**. Paper tự thừa nhận *"over-refusal rate does not lead to significant improvements"*. Ablation cho thấy đây là **đánh đổi có cấu trúc**, không phải lỗi cài đặt:

   | p | GCG ASR ↓ | Over-refusal ↓ |
   |---|---|---|
   | 0 | 50.0 | 20.0 |
   | 0.3 | 22.0 | 23.5 |
   | **0.5** | **2.0** | **28.5** |
   | 0.7 | 2.0 | 40.5 |
   | 1.0 | 0.0 | 49.5 |

   → Trên trục XSTest của mình, DeepRefusal sẽ rơi vào vùng xấu ngang **SAGE (34.8%)**. Đây thực ra là **kết quả hay cho survey** — minh hoạ rõ trade-off ASR ↔ over-refusal, và có sẵn bảng quét p để vẽ đường cong.

2. **Checkpoint public bị nhiễm XSTest.** Họ train trên **500 sample XSTest + Or-bench**. Nếu mình dùng ckpt sẵn rồi đo over-refusal bằng `data/xstest_safeprompts.csv` thì **rất có thể trùng data train** → số over-refusal không sạch. Ba lựa chọn:
   - (a) dùng ckpt, ghi rõ nguy cơ nhiễm trong survey (nhanh nhất);
   - (b) tự train và **loại XSTest khỏi tập train**, chỉ giữ Or-bench cho phần chống over-refusal (sạch, nhưng over-refusal sẽ còn tệ hơn 28.5%);
   - (c) đo over-refusal bằng **Or-bench** thay vì XSTest cho riêng bài này (phá vỡ tính đồng nhất của bảng).
   → Khuyến nghị **(a)** cho vòng đầu, ghi chú rõ; nếu survey cần chặt thì làm thêm (b).

**Điểm cộng cuối:** dùng chung eval harness với **Circuit Breakers** (cùng dựa trên repo CircuitBreaker, và CircuitBreaker chính là baseline mạnh nhất trong bảng của họ) → làm 2 bài một công. Ngoài ra **CAT** (bài #2) cũng là baseline trong paper này → 3 bài chia sẻ chung hạ tầng.
⚠️ Ckpt là **Llama-3**, target chuẩn của mình là **Llama-3.1** → phải chạy thêm `no_defense` trên **đúng Llama-3-8B-Instruct** làm mốc riêng, nếu không so sánh sẽ khập khiễng.

---

### 2. C-AdvUL / C-AdvIPO — Efficient Adversarial Training in LLMs with Continuous Attacks — NeurIPS 2024 Spotlight

**Paper:** https://arxiv.org/abs/2405.15589 · **Repo:** https://github.com/sophie-xhonneux/continuous-advtrain
Tác giả: Sophie Xhonneux, Alessandro Sordoni, Stephan Günnemann, Gauthier Gidel, Leo Schwinn.

**Phân loại:** INTRA chuẩn — adversarial training, đổi weight vĩnh viễn.

**Ý tưởng:** adversarial training cho LLM xưa nay đắt vì phải tìm **chuỗi token tấn công rời rạc** (GCG cần hàng nghìn forward). Bài này tính adversarial attack thẳng trong **không gian embedding liên tục** → rẻ hơn hàng bậc độ lớn, vì gradient chạy trực tiếp không cần search rời rạc.

**Hai thuật toán:**
- **CAT (C-AdvUL)** — 2 loss: (a) *toward/away* (Unlikelihood) robust trước continuous embedding attack tính trên AT dataset, (b) loss trên **utility data** để giữ năng lực. Có cutoff value chặn việc tối ưu quá đà một trong hai vế.
- **CAPO (C-AdvIPO)** — biến thể adversarial của **IPO** (dùng IPO thay DPO vì ít overfit hơn). Loss này ngầm minimize KL với model gốc → chặn hiện tượng model suy sụp thành "từ chối tất cả", nhờ vậy **bỏ được utility dataset**.
- Chi tiết cài đặt đáng chú ý: **không** áp attack `δ` lên input của reference model — thử rồi thấy train mất ổn định.

**Số LLM:** 1.

**Model gốc (paper):** **Gemma 2B, Phi-3-Mini 3.8B, Mistral-7B, Zephyr-7B, Llama2-7B** — đều bản instruction-tuned. Có thêm **Zephyr+R2D2** làm mốc so sánh.
**Attack đánh giá:** **GCG, AutoDAN, PAIR** + Adaptive Attacks + ICL. **ASR chấm bằng chính classifier của HarmBench** — trùng đúng scorer của mình.
**Utility:** MMLU (100 câu/category), ARC-E, ARC-C, MT-Bench. Thêm **HARMLESS** = 40 câu vô hại tự viết để đo over-refusal.

### Repo cung cấp những gì

Cấu trúc: `config/` · `data/` · `src/` · `requirements.txt` · MIT license.

| Thứ | Có? | Chi tiết |
|---|---|---|
| Code train | ✅ | `src/run_experiments.py`, config bằng **Hydra** |
| Lệnh chạy | ✅ | `python src/run_experiments.py --config-name=adv_train_ul path=example_path` → đổi `adv_train_ul` ↔ `adv_train_ipo` để chuyển CAT/CAPO; override hyperparam qua CLI kiểu `adversarial.eps=0.075` |
| Config | ⚠️ | phải tự tạo file trong `config/path` (có `example_path.yaml` mẫu); các giá trị khác "xem trong paper" |
| Data | ✅ | thư mục `data/`, nguồn gốc từ **HarmBench repo** + file bổ sung của paper |
| Checkpoint | ✅✅ | xem bảng dưới |
| Code eval | ❌ | README không nói; phải tự nối vào pipeline của mình (mà mình có sẵn rồi) |

**Checkpoint trên HF (org `ContinuousAT`):**
| Model | Link |
|---|---|
| Phi-CAT | `ContinuousAT/Phi-CAT` |
| Phi-CAPO | `ContinuousAT/Phi-CAPO` |
| Zephyr-CAT | `ContinuousAT/Zephyr-CAT` (card ghi rõ: **LoRA weights** của zephyr-7b-beta) |
| Llama-2-7B-CAT | `ContinuousAT/Llama-2-7B-CAT` |
| **Llama3-8B-IT-CAT** | **`ContinuousAT/Llama3-8B-IT-CAT`** — 8B params, safetensors F32 → có vẻ là **full weight đã merge**, README trống |

⭐ **`Llama3-8B-IT-CAT` là phát hiện quan trọng**: Llama-3-8B-Instruct **không nằm trong paper** (paper chỉ có 5 model kia) nhưng tác giả vẫn release ckpt cho nó → khớp đúng base mà DeepRefusal / Circuit Breakers / DeRTa đang dùng. Xem §"Chốt base model" bên dưới.

### Train — RẺ NHẤT trong cả nhóm INTRA

| Hạng mục | CAT | CAPO |
|---|---|---|
| Iterations | 780 | 360 |
| Batch size | 64 | 64 |
| Thời gian/step (1×A100, Mistral) | **3.2 giây** | 3.2 giây |
| **Tổng ước tính** | **~42 phút** | **~19 phút** |
| So với R2D2 | R2D2 mất **1567.8 s/step** → CAT rẻ hơn **299×** toàn bộ quá trình train | |

Cấu hình: **LoRA trên toàn bộ linear layer** (không full fine-tune) + **quantize 4-bit** → memory footprint rất nhẹ. Perturbation dùng chuẩn ℓ2, **10 attack iteration** mỗi step, ε đặt tương đối theo magnitude trung bình của token embedding: ε=0.1 (Gemma, Phi-3-Mini), ε=0.05 (Mistral-7B, Llama-7B), ε=0.075 (Zephyr-7B).

**Hardware:** cluster nội bộ gồm V100 / **A100 40GB** / A100 80GB → **40GB là đủ, đã được chứng minh**.
⚠️ Con số "**≥1904 GPU hours**" trong paper là **tổng TẤT CẢ thí nghiệm** (5 model × train + chạy GCG/AutoDAN/PAIR để đánh giá — attack mới là thứ ngốn GPU). **Một lần train CAT chỉ ~42 phút.** Pipeline của mình **không chạy GCG/AutoDAN/PAIR** (dùng thẳng prompt HarmBench) nên né hoàn toàn phần đắt đó.

**Data (đều public):**
| Nguồn | Vai trò |
|---|---|
| **AT dataset của HarmBench** | harmful behaviors; safe answer `y` luôn cố định = `"Sorry, I can't do that."` |
| **UltraChat200k** | utility data cho CAT (CAPO không cần) |
| HarmBench test set (40 sample đầu) | eval robustness của họ (giới hạn vì GCG quá đắt) |

**Độ khó:**
- **🟢 RẤT THẤP nếu dùng ckpt** `Llama3-8B-IT-CAT` — tải về, chạy `method.py response` y như no_defense.
- **🟢 THẤP nếu tự train** — đây là bài **duy nhất trong nhóm INTRA mà mình train lại được thoải mái**: LoRA + 4-bit, 40GB đã đủ theo chính paper, ~42 phút. Muốn train CAPO thì càng rẻ (19 phút) và **khỏi cần utility data**.

**⚠️ Caveat:** safe answer khi train **cố định một câu** `"Sorry, I can't do that."` → model sau train có xu hướng từ chối bằng đúng template đó, cụt lủn. Ảnh hưởng tới **JustEval** (điểm engagement/depth) chứ không ảnh hưởng ASR. Nhớ ghi nhận khi đọc kết quả utility.
**✅ Điểm sạch hơn DeepRefusal:** over-refusal họ đo bằng bộ HARMLESS tự viết, **không đụng XSTest** → dùng ckpt của họ rồi đo XSTest của mình thì **không bị nhiễm data train**.

**Điểm cộng cho survey:** đây là **baseline adversarial training được trích dẫn nhiều nhất hiện nay** — ReFAT lấy CAT làm mốc chi phí, và **DeepRefusal cũng lấy CAT làm baseline** trong Table 1. Có nó thì bảng INTRA có sức nặng và đối chiếu chéo được.

---

### 3. Safe RLHF — ICLR 2024 Spotlight

**Paper:** https://arxiv.org/pdf/2310.12773 · **Repo:** https://github.com/PKU-Alignment/safe-rlhf
*(Bài này **đã có** trong `PHUONG_PHAP_MOI.md` mục intra — nhóm tìm lại trùng, xác nhận là lựa chọn đúng.)*

**Phân loại:** INTRA chuẩn — RLHF có ràng buộc, đổi weight vĩnh viễn.

**Ý tưởng:** tách hẳn **helpfulness** và **harmlessness** thành 2 model chấm riêng — **reward model** (hữu ích) và **cost model** (có hại) — rồi tối ưu bằng **PPO-Lagrangian**: cực đại reward **với ràng buộc** cost dưới ngưỡng. Nhờ tách đôi nên tránh được cảnh annotator phải đánh đổi 2 tiêu chí trong cùng một nhãn.

**Flow chạy (3 giai đoạn):** SFT → train reward model + cost model từ output SFT → Safe-RLHF (PPO-Lagrangian) ghép cả ba.

**Số LLM:** 4 trong lúc train (actor + critic + reward + cost); **1 lúc deploy**.

**Model gốc:** LLaMA / OPT / Baichuan / InternLM.

**Data:** **PKU-SafeRLHF** — public trên HF, từ bản 10K tới bản đầy đủ **1M cặp preference** gán nhãn người cho cả helpful lẫn harmless. Đây là một trong những dataset safety preference lớn nhất đang có.

**Checkpoint:** ✅ **Beaver-7B v1/v2/v3** + reward model + cost model trên HuggingFace.

**Hardware:** tác giả dùng **8 × A800-80GB** cho LLaMA-7B (có DeepSpeed ZeRO-Offload để giảm).

**Độ khó — 🟢 THẤP nếu chỉ chạy `beaver-7b` inference** / 🔴 **CAO nếu train** (mình không có 8×80GB).
⚠️ Lưu ý so sánh: Beaver-7B dựa trên **LLaMA-1/Alpaca-7B**, **khác họ** với Llama-3.1-8B-Instruct → cần `no_defense` riêng cho base tương ứng, và phải nói rõ trong survey là "khác base model, chỉ so tương đối". Đây là điểm yếu của Safe RLHF so với DeepRefusal.

---

### 4. ReFAT — Robust LLM Safeguarding via Refusal Feature Adversarial Training

**Paper:** https://arxiv.org/abs/2409.20089 · **OpenReview:** https://openreview.net/forum?id=s5orchdb33 · **Repo:** ❌ không tìm thấy repo public
⚠️ **Venue chưa xác nhận** — chỉ thấy arXiv + OpenReview forum, chưa xác nhận được là đã accept ở đâu. Cần check lại trước khi ghi vào bảng.

**Phân loại:** INTRA chuẩn.

**Ý tưởng (rất gọn và hay):** quan sát rằng **mọi** adversarial attack đều đi qua **một cơ chế chung** — làm mất (ablate) một chiều trong residual stream gọi là **refusal feature**. Vậy thay vì đi tìm chuỗi tấn công tệ nhất (đắt), cứ **mô phỏng thẳng hiệu ứng cuối cùng**: xoá refusal feature với xác suất p qua các layer trong lúc forward, rồi SFT để model **vẫn từ chối** dù đã bị xoá.

**Flow train:** input = cặp (harmful request, refusal answer); minimize NLL của câu trả lời an toàn **trong điều kiện refusal feature bị ablate**; song song train trên utility dataset (harmless request-answer) để giữ năng lực. Pseudo-code ở Algorithm 1 của paper.

**Chi phí — điểm bán hàng chính:** rẻ hơn **R2D2 ~1700×** và **CAT ~10×** tính theo tổng số forward/backward pass.

**Số LLM:** 1. **Model gốc:** 3 LLM phổ biến (paper).

**Độ khó — 🟡 TRUNG BÌNH.**
- Không có code → phải tự viết. Nhưng thuật toán **ngắn**: lấy refusal direction (dùng lại repo Refusal Direction, giống DeepRefusal) + hook ablate + SFT thường. Dễ hơn DeAL bên nhóm IN nhiều.
- Vẫn phải **train thật** → vướng ràng buộc 40GB, phải LoRA.
- **Quan hệ họ hàng:** ReFAT, DeepRefusal, Circuit Breakers đều xoay quanh *refusal direction / representation*. Nếu đã làm DeepRefusal + Circuit Breakers thì ReFAT là **bài thứ ba cùng một hướng** → giá trị biên thấp. Chỉ làm nếu muốn survey có mục "so sánh nội bộ nhóm refusal-direction".

---

### 5. Booster — ICLR 2025 Oral

**Paper:** https://arxiv.org/abs/2409.01586 · **Repo:** https://github.com/git-disl/Booster

**Phân loại cơ chế:** INTRA chuẩn.
**⚠️ Nhưng threat model KHÁC:** Booster chống **harmful fine-tuning attack** — kẻ tấn công **cầm được model đã align và fine-tune lại nó** bằng data độc (ví dụ qua API fine-tuning). Đây **không phải** prompt jailbreak. Đúng nhóm với **TAR** trong `PHUONG_PHAP_MOI.md` (đã đánh dấu ⚠️ cùng lý do).

**Ý tưởng:** ngay ở **giai đoạn alignment**, thêm một regularizer **mô phỏng trước** nhiễu loạn độc hại sẽ xảy ra khi bị fine-tune, và ép mức giảm harmful loss sau nhiễu loạn đó phải **nhỏ đi** → model "cứng đầu" hơn trước fine-tuning độc.

**Flow train:** mỗi step tính **3 gradient** — loss trên harmful dataset, loss harmful **sau khi perturb**, và alignment loss — rồi gộp theo 2 siêu tham số α, λ. Cài bằng `BoosterAlignmentTrainer` kế thừa HF Trainer.

**Model gốc:** **Llama2-7B** (gated repo, phải xin quyền Meta). **Data:** alignment dataset (BeaverTails có refusal) + harmful dataset **BeaverTails** (PKU-Alignment); task downstream để đo utility: SST2, GSM8K, AG News (có script chuẩn bị sẵn).

**Checkpoint:** ❌ không có. Tác giả khuyến nghị chạy bằng **Slurm** (`sbatch smooth_align.sh`, `sbatch smooth_poison_ratio.sh 0.1`).

**Độ khó — 🔴 CAO.** Bắt buộc tự train, 3 gradient/step trên 7B, không ckpt, hạ tầng thiết kế cho cluster Slurm.

**Đề xuất:** **không đưa vào bảng chính**. Nếu muốn giữ, mở một mục riêng *"defense chống harmful fine-tuning"* gồm **Booster + TAR** và ghi rõ metric riêng (harmful score sau khi bị fine-tune với poison ratio p), vì HarmBench ASR trên prompt không đo được thứ nó bảo vệ.

---

## NHÓM B — đúng INTRA, lệch chủ đề (tóm tắt)

Cả 8 bài đều **có sửa weight**, nên về cơ chế thì hợp lệ. Vấn đề nằm ở **cửa 2**: không bài nào có threat model jailbreak, nên không điền được cột ASR.

| Paper | Cơ chế | Base model | Data | Vì sao không dùng được |
|---|---|---|---|---|
| **Quark** (NeurIPS 2022) | RL "unlearning": phân đoạn theo reward quantile rồi conditioned-training để đẩy model xa vùng độc | **GPT-2** | RealToxicityPrompts, reward = **PerspectiveAPI** (cần API key) | Detox continuation, không có prompt tấn công. Model quá cũ. Repo chỉ 5 commit, task khác nằm ở branch `sentiment` / `repetition` |
| **DAPT Detox** (NeurIPS 2022, NVIDIA) | Domain-adaptive pretraining trên corpus tự sinh + adapter-only | 126M → **530B** | Corpus tự sinh (self-generated) | Chạy trên **Megatron-LM**, sai hoàn toàn stack HF của mình. Chủ đề detox |
| **Fine-Grained RLHF** (NeurIPS 2023) | RLHF với reward **dày** (mỗi câu 1 reward) + nhiều reward model theo khía cạnh | **T5-large** (QA), **GPT-2** (detox) | QA-Feedback (có trong repo), RealToxicityPrompts | Base model không phải LLM chat hiện đại; task = QA dài + detox. Cần 80G A100 |
| **DRLC** (EMNLP 2024) | Critic LM sinh feedback → quy ra reward token/span-level cho RL | — | — | **Không tìm thấy code.** Task = sentiment control / detox / summarization |
| **FIGA** (ICLR 2024) | SFT với loss có trọng số theo tín hiệu chất lượng token/phrase, đối chiếu response tốt–xấu | — | **SPA dataset** (HF, public) | Alignment chất lượng tổng quát, **không safety-specific** |
| **InstructGPT** (NeurIPS 2022) | SFT → reward model → PPO (bài khai sinh RLHF) | GPT-3 family | ❌ **labeler data private** | **Không code, không data, không phải defense.** Chỉ nên trích ở phần background của survey |
| **Okapi** (EMNLP 2023 Demo) | RLHF đa ngôn ngữ | BLOOM / LLaMA | Instruction + response-ranked data **26 ngôn ngữ** (public) | Mục tiêu là instruction-following đa ngữ, không phải an toàn. Có thể hữu ích nếu sau này mở rộng XSTest sang tiếng Việt |
| **Self-Criticism** (EMNLP 2023 Industry) | Model tự sinh critique theo tiêu chí HHH → lọc → fine-tune trên chính data đó | — | — | **Không tìm thấy code.** Industry track. Ý tưởng gần với Self-Refine (đã làm, nhóm POST) nhưng có thêm bước fine-tune |

---

## NHÓM C — loại (tóm tắt lý do)

| Paper | Threat model thật sự | Đối tượng | Kết luận |
|---|---|---|---|
| **ATM** (EMNLP 2024) | Tài liệu **nhiễu / bịa đặt** trong RAG | Generator của hệ RAG | Đo bằng QA (F1/EM). Không code. → loại |
| **RAAT** (ACL 2024) | Retrieval noise (3 loại) | LLaMA-2 7B trong RAG | Có code+data ([calubkk/RAAT](https://github.com/calubkk/RAAT)) nhưng metric là F1/EM → loại |
| **Impact of AdvTrain…** (Findings ACL 2023) | — (bài **phân tích**) | Encoder LM | So sánh input-space vs embedding-space perturbation. Không đề xuất defense mới → loại |
| **FLAT** (AAAI 2022) | Tấn công **thay từ đồng nghĩa** lên text classifier | LSTM / CNN / BERT / DeBERTa | Không phải LLM sinh. Có code tốt nhưng sai đối tượng → loại |
| **Veiled Toxicity** (EMNLP 2020) | Toxic speech **ẩn ý** né detector | Detector BERT-era (pytorch 1.3.1) | Củng cố detector, không phải model sinh → loại |
| **Moderate-fitting** (NeurIPS 2022) | **Backdoor / data poisoning** lúc fine-tune | PLM classification | Defense bằng cách giảm capacity/epoch/lr. Khác trục hoàn toàn → loại |

*Ghi chú:* 3 bài **ATM / RAAT** (RAG) và **Moderate-fitting** (backdoor) tuy loại khỏi bảng nhưng đáng nhắc **một dòng** ở phần "phạm vi survey" để chứng minh mình đã rà qua các threat model lân cận và **chủ động giới hạn phạm vi ở prompt-level jailbreak** — cái này làm survey chặt hơn, không phải bỏ phí công tìm.

---

## 🎯 CHỐT BASE MODEL LOCAL: nên chuyển sang `Meta-Llama-3-8B-Instruct`

Quy ước hiện tại (`02_QUY_UOC_MODEL.md`): nhóm local-train dùng **`Llama-3.1-8B-Instruct`**. **Đề xuất đổi sang `Meta-Llama-3-8B-Instruct` (bản 3.0)**. Lý do rất cụ thể — **toàn bộ checkpoint INTRA có sẵn đều nằm trên đúng base này**:

| Method | Checkpoint HF | Base |
|---|---|---|
| DeepRefusal | `skysys00/Meta-Llama-3-8B-Instruct-DeepRefusal` | Llama-3-8B-Instruct |
| **CAT** | `ContinuousAT/Llama3-8B-IT-CAT` | Llama-3-8B-Instruct |
| Circuit Breakers | `GraySwanAI/Llama-3-8B-Instruct-RR` | Llama-3-8B-Instruct |
| DeRTa | `Youliang/llama3-8b-instruct-lora-derta-100step` (LoRA) | Llama-3-8B-Instruct |
| Targeted LAT | `LLM-LAT/robust-llama3-8b-instruct` *(cần verify khi tải)* | Llama-3-8B-Instruct |

Ép sang 3.1 thì **mất sạch 5 checkpoint này** và phải tự train lại tất cả — trong khi giữ 3.0 thì **chạy được cả 5 mà không train một dòng nào**, tất cả cùng một base, chỉ khác đúng cái defense. Đây chính là điều kiện lý tưởng cho một bảng survey.

**Việc phải làm khi đổi:** chạy thêm `no_defense` trên **`Meta-Llama-3-8B-Instruct`** làm mốc riêng cho nhóm local. Bảng API (Groq `llama-3.1-8b-instant`) giữ nguyên; hai bảng bắc cầu qua no_defense như thiết kế cũ, chỉ đổi con local từ 3.1 → 3.0.

**Phần thưởng thêm:** DeepRefusal Table 1 đã công bố ASR trên **đúng Llama3-8B-instruct** cho RT, RT-Augmented, **LAT**, **CAT**, **CircuitBreaker**, DeepRefusal → mình có **bảng đối chiếu công bố sẵn** để kiểm tra pipeline của mình chạy đúng hay sai. Cực kỳ giá trị cho phần validation của survey.

---

## 🎯 CHỐT 5 METHOD CHO NHÓM INTRA

Survey cần 5 pre / 5 post / 5 in / 5 intra. Đây là 5 bài đề xuất cho INTRA — **tất cả cùng base `Meta-Llama-3-8B-Instruct`, tất cả có checkpoint, không phải train gì**:

| # | Method | Venue | Cơ chế (để bảng có độ đa dạng) | Checkpoint | Train nếu muốn |
|---|---|---|---|---|---|
| 1 | **CAT / CAPO** | NeurIPS'24 Spotlight | Adversarial training trong **embedding space liên tục** | `ContinuousAT/Llama3-8B-IT-CAT` | ✅ LoRA+4bit, **~42 phút / 40GB** |
| 2 | **DeepRefusal** | Findings EMNLP'25 | Ablate **refusal direction** theo layer+token depth lúc fine-tune | `skysys00/...-DeepRefusal` | ✅ LoRA, ~45 phút / A100 80GB |
| 3 | **Circuit Breakers** | NeurIPS'24 | **RepE loss** — bẻ gãy biểu diễn dẫn tới output hại | `GraySwanAI/Llama-3-8B-Instruct-RR` | ✅ LoRA |
| 4 | **DeRTa** | ACL'25 | **SFT + RTO** — dạy model từ chối *giữa chừng* khi đã lỡ sinh | `Youliang/llama3-8b-instruct-lora-derta-100step` | ✅ LoRA |
| 5 | **Targeted LAT** | 2024 | **Latent adversarial training** — perturb ở không gian ẩn | `LLM-LAT/robust-llama3-8b-instruct` | ✅ (~36× rẻ hơn R2D2) |

**Vì sao 5 bài này chứ không phải bài khác:**
- **Safe RLHF** → base Beaver-7B là **LLaMA-1/Alpaca**, khác họ, phá vỡ tính đồng nhất của bảng. Giữ làm dự phòng.
- **SecAlign** (đang trong bảng #20) → threat model là **prompt injection**, không phải jailbreak. Nên chuyển sang mục riêng hoặc thay.
- **Booster / TAR** → threat model **harmful fine-tuning**, tách mục riêng.
- **ReFAT** → không có code, và trùng hướng refusal-direction với DeepRefusal + Circuit Breakers.
- **R2D2** → có code trong HarmBench nhưng **không có ckpt Llama-3**, và train cực đắt (GCG mỗi step). Nó xuất hiện gián tiếp rồi: CAT lấy R2D2 làm mốc chi phí, DeepRefusal lấy LAT/CAT/CircuitBreaker làm baseline.

⚠️ **Cần sửa lại một ghi chú cũ:** `PHUONG_PHAP_MOI.md` ghi LAT là *"❌ không ckpt, khó — fork kiến trúc Llama-2"*. Ghi chú đó trỏ vào repo cũ `thestephencasper/latent_adversarial_training`. Bản **Targeted LAT** (Sheshadri et al., arXiv 2407.15549) là repo khác — `aengusl/latent-adversarial-training` — và org HF **`LLM-LAT`** có sẵn `robust-llama3-8b-instruct` cùng `llama3-8b-instruct-rt-jailbreak-robust2/3`. Verify lại khi tải.

---

## Kiến nghị thứ tự triển khai

**Thứ tự chạy đề xuất: CAT → DeepRefusal → Circuit Breakers → DeRTa → Targeted LAT.**
Bắt đầu bằng **CAT** vì nó vừa có ckpt vừa **train lại được trong 42 phút trên đúng 40GB mình có** → dùng nó để dựng + kiểm thử toàn bộ đường ống INTRA (tải ckpt → generate → judge → cost), sau đó 4 bài còn lại chỉ là đổi tên model.

| Ưu tiên | Method | Cách chạy | Lý do |
|---|---|---|---|
| 1 | **DeepRefusal** | ckpt `Meta-Llama-3-8B-Instruct-DeepRefusal`, inference-only. Tự train được nếu cần: **LoRA 45 phút / A100 80GB**, data public hết | Ckpt **Llama-3-8B** khớp nhất với target local; vừa MIG 40GB; dùng chung harness với Circuit Breakers + CAT. ⚠️ over-refusal ~28.5% và ckpt có thể nhiễm XSTest — xem 2 caveat ở §1 |
| 2 | **C-AdvUL / C-AdvIPO** | ckpt Phi-CAT / Zephyr-CAT / Llama variants, inference-only | Baseline adversarial training chuẩn mực (NeurIPS Spotlight), data gốc từ **HarmBench** — trùng nguồn với mình |
| 3 | **Safe RLHF** | ckpt `beaver-7b-v3`, inference-only | Đã nằm trong kế hoạch; dataset PKU-SafeRLHF 1M là tài sản tốt để trích trong survey. Nhớ chạy no_defense trên base tương ứng |
| 4 | **ReFAT** | tự implement | Chỉ làm nếu muốn hoàn thiện cụm *refusal-direction* (cùng DeepRefusal + Circuit Breakers). Không code, giá trị biên thấp |
| 5 | **Booster** | tự train | Tách ra mục riêng "chống harmful fine-tuning" cùng TAR, metric riêng. Không nhét vào bảng ASR chung |
| — | 14 bài còn lại | — | Related work / phần giới hạn phạm vi |

**Tác động lên `BANG_PHUONG_PHAP.md`:** nhóm INTRA đang **chỉ có 1 bài** (SecAlign) — đây là chỗ mỏng nhất của survey. Nhận 3 bài nhóm A (DeepRefusal, C-AdvUL/IPO, Safe RLHF) + 2 bài đã lên kế hoạch từ trước (**Circuit Breakers**, **DeRTa**) → INTRA lên **6 bài**, cân bằng hẳn với pre (10) và post (6). Và vì cả 5 bài đều **có checkpoint sẵn**, toàn bộ nhóm chạy được trên 1 MIG 40GB mà **không cần train một dòng nào**.
