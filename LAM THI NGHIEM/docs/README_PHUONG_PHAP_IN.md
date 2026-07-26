# Kiểm tra 7 paper nhóm tự tìm cho nhóm IN

Ngày check: 26/07/2026. Nguồn: paper gốc + đọc trực tiếp README/code các repo.

Danh sách gửi lên có 8 dòng nhưng **SafeDecoding bị lặp 2 lần** → thực tế **7 paper**.

Nhắc lại định nghĩa IN của mình (`CLAUDE.md` §2):
> **in-processing** = can thiệp lúc sinh (decoding), thay đổi **TẠM THỜI** (logits / xác suất token / activation), hết request thì model về nguyên trạng. Không tạo checkpoint mới.
> Ranh giới in vs intra = **tạm thời vs vĩnh viễn**. Ranh giới in vs pre/post = có đụng vào **nội tại model lúc sinh** hay chỉ đụng text vào/ra.

---

## Kết luận nhanh

| # | Paper | Venue · Năm | Đúng IN? | GitHub | Code thật? | Chủ đề hợp jailbreak? | Độ khó chạy lại | Đề xuất |
|---|---|---|---|---|---|---|---|---|
| 1 | **SafeDecoding** | ACL 2024 (long) | ✅ **Đúng IN** (sửa logits) | [uw-nsl/SafeDecoding](https://github.com/uw-nsl/SafeDecoding) | ✅ code + **LoRA expert sẵn** | ✅ | 🟢 **Dễ nhất nhóm IN** | **LÀM** — đã có trong bảng (#17) |
| 2 | **JBShield** | USENIX Security 2025 | ✅ **Đúng IN** (sửa hidden repr lúc infer) | [NISPLab/JBShield](https://github.com/NISPLab/JBShield) | ✅ code + **data sẵn trong repo** | ✅✅ | 🟡 Trung bình (cần 2×24GB) | **LÀM** — bài mới, mạnh |
| 3 | **DeAL** | ACL 2025 (long) | ✅ **Đúng IN** (A\* search lúc decode) | ❌ **không có repo chính thức** | ❌ tự viết từ paper | ✅ (có mục harmlessness) | 🔴 Khó (tự implement + rất chậm) | Cân nhắc / để sau |
| 4 | **TaskTracker** (Get My Drift?) | IEEE SaTML 2025 | ⚠️ **Ranh giới** — chỉ **ĐỌC** activation, không sửa → thực chất là detector | [microsoft/TaskTracker](https://github.com/microsoft/TaskTracker) | ✅ code + **probe đã train sẵn** | ⚠️ **task drift / prompt injection**, KHÔNG phải jailbreak | 🔴 Khó (activation data phải xin form) | Giữ ở IN nhưng ghi rõ sub-type; ưu tiên thấp |
| 5 | **SelfDefend** | USENIX Security 2025 | ❌ **KHÔNG phải IN** → là **PRE** (detector-gate) | [selfdefend/Code](https://github.com/selfdefend/Code) | ✅ code + **LoRA ckpt + data sẵn** | ✅✅ | 🟢 **Rất dễ** (bản prompt-only chạy thuần API) | **LÀM NGAY, xếp vào PRE** |
| 6 | **Hot or Cold? (AdapT)** | AAAI 2024 | ⚠️ Cơ chế là decoding-time, nhưng… | ❌ không tìm thấy repo | ❌ | ❌ **lạc chủ đề** — sinh CODE, đo pass@k | — | **LOẠI** (đã từng bị loại 1 lần) |
| 7 | **Hybrid UQ for Selective Text Classification** | ACL 2023 | ❌ **KHÔNG phải IN**, cũng không phải defense LLM | [AIRI-Institute/hybrid_uncertainty_estimation](https://github.com/AIRI-Institute/hybrid_uncertainty_estimation) | ✅ code đầy đủ | ❌ **lạc chủ đề** — UQ cho classifier BERT-like | — | **LOẠI** (chỉ dùng làm related work) |

**Tóm tắt:** trong 7 bài, chỉ **3 bài đúng nghĩa IN + đúng chủ đề jailbreak** (SafeDecoding, JBShield, DeAL). SelfDefend là bài tốt nhưng **phải xếp sang PRE**. TaskTracker nằm ở ranh giới và lệch chủ đề. 2 bài còn lại nên loại.

---

## 1. SafeDecoding — ACL 2024

**Paper:** https://aclanthology.org/2024.acl-long.303.pdf · **Repo:** https://github.com/uw-nsl/SafeDecoding

**Phân loại:** IN chuẩn — trộn phân phối xác suất token ở mỗi bước decode, không đổi trọng số target.

**Flow chạy:**
1. *Offline (1 lần / model):* LoRA fine-tune chính target model → **expert model** chuyên từ chối.
2. *Inference, mỗi bước decode:*
   - lấy **top-k** token của base model và của expert model → **giao nhau** = sample space
   - phân phối mới: `P_n(x) = p_θ(x) + α · (p_expert(x) − p_θ(x))`, mặc định **α = 3**
   - chỉ áp cho **m = 2 token đầu tiên**, sau đó decode greedy bình thường (vì "safety disclaimer" nằm ở đầu response)

**Số LLM:** 2 forward mỗi bước (base + expert), nhưng expert là **LoRA adapter trên chính base** → VRAM ≈ 1 model + adapter, không phải 2 model rời.

**Model gốc trong paper:** 5 model 7B — Vicuna, Llama-2-chat, Guanaco, Falcon, Dolphin. Đánh 6 attack, 4 benchmark, so với 6 defense.

**Train cái gì:** LoRA cho expert. Config: **r=16, alpha=64, 2 epoch, batch 1, lr 2e-3, max len 2048** — tác giả nói train **dưới 1 phút/model**.

**Data:** cực nhỏ — **36 harmful query** thuộc 18 category lấy từ Ganguli et al. 2022 (Anthropic red-team), mỗi query sinh 2 response từ chối bằng chính model (top-p 0.9, temp 0.7), GPT-4 verify → **≤72 cặp**. Repo **có sẵn LoRA expert** cho vicuna / llama2 / guanaco / falcon / dolphin trong `lora_modules/` → khỏi train nếu dùng đúng 5 model đó.

**Overhead:** ATGR **1.03× (Llama-2)** – **1.07× (Vicuna)** → gần như miễn phí. Đây là con số đẹp để đưa vào cột cost.

**Độ khó chạy lại — 🟢 THẤP.**
- Chạy nguyên repo với llama2-7b: gần như plug-and-play, `python defense.py --model_name ... --attacker ... --defender ...`.
- Chạy trên target của mình (`Llama-3.1-8B-Instruct` local): **không có LoRA sẵn** → phải tự sinh 72 cặp refusal + train LoRA. Nhưng data chỉ 72 sample, train <1 phút → rẻ. Đây là phần "tự làm" duy nhất.
- Lưu ý: repo dùng `--GPT_API` để chấm harmful — mình **bỏ qua**, chấm bằng `metrics/harmbench.py` như các method khác.

---

## 2. JBShield — USENIX Security 2025

**Paper:** https://arxiv.org/pdf/2502.07557 · **Repo:** https://github.com/NISPLab/JBShield

**Phân loại:** IN chuẩn — **JBShield-M** cộng/trừ anchor vector vào hidden representation lúc inference (tạm thời, không đổi weight). **JBShield-D** là detector đọc activation.

**Ý tưởng:** dựa trên Linear Representation Hypothesis. Tách 2 khái niệm trong hidden space:
- **toxic concept** — ngữ nghĩa độc hại (xuất hiện ở CẢ harmful prompt lẫn jailbreak prompt)
- **jailbreak concept** — thứ khiến model chuyển từ *từ chối* → *tuân theo* (chỉ xuất hiện ở jailbreak prompt)

**Flow chạy:**
1. *Calibration (1 lần / model):* trích concept token bằng logit-lens trên calibration set → chọn **critical layer** + ngưỡng.
2. *Detection (JBShield-D):* prompt kích hoạt **cả hai** concept → gắn cờ jailbreak (accuracy trung bình **0.95**).
3. *Mitigation (JBShield-M):* trong lúc sinh, **cộng** anchor vector của toxic subspace và **trừ** anchor vector của jailbreak subspace vào hidden repr → ASR trung bình từ **61% → 2%**.

**Số LLM:** 1 (chính target) + 1 judge phụ (`ft-mistral-7b-instruct`) chỉ để chấm trong paper — mình thay bằng judge của mình.

**Model gốc:** Mistral-7B-Instruct-v0.2, Llama-2-7b-chat-hf, **Meta-Llama-3-8B-Instruct**, vicuna-7b-v1.5, vicuna-13b-v1.5.

**Train cái gì:** **KHÔNG train model nào.** Chỉ concept extraction + chọn layer + set threshold trên calibration set → rất rẻ.

**Data:** **có sẵn trong repo** (`./data/jailbreak/`, đã chia calibration/test): jailbreak prompt của **9 loại attack × 5 LLM**, benign từ **Alpaca**, harmful từ **AdvBench** + **Hex-PHI**.

**Hardware:** tối thiểu **2 GPU ≥24GB**; khuyến nghị 4×RTX4090 hoặc **1×A100 80GB**. Chạy bằng `./interpret.sh`, `./evaluate_detection.sh`, `./evaluate_mitigation.sh`.

**Độ khó chạy lại — 🟡 TRUNG BÌNH.**
- Điểm cộng lớn: code + data đầy đủ, **có sẵn Llama-3-8B-Instruct** → gần khớp target local của mình (`Llama-3.1-8B-Instruct`).
- Điểm trừ: server MIG của mình là **1×40GB** — chạy 7B/8B ổn, **vicuna-13b thì chật** (xem [gpu-server-mig] trong memory: batch ≤ 4).
- ⚠️ **Caveat quan trọng cho pipeline của mình:** JBShield cần **calibration set theo từng loại attack** để học "jailbreak concept". Nhưng `data/harmbench_300.csv` của mình là **harmful prompt thô, không bọc jailbreak template** → jailbreak concept gần như không kích hoạt, method sẽ chỉ còn tác dụng qua toxic concept. Trước khi chạy phải quyết: (a) chấp nhận đo trên prompt thô và ghi rõ, hay (b) bổ sung một tập HarmBench có bọc template tấn công.

---

## 3. DeAL — ACL 2025

**Paper:** https://arxiv.org/pdf/2402.06147 · **ACL:** https://aclanthology.org/2025.acl-long.1274/ · **Repo:** ❌ không có

**Phân loại:** IN chuẩn — can thiệp thẳng vào quá trình search lúc decode.

**Đã kiểm tra kỹ vụ code:** trang ACL Anthology **không có mục Software/code link**, bản arXiv **không nêu URL code**, không tìm thấy repo nào trên `amazon-science` / `amzn`. Tác giả: James Y. Huang (USC) + AWS AI Labs. → **Muốn chạy phải tự implement 100% từ paper.**

**Flow chạy:** coi sinh văn bản là **A\* search**:
- hàm điểm: `c(y_t) = log P(y_1:t | p) + λ · h(y_1:t+l, p)`
- giữ **beam top-k (k = 5–10)**; với mỗi ứng viên, **lookahead l = 32 token** bằng greedy decode để chấm heuristic sớm
- `h(·)` = heuristic:
  - **programmatic** (keyword coverage, độ dài) → rule, miễn phí
  - **abstract** (harmless / helpful) → **reward model tham số**, có thể trộn tuyến tính `λ_harmless·R_harmless + λ_helpful·R_helpful`
- thêm "start-state adaptation": đổi phần alignment prompt `p_a` khi mục tiêu diễn đạt được bằng ngôn ngữ tự nhiên

**Số LLM:** 1 base LLM + 1 reward model nhỏ (**OPT-125M**).

**Model gốc:** Falcon-7B-instruct, MPT-7B-instruct, Dolly-v2-3B. **Data:** CommonGen (keyword), XSUM (độ dài), **HH-RLHF** (train reward model harmless/helpful), **HarmfulQ** (đánh giá out-of-domain harmful).

**Train cái gì:** reward model **OPT-125M** fine-tune trên subset HH-RLHF (harmless-only, helpful-only, và hh gộp). Model 125M → train rất rẻ, data public trên HF.

**Độ khó chạy lại — 🔴 CAO.** Hai lý do:
1. **Không có code** → phải tự viết vòng lặp beam + lookahead + tích hợp reward model. Đây là phần dễ sai và khó đối chiếu với số của paper.
2. **Cực chậm:** mỗi bước decode tốn ~`k × l` forward (5×32 = 160 token lookahead). Sinh 512 token thì chi phí gấp hàng chục–trăm lần no_defense. Chính tác giả thừa nhận "generality makes decoding slower" và để việc tối ưu cho future work. Với 300 prompt HarmBench + 250 XSTest + 800 JustEval trên 1 MIG 40GB thì đây là bài **tốn giờ GPU nhất** trong toàn bộ survey.

**Đề xuất:** để cuối. Nếu chỉ cần 1 đại diện cho hướng "search/reward-guided decoding" thì bài này đúng vai (nó là bài canonical, ACL 2025), nhưng nếu muốn kết quả nhanh thì **ROSE (contrastive decoding, training-free, 2× forward)** hoặc **SafeInfer** trong `PHUONG_PHAP_MOI.md` rẻ hơn nhiều.

---

## 4. TaskTracker (Get My Drift?) — IEEE SaTML 2025

**Paper:** https://arxiv.org/pdf/2406.00799 · **Repo:** https://github.com/microsoft/TaskTracker

**Phân loại — ⚠️ cần bàn:** method **trích activation rồi đưa vào probe** để phát hiện *task drift*. Nó **chỉ đọc** nội tại model, **không sửa** logits/activation, **không đổi** output. Theo định nghĩa của mình ("can thiệp lúc sinh, thay đổi tạm thời") thì nó **không thực sự can thiệp** → đúng bản chất là **detector-gate dùng tín hiệu white-box**.

Hai lựa chọn, chọn 1 rồi ghi rõ trong survey:
- **(a)** giữ ở IN, thêm sub-type *"white-box detector (read-only activation)"* — cùng nhóm với **JBShield-D**. Nhất quán và giải thích được.
- **(b)** chuyển sang PRE (detector-gate), cùng nhóm Perplexity / FJD — vì nó chặn request trước khi trả lời.
→ Khuyến nghị **(a)**, vì nếu chuyển sang PRE thì nhóm IN lại càng mỏng, mà tín hiệu nó dùng (hidden state) khác hẳn các detector black-box ở PRE.

**⚠️ Lệch chủ đề:** threat model là **task drift do prompt injection gián tiếp** (dữ liệu ngoài chèn nhiệm vụ mới), **không phải jailbreak**. HarmBench ASR không đo được trực tiếp — muốn đưa vào bảng chung phải thêm cột/benchmark riêng, hoặc chấp nhận nó là "bài khác trục".

**Flow chạy:** lấy activation của model ở 2 thời điểm (trước/sau khi đọc data block) → **delta** → probe (linear hoặc triplet metric-learning) phán có drift không.

**Số LLM:** 1 (chỉ cần forward, không sinh) + probe siêu nhẹ.

**Model gốc:** Phi-3 3.8B, Mistral 7B, **Llama-3 8B**, Mixtral 8x7B, Llama-3 70B.

**Train cái gì:** probe — `train_linear_model.py` (linear) và `train_per_layer.py` (triplet). **Repo có sẵn probe đã train** trong `trained_linear_probes/` và `trained_triplet_probes/` → dùng ngay được, không cần train.

**Data — cần phân biệt 2 loại, KHÔNG phải "họ giấu data":**

| Loại | Tình trạng |
|---|---|
| **Text data** (500K+ example task-drift) | **Public về nội dung** nhưng không đóng gói sẵn trên HF. Repo có notebook `task_tracker/dataset_creation/recreate_dataset` **tái tạo y hệt bản gốc**, kèm **prompt hash** để verify tái tạo đúng |
| **Activation data** (phần đắt tiền) | Phải điền [form](https://forms.microsoft.com/r/wXBfXQpuR2), tải qua Azure Blob + azcopy/SAS token. README còn ghi *"coming soon, we will reply with links to download once they are available"* → xin cũng chưa chắc có ngay |
| **Probe đã train** | ✅ nằm sẵn trong repo (`trained_linear_probes/`, `trained_triplet_probes/`), không cần xin |

→ Bản chất: activation là thứ **tốn GPU để sinh ra** nên họ share có kiểm soát vì file rất nặng, chứ không phải khoá dataset. Mình **tự làm trọn được**: tái tạo text → forward qua model → tự trích activation. Chỉ tốn giờ GPU (tác giả ghi rõ bước này "computationally expensive").
→ Thêm nữa: activation họ share là của **model họ chạy** (Mistral 7B, Llama-3 8B, Mixtral, Phi-3, Llama-3 70B). Muốn đo trên target riêng thì **có xin được cũng không dùng lại được** — vẫn phải tự trích. Nên với mình cái form đó gần như vô nghĩa.

**Độ khó chạy lại — 🔴 CAO** nếu làm đầy đủ: nút thắt **không phải xin data** mà là **giờ GPU để trích activation** cho 500K example. **🟡 Trung bình** nếu chỉ dùng probe sẵn cho Llama-3 8B trên tập test của họ — nhưng lúc đó là *đo lại bài của họ*, không nhập được vào bảng ASR/over-refusal của mình.

---

## 5. SelfDefend — USENIX Security 2025 → **xếp vào PRE, không phải IN**

**Paper:** https://arxiv.org/pdf/2406.05498 · **Site:** https://selfdefend.github.io/ · **Repo:** https://github.com/selfdefend/Code

> ⚠️ **ĐỪNG NHẦM với `LLM Self Defense`** (#11 nhóm POST, đã chạy xong — ASR 9.7% / over-refusal 35.6%). Hai bài khác hẳn nhau, chỉ trùng tên:
>
> | | **LLM Self Defense** (post, đã chạy) | **SelfDefend** (pre, bài này) |
> |---|---|---|
> | Tác giả · venue | Phute et al., arXiv 2023 | Wang et al., USENIX Security 2025 |
> | Repo | `poloclub/llm-self-defense` | `selfdefend/Code` |
> | **Soi cái gì** | **RESPONSE** (sinh xong mới hỏi "có hại không?") | **PROMPT** (shadow LLM đọc câu hỏi) |
> | Nhóm | **POST** | **PRE** |
> | Trong repo | `methods/post/Self_Defense/` | chưa có |
>
> SelfDefend *giữ output trong cache rồi mới thả* nên nhìn giống post, nhưng thứ quyết định chặn/không là **input** → vẫn là pre.

**Phân loại — ❌ KHÔNG phải IN.** Ý tưởng mượn từ *shadow stack*: dựng thêm một **shadow LLM** chạy **song song** với target. Shadow LLM đọc **prompt** (không đọc logits, không đụng nội tại target) rồi ra tín hiệu checkpoint. Target không hề bị can thiệp lúc sinh → theo taxonomy của mình đây là **PRE, sub-type detector-gate** (có màu post ở chỗ nó giữ output trong cache, nhưng thứ được kiểm tra là *input*).

**Flow chạy:**
1. Query đi **đồng thời** vào target LLM (trả lời bình thường) và defense LLM (chế độ detect).
2. Defense LLM bọc query bằng 1 trong 2 prompt: **P_direct** (phát hiện nội dung có hại trực tiếp) hoặc **P_intent** (phân tích ý định).
3. Target sinh token nhưng **cache lại**, chờ tín hiệu checkpoint.
4. Defense trả **"No"** (an toàn) → thả cache ra; ngược lại → trả template từ chối.

**Số LLM:** 2 (target + shadow). Vì chạy song song nên **latency thêm rất ít** — đây là điểm bán hàng chính của paper.

**Model gốc:** target = GPT-3.5/4, Claude, Llama-2-7b/13b-chat, Mistral-7B-Instruct-v0.2. Defense = **Llama-2-7b + LoRA**.

**Train cái gì:** LoRA cho shadow model, **chưng cất (distill) từ output của bản SelfDefend chạy bằng GPT-4** → bản open-source rẻ mà vẫn mạnh. `fine_tuning.py` cho phép train trên base model khác.

**Data:** trong repo — AlpacaEval, JailbreakHub, JailbreakBench, MultiJail, Anthropic red-team (đã tiền xử lý sẵn để train). **Checkpoint LoRA có sẵn** trong `checkpoint/`: `llama-2-7b-lora-direct` và `llama-2-7b-lora-intent` (base model tự tải riêng).

**Độ khó chạy lại — 🟢 THẤP, và rất hợp pipeline hiện tại:**
- **Bản prompt-only** (defense LLM = LLM chat thường, chỉ dùng P_direct / P_intent): chạy **thuần API Groq**, đúng 2 call → gần như copy y hệt cấu trúc **LLM Self Defense** hoặc **IA** đã làm. Ước **P2**, làm trong 1 buổi. Chỉ cần lấy verbatim 2 prompt từ repo nhét vào `method.py`.
- **Bản tuned**: cần local Llama-2-7b + LoRA có sẵn → P4, nhưng vẫn dễ vì không phải train.
- Repo có `jailbreaking.py` / `evaluate.py` — mình chỉ lấy **prompt + logic quyết định**, phần chạy dữ liệu vẫn dùng `core/runner.py`.

**→ Đây là bài đáng làm sớm nhất trong 7 bài**, chỉ cần chuyển nhóm từ IN sang PRE.

---

## 6. Hot or Cold? Adaptive Temperature Sampling (AdapT) — AAAI 2024 → **LOẠI**

**Paper:** https://arxiv.org/abs/2309.02772 (Yuqi Zhu, Jia Li, Ge Li, et al. — PKU)

- **Cơ chế**: đúng là decoding-time — chia token thành *challenging* (khó đoán) và *confident* (dễ suy ra), dùng **temperature cao** cho token khó để tăng đa dạng, **temperature thấp** cho token dễ để tránh nhiễu đuôi phân phối.
- **Nhưng chủ đề là sinh CODE**, đo bằng pass@k trên benchmark code — **không có yếu tố an toàn/jailbreak nào**. Không có khái niệm ASR hay over-refusal để điền vào bảng.
- **Repo:** không tìm thấy repo chính thức (arXiv/AAAI đều không nêu link).
- Bài này **đã từng bị loại một lần** khỏi `BANG_PHUONG_PHAP.md` với lý do "lạc chủ đề" (xem dòng cập nhật đầu file: *"bỏ AdapT + Constitutional AI (lạc chủ đề)"*).

**→ LOẠI.** Nếu vẫn muốn nhắc, chỉ nên để ở phần related work: "điều khiển temperature theo token là một hướng in-processing khác, nhưng mục tiêu là chất lượng sinh chứ không phải an toàn".

---

## 7. Hybrid Uncertainty Quantification for Selective Text Classification — ACL 2023 → **LOẠI**

**Paper:** https://aclanthology.org/2023.acl-long.652/ · **Repo:** https://github.com/AIRI-Institute/hybrid_uncertainty_estimation (AIRI Institute — Vazhentsev, Kuzmin, Tsvigun, Panchenko, Panov, Burtsev, Shelmanov)

- **Nội dung**: kết hợp **epistemic** + **aleatoric** uncertainty để làm **selective classification** (model được quyền *từ chối trả lời* khi không chắc) trên các task **mơ hồ**: toxicity detection, sentiment analysis, phân loại đa lớp.
- **Không phải IN**: không đụng decoding của LLM sinh văn bản. Đối tượng là **classifier transformer kiểu encoder**, không phải LLM chat.
- **Không phải defense**: không có threat model tấn công, không có jailbreak. "Từ chối" ở đây là *abstention vì không chắc*, khác hẳn *refusal vì có hại*.
- **Code**: repo đầy đủ, có config + script train + notebook toy example — nhưng là pipeline train classifier, không cắm được vào `core/runner.py`.
- Không đo được bằng cả 3 metric của mình (HarmBench / XSTest / JustEval).

**→ LOẠI khỏi bảng method.** Giá trị duy nhất: nếu sau này mình muốn thêm **cơ chế abstention có hiệu chỉnh** cho các detector (Perplexity, FJD, JBShield-D) thì đây là reference tốt cho phần "chọn ngưỡng / trade-off coverage–risk".

---

## 🎯 CHỐT 5 METHOD CHO NHÓM IN

Survey cần 5 pre / 5 post / 5 in / 5 intra. Gộp 7 bài file này với các ứng viên IN trong `PHUONG_PHAP_MOI.md` (đã clone repo verify 26/07/2026), đây là 5 bài đề xuất — **tất cả đều có code thật, chạy local, đúng trục jailbreak, và mỗi bài một cơ chế khác nhau** để bảng có độ phủ:

| # | Method | Venue | Cơ chế con | Code | Ghi chú |
|---|---|---|---|---|---|
| 1 | **SafeDecoding** | ACL'24 | **Trộn logits** với expert model | ✅ `uw-nsl/SafeDecoding` + LoRA expert sẵn | Expert train <1 phút / 72 sample. Overhead 1.03–1.07× |
| 2 | **JBShield** | USENIX Sec'25 | **Sửa hidden representation** (anchor vector) | ✅ `NISPLab/JBShield` + data sẵn | Không train. Có sẵn Llama-3-8B |
| 3 | **ROSE** | Findings ACL'24 | **Contrastive decoding** (training-free) | ✅ `WHU-ZQH/ROSE` | Rẻ, 2× forward, không train gì |
| 4 | **SafeInfer** | AAAI'25 | **Steer activation + logit** | ✅ `NeuralSentinel/SafeInfer` | Cần thêm 1 con `M_unsafe` → 2 model |
| 5 | **DRO** | ICML'24 | **Soft prompt** theo refusal direction | ✅ `chujiezheng/LLM-Safeguard` | Dễ nhất về code; train soft prompt nhỏ |

**Vì sao không chọn 3 bài còn lại của file này:**
- **DeAL** → không có repo chính thức + decoding chậm khủng khiếp (k×l forward mỗi bước). Nếu vẫn muốn một đại diện *search/reward-guided decoding* thì thay **DRO** bằng DeAL — nhưng phải chấp nhận tự viết và tốn nhiều giờ GPU nhất survey.
- **TaskTracker** → lệch chủ đề (task drift / prompt injection), không điền được cột ASR.
- **AdapT**, **Hybrid UQ** → loại, lạc chủ đề.

**Lưu ý phân loại cho DRO:** nó tối ưu một **soft prompt** rồi prepend vào input — xét chặt thì gần **pre (optimized-prompt, cùng nhóm RPO/Prompt-Tuning)** hơn là in. Nhưng nó tối ưu **theo refusal direction trong không gian biểu diễn** nên `PHUONG_PHAP_MOI.md` xếp vào in. Hai lựa chọn: (a) giữ ở in và định nghĩa rõ "in = can thiệp ở tầng biểu diễn/decoding", hoặc (b) thay DRO bằng **Self-CD** (contrastive decoding, ACL'24) cho khỏi tranh cãi — nhưng Self-CD nhắm giảm over-refusal chứ không chống jailbreak, nên bù trục XSTest chứ không bù trục ASR. → Khuyến nghị **(a)**, và nói rõ định nghĩa ở đầu survey.

**Base model:** cả 5 bài đều chạy local white-box. Xem đề xuất chuyển base local sang **`Meta-Llama-3-8B-Instruct`** ở `README_PHUONG_PHAP_INTRA.md` — JBShield có sẵn Llama-3-8B trong repo nên ăn khớp; SafeDecoding thì không có LoRA expert cho Llama-3 (chỉ vicuna/llama2/guanaco/falcon/dolphin) → phải tự train expert, nhưng chỉ mất <1 phút với 72 sample.

**Thứ tự chạy đề xuất:** ROSE (training-free, rẻ nhất → dựng đường ống) → SafeDecoding → JBShield → SafeInfer → DRO.

---

## Kiến nghị thứ tự triển khai

| Ưu tiên | Method | Nhóm | Lý do |
|---|---|---|---|
| 1 | **SelfDefend (bản prompt-only)** | **PRE** (không phải IN) | 2 call API, không GPU, tái dùng khuôn Self_Defense/IA có sẵn. Nhanh ra kết quả nhất. |
| 2 | **SafeDecoding** | IN | Đã nằm trong bảng (#17). Code + LoRA sẵn, expert train <1 phút trên 72 sample, overhead chỉ 1.03–1.07×. |
| 3 | **JBShield** | IN | Bài mới USENIX'25, code + data đầy đủ, không cần train, có sẵn Llama-3-8B. Cần xử lý caveat calibration/attack-type. |
| 4 | **TaskTracker** | IN (sub-type detector) | Chỉ làm nếu muốn giữ trục *prompt injection*. Nút thắt là **giờ GPU trích activation**, không phải data (text tái tạo được, probe có sẵn); và phải đo bằng benchmark riêng. |
| 5 | **DeAL** | IN | Không có code + decoding cực chậm. Chỉ làm khi đã dư thời gian GPU và muốn 1 đại diện search-based. |
| — | AdapT, Hybrid UQ | — | Loại, lạc chủ đề. |

**Tác động lên `BANG_PHUONG_PHAP.md`:** nhóm IN hiện có 3 bài (SafeDecoding, GeDi, TaskTracker). Nếu nhận thêm **JBShield** + **DeAL** thì IN lên **5**; **SelfDefend** đẩy PRE lên 11. Cân đối tổng thể vẫn nghiêng về PRE — vẫn nên bổ sung thêm từ danh sách IN trong `PHUONG_PHAP_MOI.md` (SafeInfer, ROSE, Jailbreak Antidote) vì mấy bài đó rẻ hơn DeAL nhiều.
