# BÁO CÁO THÍ NGHIỆM — Đánh giá phương pháp phòng thủ Jailbreak cho LLM

*Số kết quả chi tiết ở `KET_QUA_FULL_LAN1.md`; chi tiết từng method ở `methods/<nhóm>/<Method>/README.md`.*

---

## 1. Mục tiêu

Khảo sát (survey) + **đánh giá thực nghiệm đồng nhất** các phương pháp phòng thủ LLM trước tấn công jailbreak. Mỗi phương pháp trả lời 4 câu hỏi:

- **Chặn được tấn công không?** → ASR (Attack Success Rate), càng thấp càng tốt.
- **Có từ chối oan prompt vô hại không?** → Over-refusal, càng thấp càng tốt.
- **Áp vào rồi model còn trả lời tử tế không?** → Utility, càng cao càng tốt.
- **Tốn bao nhiêu?** → Cost (train một lần + infer mỗi request).

Điểm mấu chốt: **tất cả method chạy trên CÙNG một target model, CÙNG data, CÙNG judge** → so sánh là so *phương pháp phòng thủ*, không phải so *model*.

## 2. Taxonomy (tự định nghĩa — theo "can thiệp ở đâu trong vòng đời 1 request")

| Nhóm | Can thiệp | Đặc điểm |
|---|---|---|
| **pre** | INPUT (trước khi model xử lý) | lọc/viết lại prompt, thêm safety system prompt, detect-rồi-chặn |
| **post** | OUTPUT (sau khi model sinh xong) | guard đọc response, self-critique, kiểm duyệt |
| **in** | lúc DECODE (logits/activation) — **tạm thời**, hết request là hết | KHÔNG tạo checkpoint mới |
| **intra** | sửa **TRỌNG SỐ vĩnh viễn** | fine-tune an toàn, DPO, adversarial training → tạo model mới |

Ranh giới **in vs intra = tạm thời vs vĩnh viễn**. Đây là điểm quan trọng để hiểu phần cost bên dưới.

## 3. Thiết lập đánh giá (setup)

### 3.1 Ba metric
| Metric | Bộ data | Đo | Chấm ở đâu |
|---|---|---|---|
| **ASR** (HarmBench) | 300 behavior độc hại | tỷ lệ bị jailbreak | classifier `cais/HarmBench-Llama-2-13b-cls` (GPU) |
| **Over-refusal** (XSTest) | 250 prompt AN TOÀN (nhưng nghe "nhạy cảm") | tỷ lệ từ chối oan | LLM judge `openai/gpt-oss-20b` (API) |
| **Utility** (JustEval) | 200 câu hữu ích (subset của 800) | chất lượng trả lời, 5 aspect 1-5 → TB | LLM judge `gpt-oss-20b` (API) |

### 3.2 Quy ước model (BẤT BIẾN cho mọi method)
| Vai trò | Model |
|---|---|
| Target (bài gọi API) | `llama-3.1-8b-instant` (Groq) |
| Target (bài local — in/intra) | **`Meta-Llama-3-8B-Instruct`** (mirror `NousResearch/*`, SHA256 trùng bản gated) |
| Judge (over-refusal + utility) | `openai/gpt-oss-20b` (Groq) |
| Classifier ASR | `HarmBench-Llama-2-13b-cls` |

→ **2 bảng kết quả tách riêng** (API vs local) vì base khác nhau (Groq Llama-3.1 quantized vs local Llama-3.0 fp16), mỗi bảng có dòng `no_defense` mốc riêng, **không so chéo**.

### 3.3 Tài nguyên
- **1× GPU H100 80GB** ở chế độ **MIG → 1 slice 40GB**. Chạy tuần tự trên 1 slice.
- Judge/target API: Groq (pool ~22 key, xoay vòng khi rate-limit).

### 3.4 ⭐ Giao thức "fair-cost" cho bài phải train (QUAN TRỌNG cho báo cáo)
**Mọi bài có train đều được TRAIN LẠI trên Llama-3-8B của mình**, KHÔNG lấy số/checkpoint từ paper. Lý do: nếu mỗi bài dùng base + recipe riêng thì cột cost vô nghĩa.

**Ngân sách cố định: 500 optimizer step, cùng một base** → cột train-cost **cùng thang**, khác biệt phản ánh overhead/step thật của thuật toán.

> ⚠️ **Đây là DEVIATION có chủ ý, phải khai báo:** 500 step ≠ recipe hội tụ của paper gốc. Với data lớn (vd DeRTa 65k mẫu → 500 step ≈ 1.5% của 1 epoch) thì đây là "cắt gọn". Vì vậy **số ASR/over-refusal của bài intra KHÔNG so trực tiếp với số công bố của paper** (họ train tới hội tụ). Ta so *tương đối giữa các method dưới cùng ngân sách*, và ghi rõ giới hạn này.

---

## 4. PHẦN DỄ — pre/post & các bài chỉ gọi API (KHÔNG train)

Các bài này chỉ orchestrate lời gọi API tới target model (viết lại prompt / thêm system prompt / để 1 LLM phụ đọc & phán). Không train, không cần GPU. Chỉ cần cắm hàm `transform_prompt` hoặc `generate` vào pipeline chung.

| Nhóm | Method | Cơ chế (1 dòng) | Venue |
|---|---|---|---|
| pre | **SAGE** | rewrite prompt an toàn | — |
| pre | **IA** (Intention Analysis) | 2 bước: phân tích ý định → trả lời | — |
| pre | **G4D** | guide decoding bằng 4 câu hỏi (multi-call) | — |
| pre | **erase-and-check** | xoá dần token + DistilBERT filter (O(n) call) | — |
| pre | **Self-Reminder** | thêm nhắc nhở an toàn vào system prompt | — |
| pre | **Goal Prioritization** | prompt "ưu tiên an toàn hơn hữu ích" + in-context | ACL 2024 |
| post | **LLM Self Defense** | 1 LLM đọc response → phán harmful | — |
| post | **Self-Refine** | model tự phê bình & sửa response (multi-call) | — |
| post | **Backtranslation** | dịch ngược response ra "ý định" rồi kiểm | — |
| post | **AutoDefense** | nhiều agent LLM tranh luận (4 call) | — |
| post | **SelfDefend** | shadow LLM đọc prompt song song | — |

→ Chi phí = **số token API / request** (đo tại chỗ). Đắt nhất: AutoDefense (4 call, 2944 token vào), G4D (3.95 call). Rẻ nhất: SAGE/Self-Reminder (1 call).

---

## 5. PHẦN KHÓ — bài phải TRAIN / REIMPLEMENT (in & intra, chạy local trên GPU)

Đây là phần thầy quan tâm. Bảng dưới ghi rõ: **chạy code gốc hay tự reimplement · data gì · train ra sao · bao nhiêu step/epoch · so với paper gốc thế nào.**

Nguyên tắc bám sát paper (áp cho mọi bài): **(1)** có repo chạy được → chạy code họ, chỉ thêm plumbing tối thiểu, KHÔNG viết lại; **(2)** không có repo / repo rỗng → tự viết nhưng lấy **verbatim** thứ quyết định con số (prompt, hằng số, loss, thuật toán). Mọi sai khác đều khai báo.

### 5.1 Nhóm IN (can thiệp lúc decode — không đổi trọng số)

| Method | Venue | Chạy repo / Reimplement | Data | Train / Calibrate | So với paper gốc |
|---|---|---|---|---|---|
| **SafeDecoding** | ACL'24 | **Chạy code họ** + tự train "expert" | 72 mẫu an toàn (của họ) | Expert = **LoRA r=16, 2 epoch, ~18 giây** | Paper phát hành expert cho **Llama-2**; mình **train lại expert trên Llama-3** (base đổi L2→L3). Cùng thuật toán, khác base → khai báo "reimplement trên Llama-3" |
| **JBShield** | USENIX Sec'25 | **Chạy code họ** (hook import verbatim) | calibration split + jailbreak prompt 9 attack (của họ) | **Calibrate ~15 phút** (KHÔNG train — chỉ trích concept vector + threshold + critical layer) | **Tái hiện được detection accuracy của paper: 0.958** (paper ~0.95). Deviation: chỉnh `GATE=1, FIRST_M=2` để output đọc được (paper cũng nói phải tune scaling factor) |
| **Jailbreak Antidote** | ICLR'25 | **REIMPLEMENT** (không có repo) | AdvBench (harmful) + Alpaca (harmless), 128+128 | **Calibrate ~2.5 phút** (KHÔNG train — dựng PCA direction) | Paper dùng data Phan-2023; mình dùng AdvBench+Alpaca (đều held-out khỏi HarmBench). Cơ chế PCA-PC1 + mask top-5% + hook `h+=α·dir` giữ đúng Eq.1-5. α=0.4 (paper cho khoảng [-0.6,0.6]) |
| **ROSE** | 2024 | **Training-free** (tham khảo repo) | Không cần | Không | Contrastive decode 2 prompt (POS an toàn − α·REV độc hại), α=0.5 (default paper). Prompt lấy verbatim. Chưa tune α cho Llama-3 |

### 5.2 Nhóm INTRA (sửa TRỌNG SỐ vĩnh viễn — đều LoRA/edit 500-step)

| Method | Venue | Chạy repo / Reimplement | Data train | Cách train | Thời gian (H100 40GB) | So với paper gốc |
|---|---|---|---|---|---|---|
| **CAT** | 2024 | **Chạy code họ** | behavior dataset (của họ), mix 0.5 utility / 0.5 adversarial | LoRA, adversarial unlearning, iters=2 | **225 s / 500 step** | Cùng thuật toán; cắt còn 500 step (paper train dài hơn) |
| **DeRTa** | 2024 | **Chạy code họ** | data DeRTa (~65.000 mẫu) | LoRA **r=96**, SFT refusal-shift | **429 s / 500 step** | 500 step = **~1.5% của 1 epoch** trên 65k mẫu → cắt gọn mạnh, khai báo rõ |
| **Circuit Breakers** | NeurIPS'24 | **Chạy code họ** (+ patch tương thích transformers v5) | retain set + circuit-breaker set (của họ) | LoRA, representation rerouting | **507 s / 500 step** | ⚠️ **500 step KHÔNG đủ → ASR = 11% = y hệt no_defense (undertrained)**. Paper train ~3 epoch. Đây là **bằng chứng rõ nhất** cho giới hạn "500-step budget". Đã sửa 1 bug loss (UnboundLocalError) của code họ ở 2 đầu schedule |
| **LED** | Findings EMNLP'24 | **REIMPLEMENT phần editing** (repo chỉ có notebook phân tích, file data rỗng) | 200 AdvBench → câu từ chối (template) | **Edit full-weight 6 layer** E={4,5,6,13,14,15}; loss = early-exit logit-lens tại toxic layer T={29,30,31} (Eq.4) | **~110 s / 500 step** | Layer E/T **tái dùng chỉ số Llama-2-7B** (paper không báo cho Llama-3). Data thay TDC-2023 → AdvBench. LR/step paper không công bố → mình chọn (khai báo) |
| **Safe Unlearning** | 2024 (thu-coai) | **PORT code họ** — loss `safe_unlearning` giữ **VERBATIM** | 1100 mẫu (100 harmful × unlearn/refuse + 500 benign), re-template vicuna→Llama-3 | LoRA + **reference model 4-bit**; loss 3 thành phần (unlearn kiểu DPO + học từ chối + giữ năng lực); batch 4×accum 3 = eff-batch 12 | **1615 s / 500 step (≈5 epoch)** | Paper **full-FT trên 4 GPU (deepspeed)**; mình **LoRA + ref 4-bit trên 1 slice 40GB** (khai báo). Sửa **4 lỗi tương thích v5/numerical** (sampler, num_items_in_batch, guard NaN subgroup rỗng, log_softmax fp32) — không đổi ngữ nghĩa loss |
| **ReFAT** | ICLR'25 | **REIMPLEMENT** (không có repo) | D_r (harmful→refuse) + D_u (benign→help), tái dùng data có sẵn; direction từ AdvBench+Alpaca (500+500) | LoRA **r=128**; forward hook chiếu-bỏ "refusal direction" ở layer 8-31 với xác suất p=0.5 trên batch harmful (mô phỏng tấn công); tính lại direction mỗi 4 step | **~485 s / 500 step** | Paper cũng LoRA r=128 (khớp); 500 step (paper ~313 step/1 epoch); data thay thế (paper dùng Zou-2024 refusals + UltraChat). Layer 8-31 theo Table 4 của paper cho Llama-3 |

**Tóm tắt cách phân loại 6 bài intra:**
- **Chạy code gốc (3):** CAT, DeRTa, Circuit Breakers — chỉ thêm plumbing để cắm vào pipeline.
- **Port code (1):** Safe Unlearning — giữ loss verbatim, chỉ viết lại launcher cho 1 GPU + sửa lỗi version.
- **Reimplement từ paper (2):** LED (repo thiếu editing), ReFAT (không repo).

Và 4 bài IN: chạy code gốc (SafeDecoding, JBShield), reimplement (Antidote), training-free (ROSE).

---

## 6. KẾT QUẢ (số cuối cùng)

*(số đầy đủ + cột cost ở `KET_QUA_FULL_LAN1.md`. `⏳` = đang chấm nốt utility ở local.)*

### Nhóm API (12 bài — đủ 3 metric)
| Method | ASR↓ | Over-refusal↓ | Utility↑ |
|---|---|---|---|
| no_defense (mốc) | 30.7 | 8.0 | 3.63 |
| **Goal Prioritization** | 3.3 | **11.9** | **3.86** |
| IA | 2.0 | 12.4 | 3.76 |
| SAGE | 0.7 | 34.8 | 3.61 |
| G4D | 7.0 | 10.8 | 3.56 |
| erase-and-check | 14.7 | 8.4 | 3.35 |
| Self-Reminder | 4.3 | 22.0 | 3.78 |
| Self_Defense | 9.7 | 35.6 | 3.42 |
| Self_Refine | 6.3 | 12.0 | 3.66 |
| Backtranslation | 17.0 | 9.6 | 3.55 |
| AutoDefense | 18.7 | 9.2 | 3.69 |
| SelfDefend | 0.3 | 28.0 | 3.30 |

### Nhóm LOCAL (11 bài) — **🆕 = 5 bài mới lần này**
| Method | Nhóm | ASR↓ | Over-refusal↓ | Utility↑ |
|---|---|---|---|---|
| no_defense_local (mốc) | — | 11.0 | 8.0 | 3.73 |
| **ROSE** 🆕 | in | **0.3** | 18.8 | 3.80 |
| JBShield | in | 0.0 | 39.2 | 3.43 |
| SafeDecoding | in | 4.3 | 45.8 | 3.20 |
| **Jailbreak Antidote** 🆕 | in | 1.3 | 59.8 | 3.40 |
| CAT | intra | 0.0 | 61.8 | 3.34 |
| DeRTa | intra | 0.3 | 36.0 | 3.45 |
| **LED** 🆕 | intra | **0.7** | 38.8 | 3.59 |
| **Safe Unlearning** 🆕 | intra | 4.0 | 25.6 | 3.57 |
| **ReFAT** 🆕 | intra | 6.3 | **6.4** | 3.53 |
| Circuit Breakers | intra | 11.0 ⚠️ | 7.4 | 3.75 |

---

## 7. NHẬN XÉT CHÍNH

1. **"Không có bữa trưa miễn phí" rất rõ:** bài chặn càng mạnh thì over-refusal càng cao — CAT (ASR 0.0 nhưng over-refusal **61.8%**), Antidote (1.3 / **59.8%**), SafeDecoding (4.3 / 45.8%). Muốn ASR về 0 thường phải trả giá bằng từ chối oan.
2. **Cân bằng tốt nhất:**
   - Nhóm API: **Goal Prioritization** (3.3 / 11.9 / 3.86) và **IA** (2.0 / 12.4 / 3.76).
   - Nhóm local: **ROSE** (0.3 / 18.8 / 3.80) — chặn gần như tuyệt đối mà over-refusal vừa phải, utility cao nhất nhóm local. **ReFAT** đáng chú ý vì over-refusal thấp nhất (6.4%) đúng như mục tiêu adversarial-training.
3. **Bằng chứng cho giới hạn "500-step budget":** Circuit Breakers ở 500 step cho ASR **11.0% = y hệt no_defense** → defense chưa kích hoạt. Trái lại CAT/DeRTa cùng 500 step vẫn về ASR 0-0.3%. Cho thấy một số method (representation rerouting) cần nhiều epoch hơn hẳn — cần thêm giờ GPU nếu muốn số "công bằng với paper".
4. **5 bài mới đều chặn tốt** (ASR 0.3–6.3%, so mốc 11.0%) và đã có over-refusal/utility → đủ để đưa vào survey.

## 8. DEVIATION & CAVEAT (bắt buộc ghi trong báo cáo)

- **Ngân sách train 500 step** ≠ recipe hội tụ của paper → không so trực tiếp số intra với số công bố.
- **Bài local đều train lại trên Llama-3-8B của mình** (không lấy checkpoint/số của paper) → cost cùng thang, nhưng lệch khỏi cấu hình gốc; mỗi bài khai báo cụ thể ở §5 + README.
- **Reimplement (Antidote, LED, ReFAT):** không có/không đủ repo → tự viết theo paper, lấy verbatim phần quyết định con số; các lựa chọn paper bỏ ngỏ (layer set, α, LR) đều là của mình và được ghi rõ.
- **JustEval 200** thay 800 (subset đại diện, seed cố định).
- **Circuit Breakers undertrained** ở 500 step (số không phản ánh khả năng thật của method).
- **DeepRefusal + Targeted LAT:** đã có ASR + over-refusal bằng **checkpoint tác giả** (không train lại) → để riêng ở phần "Phase sau", KHÔNG trộn vào bảng chính để giữ nguyên tắc fair-cost.

---

*Tổng: 23 bài (12 API + 11 local). Toàn bộ response + kết quả đã lưu; over-refusal/utility của vài bài local đang chấm nốt ở local (server GPU đã hết hạn nhưng không ảnh hưởng — judge chỉ cần API).*
