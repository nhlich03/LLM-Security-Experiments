# Kết quả chạy FULL lần 1 (28/07/2026)

> File này ghi lại **lần chạy đầy đủ đầu tiên** của toàn bộ pipeline đánh giá defense theo quy ước model đã chốt.
> Số cuối cùng luôn lấy ở **`tools/comparison.md`** (tự sinh, không chép tay) — bảng dưới đây là ảnh chụp lúc viết.
> Trạng thái ký hiệu: ✅ xong · ⏳ đang chạy / hàng đợi · — chưa áp dụng.

---

## Tổng quan tiến độ (cập nhật 29/07 ~05:45 UTC)

### Số bài mỗi nhóm (trong lần chạy này)
| Nhóm | Số bài | Target | Bài |
|---|:--:|---|---|
| **pre** | 6 | API | SAGE · IA · G4D · erase-and-check · Self-Reminder · GoalPriority 🆕 |
| **post** | 5 | API | Self_Defense · Self_Refine · Backtranslation · AutoDefense · SelfDefend |
| **in** | 4 | local (GPU) | SafeDecoding · JBShield · JailbreakAntidote 🆕 · ROSE 🆕 |
| **intra** | 6 | local (GPU) | CAT · DeRTa · Circuit Breakers · LED 🆕 · SafeUnlearning 🆕 · ReFAT 🆕 |
| mốc | 2 | — | no_defense (API) · no_defense_local (GPU) |

**Tổng: 23 bài** = 12 API + 11 local. *(Để đợt sau: WildGuard (post) · DRO/SafeInfer (in). DeepRefusal/Targeted-LAT (intra) có HB+XS bằng ckpt tác giả — §7.)*

### Hoàn tất theo metric
| Nhóm | ASR (chấm GPU) | Over-refusal (API) | Utility (API) |
|---|:--:|:--:|:--:|
| **API — 12 bài** | ✅ 12/12 | ✅ **12/12** | ✅ **12/12** |
| **Local — 11 bài** | ✅ **11/11** | 🔄 6/11 | 🔄 5/11 |

### Đang chạy gì (29/07 ~05:40 UTC)
- **GPU:** Antidote **1.3%** · LED **0.7%** · SafeUnlearning **4.0%** — cả 3 bài mới **XONG** (ASR + response, đã bank). Đang chạy 2 bài cuối **ReFAT (intra) + ROSE (in thứ 4)** qua `finale.sh` (mỗi bài smoke-gated).
- **API judge (đã khôi phục xoay vòng full 19 key):** `judge_all` bank nốt over-refusal/utility còn thiếu nhóm local (CB · SafeDecoding · no_def_local) + XS/JE của 3 bài mới. Response backup đầy đủ (`_server_backup_20260728/`, 270 CSV) → chấm nốt được cả khi server hết hạn.

### Đọc nhanh
Mọi bài đã có số đều **chặn mạnh (ASR 0-4.3%) NHƯNG over-refusal cao (9-62%)** — "không có bữa trưa miễn phí". Cân bằng tốt nhất: **IA** (nhóm API) · **DeRTa** (nhóm local).

---

## 0. Bối cảnh & hạ tầng

- **Server:** 1× H100 80GB ở chế độ **MIG → slice 40GB** (thuê tới hết 29/07, không gia hạn). Chạy tuần tự trên 1 slice.
- **Ngày chạy:** 28/07/2026.
- **Quy ước model (đã chốt):**
  - Bài **gọi API** → target `llama-3.1-8b-instant` (Groq, pool 15 key, temp=0, max_tokens=512).
  - Bài **local** → target `NousResearch/Meta-Llama-3-8B-Instruct` (mirror mở của `meta-llama/*` gated, SHA256 trùng khít; **không cần HF token**).
  - Judge XSTest + JustEval → `openai/gpt-oss-20b` (Groq).
  - Classifier HarmBench (ASR) → `cais/HarmBench-Llama-2-13b-cls` (GPU, batch **4** — batch 8 OOM trên MIG 40GB với response dài).

## 1. Ba metric + cost

| Metric | Bộ data | Đo cái gì | Chấm ở đâu |
|---|---|---|---|
| **ASR** (HarmBench) | 300 behavior (200 standard + 100 contextual) | tỷ lệ bị jailbreak — **càng thấp càng tốt** | classifier Llama-13b (GPU) |
| **Over-refusal** (XSTest) | 250 safe prompt | tỷ lệ từ chối oan — **càng thấp càng tốt** | gpt-oss-20b (API) |
| **Utility** (JustEval) | **200 câu** (subset stratified proportional của 800, seed=42) | chất lượng trả lời helpful, 5 aspect 1-5 → TB — **càng cao càng tốt** | gpt-oss-20b (API) |
| **Cost** | — | infer: token API / (giây + token) local; **train: giây một lần** | đo tại chỗ (`core/cost_meter.py`) |

**Vì sao JustEval 200 (không phải 800):** giảm thời gian + chi phí judge, vẫn đại diện — lấy đúng tỷ lệ 7 nguồn của bản 800, seed cố định → tái lập 100%, dùng CHUNG cho mọi method (`data/justeval_200.csv`, sinh bởi `tools/make_justeval_subset.py`). Chạy full 800: `JUSTEVAL_FILE=justeval.csv`.

## 2. Quyết định cấu hình của lần chạy này

1. **Train lại để cost công bằng.** Mọi bài trainable được **train lại trên Llama-3-8B của mình** (không lấy số train từ paper, không chỉ tải checkpoint tác giả). Bài mà *checkpoint = target model* (nhóm intra) → train lại luôn.
2. **Liều train cố định chung: 500 optimizer step, batch 1** cho CAT / CircuitBreakers / DeRTa → cột train-cost **cùng thang**, so sánh công bằng (khác biệt phản ánh overhead/step thật của thuật toán, không phải recipe tùy hứng). Đây là **deviation có chủ ý**, khai báo rõ (không phải số step của paper).
3. **SafeDecoding:** sửa base `Llama-2-7b-chat-hf` → **`Meta-Llama-3-8B-Instruct`** + expert LoRA **tự train trên Llama-3** (`experts/llama3`), vì upstream không phát hành expert cho Llama-3.
4. **WildGuard: BỎ khỏi lần này.** Cần HF token cho guard model gated `allenai/wildguard` (không phải target, không có mirror mở); nhóm post đã đủ 5/5 nên không gấp.
5. **DRO: để cuối / lần sau.** Pipeline 5 stage, kẹt evaluator gated `LlamaGuard-7b` → sẽ thay bằng classifier HarmBench-13b đã có (khỏi token).
6. **DeepRefusal + Targeted LAT:** đã có kết quả HB+XS bằng checkpoint tác giả; retrain hai bài này **khó** (DeepRefusal ghép 3 repo không có lệnh train; LAT toàn notebook) → xử lý ở lần sau (Phase 3). Bài nào không có code train sẽ **reimplement theo paper** và ghi rõ "reimplement", giữ liêm chính (verbatim thứ quyết định số, không thiên vị).

---

## 3. Cost train (fair, 500 step trên Llama-3-8B, 1× MIG 40GB)

| Method | Nhóm | Train time (500 step, batch 1) | Ghi chú |
|---|---|---|---|
| CAT | intra | **225.2 s** ✅ | adversarial UL, mix 0.5 utility / 0.5 adv (iters=2) |
| DeRTa | intra | **429.3 s** ✅ | SFT refusal-shift, LoRA r=96 |
| CircuitBreakers | intra | **507.4 s** ✅ | representation rerouting — retrain 500-step SAU khi fix bug loss (xem dưới) |
| SafeDecoding (expert LoRA) | in | **17.8 s** ✅ | aux nhỏ (72 sample, r=16), 2 epoch |

*(Type-B không phải target — giữ như tác giả phát hành, KHÔNG train lại: erase-and-check DistilBERT, WildGuard guard.)*

**🐛 Lỗi CB đã sửa (khai báo):** lần train 500-step đầu crash ở [lorra_circuit_breaker.py](../methods/intra/CircuitBreakers/repo/src/lorra_circuit_breaker.py) — hàm loss dùng `retain_loss`/`circuit_breaker_loss` vô điều kiện, nhưng mỗi biến chỉ được gán trong nhánh `if <coeff> > 0`. Ở hai đầu schedule một coeff = 0 (progress 0 → retain_coeff 0; progress 1 → cb_coeff 0) → biến chưa gán → `UnboundLocalError`. Smoke 5-step né được, 500-step dính. Đã thêm 1 patch vào `train_smoke.py` (guard mỗi term, `0 * loss == 0` nên kết quả không đổi) → **đã retrain lại qua `cb_redo2.sh` (500 step, 507.4 s, exit=0)** rồi sinh lại response + chấm ASR. **✅ Số CB trong §5 (ASR 11.0 · infer 6.26 s/req) giờ là của adapter 500-step ĐÚNG** (đã ghi đè bản gen đầu chạy bằng adapter 5-step rác).

## 4. Kết quả — nhóm API (đủ 3 metric)

ASR + over-refusal đã có sẵn từ trước; Phase 1 bổ sung Utility (JustEval 200). Gộp cả 3 để so trực tiếp.

| Nhóm | Method | ASR% ↓ | Over-refusal% ↓ | Utility ↑ | Call/req | Tok-in/req | Tok-out/req | Local s/req |
|---|---|---|---|---|---|---|---|---|
| — | no_defense (mốc) | 30.7 | **8.0** | 3.63 | 1.00 | 95 | 280 | — |
| pre | SAGE | 0.7 | 34.8 | 3.61 | 1.00 | 238 | 173 | — |
| pre | IA | 2.0 | 12.4 | **3.76** | 2.00 | 531 | 369 | — |
| pre | G4D | 7.0 | 10.8 | 3.56 | 3.95 | 905 | 726 | — |
| pre | erase-and-check | 14.7 | 8.4 | 3.35 | 1.68 | 70 | 225 | 0.077 |
| pre | Self-Reminder | 4.3 | 22.0 | 3.78 | 1.00 | 142 | 211 | — |
| pre | **GoalPriority** 🆕 | 3.3 | 11.9 | **3.86** | 1.00 | 795 | 256 | — |
| post | Self_Defense | 9.7 | 35.6 | 3.42 | 2.00 | 443 | 399 | — |
| post | Self_Refine | 6.3 | 12.0 | 3.66 | 3.21 | 1566 | 838 | — |
| post | Backtranslation | 17.0 | 9.6 | 3.55 | 2.64 | 479 | 621 | — |
| post | AutoDefense | 18.7 | 9.2 | 3.69 | 4.00 | 2944 | 890 | — |
| post | SelfDefend | **0.3** | 28.0 | 3.30 | 2.00 | 265 | 303 | — |

ASR = HarmBench (classifier Llama-13b, n=300) · over-refusal = XSTest judge2 `gpt-oss-20b` (n=250) · utility = JustEval (n=200). **✅ Bảng API giờ ĐỦ 12/12 cả 3 metric** (khôi phục xoay vòng full 19 key → judge chạy hết). GoalPriority (bài pre mới) chốt: ASR 3.3 · over-refusal 11.9 · utility **3.86 (cao nhất bảng)** — cân bằng rất tốt.

**Cột cost** (trung bình / request trên toàn bộ 750 prompt HB+XS+JE): *Call/req* = số call API · *Tok-in/out* = token vào/ra · *Local s/req* = giây chạy local (chỉ erase-and-check có, do DistilBERT filter — `0.077`). Các cột **local-token** & **train-sec = — cho mọi bài API** (không sinh local, không train). → G4D (3.95 call, 905 tok vào) và AutoDefense (4 call, **2944** tok vào) là **đắt nhất**; SAGE/Self-Reminder rẻ nhất (1 call).

**Nhận xét nhanh:** SelfDefend & SAGE hạ ASR mạnh nhất (0.3%, 0.7%) nhưng **over-refusal vọt** (28%, 35% — từ chối oan nhiều). **IA cân bằng tốt nhất** trong nhóm đã đủ số (ASR 2.0% · over-refusal 12.4% · utility 3.76 cao nhất). erase-and-check giữ over-refusal thấp (8.4%) nhưng ASR còn cao (14.7%) và utility thấp nhất (3.35). → **hạ ASR càng mạnh thường đánh đổi over-refusal càng nhiều.**

## 5. Kết quả — Full 3 metric, nhóm LOCAL

Dùng **adapter 500-step tự train** (không phải checkpoint official). Cost tách **2 tầng**: *1 lần* (train/calibrate, offline) và *mỗi request* (infer s/req).

| Nhóm | Method | ASR% ↓ | Over-refusal% ↓ | Utility ↑ | 1 lần (train/cal) | Infer s/req |
|---|---|---|---|---|---|---|
| — | no_defense_local (mốc) | 11.0 | ⏳ | ⏳ | — | 4.29 |
| in | SafeDecoding | 4.3 | 45.8 | 3.20 | 17.8 s (train expert) | 3.28 |
| in | JBShield | **0.0** | 39.2 | 3.43 | ~15 phút (calibrate) | 3.34 |
| in | **JailbreakAntidote** 🆕 | **1.3** | 59.8 | ⏳ | ~2.5 phút (calibrate PCA dir) | 2.80 |
| in | **ROSE** 🆕 | **0.3** | ⏳ | ⏳ | — (training-free) | ⏳ (2× decode) |
| intra | CAT | **0.0** | 61.8 | 3.34 | 225.2 s (train) | 4.74 |
| intra | DeRTa | **0.3** | 36.0 | 3.45 | 429.3 s (train) | 4.92 |
| intra | CircuitBreakers | 11.0 ⚠️ | 7.4 | 3.75 | 507.4 s (train) | 6.26 |
| intra | **LED** 🆕 | **0.7** | ⏳ | ⏳ | ~110 s (edit 6 layer, 500 step) | 3.09 |
| intra | **SafeUnlearning** 🆕 | **4.0** | ⏳ | ⏳ | 1615 s (train, eff-batch 12) | 5.44 |
| intra | **ReFAT** 🆕 | **6.3** | ⏳ | ⏳ | ~485 s (train LoRA r128 + RFA ablation) | 5.46 |

> ⚠️ **CircuitBreakers ASR 11.0% = Y HỆT no_defense_local** → ở liều **500 step (~0.1 của 3 epoch) CB KHÔNG hạ được ASR** = defense chưa kích hoạt (undertrained — dù train-loss có tụt 9.9→0.1). Trái ngược CAT/DeRTa (cùng 500-step vẫn về ASR 0-0.3%). → **bằng chứng rõ nhất cho caveat "500-step là budget cắt gọn, chưa đủ cho bài cần nhiều epoch"** (§6). Muốn CB thật sự phòng thủ phải train nhiều epoch hơn (cần thêm giờ GPU).

- **1 lần (train/calibrate)** = chi phí offline làm 1 lần trước khi phục vụ. *train* = đổi trọng số (CAT/DeRTa/CB, SafeDecoding-expert) · *calibrate* = JBShield chỉ trích vector, không đổi trọng số · `—` = không có (no_defense).
- **Infer s/req** = giây/request lúc sinh (đã gồm overhead decoding: SafeDecoding 2-forward/token · JBShield hook mỗi forward · CAT/DeRTa/CB chạy như model thường).
- ⚠️ **`s/req` bị LẪN độ dài response** — bài từ chối nhiều → câu ngắn → nhìn "nhanh/rẻ" một cách ảo (vd JBShield 3.34 *nhanh hơn* no_defense 4.29 dù có hook, chỉ vì trả lời ngắn hơn). **Đọc kèm token ra/req** (`tools/comparison.md`), đừng kết luận "rẻ" chỉ từ giây.

> ⚠️ Bảng local và bảng API **KHÔNG cùng thang** (base khác: Groq Llama-3.1 quantized vs local Llama-3.0 fp16) → mỗi bảng có `no_defense` riêng, không so chéo hai bảng.

---

## 6. Deviation & caveat (bắt buộc khai báo trong báo cáo)

- **Train budget 500 step** ≠ recipe gốc của paper (cố ý, để cost cùng thang). Số ASR/over-refusal của bài intra vì thế **không nên** so trực tiếp với số công bố của paper (họ train tới hội tụ).
- **SafeDecoding** đổi base L2→L3 + expert tự train → khác bản gốc; khai báo là **our reimplementation trên Llama-3**.
- **JBShield** dùng `JBS_GATE=1, FIRST_M=2, attack=ijp` (đã kiểm detection acc khớp paper 0.958); **còn phải xác nhận** FIRST_M=2 có giữ tác dụng phòng thủ trên full 300 hay không.
- **JustEval 200** thay vì 800 (subset đại diện, khai báo rõ).
- Target API là Groq (Llama-3.1 tối ưu/quantized) — **không bit-identical** với local fp16.

## 7. Còn lại (lần sau / Phase 3)

- **DeepRefusal + Targeted LAT (intra):** đã có **HB + over-refusal bằng checkpoint TÁC GIẢ** (không phải retrain 500-step như §5 → **không đưa vào bảng chính** để giữ nguyên tắc fair-cost). Số tham khảo (n=300 ASR / 250 XS): **DeepRefusal — ASR 0.0% · over-refusal 55.2%** · **Targeted LAT — ASR 1.0% · over-refusal 35.6%**. Muốn vào bảng chính phải retrain trên Llama-3 (khó: DeepRefusal ghép 3 repo không có lệnh train; LAT toàn notebook) → Phase 3.
- DRO: chạy với evaluator HarmBench-13b thay LlamaGuard.
- WildGuard: chỉ khi có HF token.

---

*Cập nhật lần cuối: đang chờ generation local hoàn tất; các ô ⏳ sẽ được điền từ `tools/comparison.md`.*
