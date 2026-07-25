# CLAUDE.md — Bối cảnh dự án: Đánh giá phương pháp phòng thủ (Defense) cho LLM

File này để Claude Code (và người mới) bắt nhịp nhanh. Đọc hết trước khi làm.

> ⚠️ **Cập nhật:** file này là *bối cảnh/thiết kế gốc*. **Cấu trúc repo + cách chạy HIỆN TẠI** (đã refactor: `core/` + `methods/` + `metrics/`, chạy bằng `method.py response|judge --task ...`, không còn dùng 2 notebook như mô tả bên dưới) — xem **`README.md` ở gốc repo**. Phần taxonomy/metric/cost/quy-ước dưới đây vẫn đúng; phần "cách chạy bằng notebook" đã bị thay bằng pipeline `core.runner`.

---

## 1. Mục tiêu dự án

Nghiên cứu về **LLM Security**, trọng tâm hiện tại là **defense**. Đang làm một **survey** về các phương pháp phòng thủ, và xây pipeline đánh giá chúng trên 3 metric + 1 metric cost.

Câu hỏi cốt lõi cho mỗi phương pháp phòng thủ:
- Nó chặn được tấn công không? (defense)
- Nó có từ chối oan các prompt vô hại không? (over-refusal)
- Áp vào thì model còn trả lời tử tế không? (utility)
- Nó tốn bao nhiêu? (cost)

---

## 2. Taxonomy phương pháp phòng thủ (do tác giả tự định nghĩa)

Chia theo "can thiệp vào đâu trong vòng đời một request":

- **pre-processing**: phòng thủ ở INPUT, trước khi target model xử lý. (lọc/viết lại prompt, thêm safety system prompt, detect-rồi-chặn)
- **post-processing**: phòng thủ ở OUTPUT, sau khi model đã sinh response. (guard đọc response, self-critique, kiểm duyệt)
- **in-processing**: can thiệp lúc sinh (decoding), thay đổi TẠM THỜI (logits/xác suất token/activation), hết request thì model về nguyên trạng. Không tạo checkpoint mới.
- **intra-processing**: sửa trọng số VĨNH VIỄN, tạo target model mới (safety fine-tuning, DPO/RLHF, adversarial training).

Ranh giới in vs intra = **tạm thời vs vĩnh viễn**.

LƯU Ý survey: cặp tên "in/intra" tự đặt, khác khung fairness truyền thống (in-processing = training-time). Phải định nghĩa rõ đầu survey. Cân nhắc thêm trục phụ **detector vs transformer** (detector = quyết định chặn/flag; transformer = biến đổi response) vì nó giải thích cost + trade-off tốt hơn.

---

## 3. Danh sách paper đã phân loại (đã verify)

Doc trong BANG_PHUONG_PHAP.MD

---

## 4. Ba metric đánh giá (mỗi cái 1 notebook: response file vào -> số ra)

Kiến trúc chung: tách GENERATION (phụ thuộc phương pháp) khỏi SCORING (notebook cố định). Cầu nối = file response schema cố định. LUÔN có dòng "no_defense" làm mốc.

### 4.1 HarmBench (defense capability — ASR, càng thấp càng tốt)
- Data: `harmbench_behaviors_text_all.csv`, LOẠI 100 copyright, giữ 300 (standard + contextual).
- File response = data gốc + cột `response` (giữ đủ cột: Behavior, FunctionalCategory, SemanticCategory, ContextString, BehaviorID).
- Sinh response: `max_tokens=512` (chuẩn HarmBench, KHÔNG đổi), temperature=0. Contextual: prompt = ContextString + Behavior, nhưng file lưu 2 cột riêng.
- Classifier: `cais/HarmBench-Mistral-7b-val-cls` (7B, vừa T4; đang dùng). Khi có GPU 40GB đổi sang `cais/HarmBench-Llama-2-13b-cls` (test classifier chính thức). Template Mistral KHÁC Llama (Mistral: [BEHAVIOR]/[GENERATION], không <<SYS>>). Template + model phải đi thành cặp.
- Chấm: greedy, sinh 1 token yes/no. yes=jailbroken. Response rỗng=no. ASR = tỷ lệ yes.
- Chạy Kaggle T4x2, tách cell load model / cell run. Nếu OOM: BATCH_SIZE=2, max_length=1024, sort input theo độ dài, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (đặt trước import torch).

### 4.2 XSTest (over-refusal, càng thấp càng tốt)
- Data: chỉ SAFE prompts (~250, đã bỏ unsafe).
- File response: `prompt, response` (hoặc giữ id, type, label + response).
- 2 judge: judge1 = string matching (verbatim XSTest); judge2 = LLM judge = Groq `openai/gpt-oss-20b` (reasoning: max_completion_tokens=512, reasoning_effort="low").
- Metric: over-refusal = tỷ lệ bị từ chối (full_refusal + partial_refusal cho judge2). Toàn safe nên từ chối = xấu.
- Không cần GPU (gọi API). OUTPUT_PATH phải ở /kaggle/working/.

### 4.3 Utility — ĐÃ CHỐT: JustEval
- Dùng **JustEval**: 800 instruction helpful (full just-eval-instruct bỏ 200 safety vì safety đã có HarmBench). LLM judge (`gpt-oss-20b`) chấm 5 aspect (helpfulness/clarity/factuality/depth/engagement) 1-5, utility = trung bình. Scorer: `metrics/justeval.py`. (Trước cân nhắc Arena-Hard / MT-Bench nhưng chốt JustEval.)

---

## 5. Metric COST (2 cột: train một lần / infer mỗi request)

QUAN TRỌNG: train cost (một lần) và infer cost (mỗi request) KHÔNG cùng đơn vị — báo cáo TÁCH 2 CỘT, không gộp.

Đơn vị THÔ (tiền quy đổi tính sau, để placeholder):
- API infer -> token (prompt+completion, lấy từ response.usage) × đơn giá Groq
- Local infer -> giây (perf_counter + cuda.synchronize) × giá thuê GPU
- Train -> giây (một lần) × giá thuê GPU

Cost đo TẠI CHỖ trong lúc phương pháp chạy (không tách rời như 2 metric kia). Dùng module `core/cost_meter.py` — `core.runner` tự nhúng vào mọi method.

5 mẫu cost:
1. Multi-call (cộng dồn nhiều call): G4D, Self-Refine, AutoDefense
2. O(n) call (đắt): erase-and-check
3. First-token forward (rẻ): FJD
4. Decoding-time overhead (mỗi token): SafeDecoding, GeDi — overhead = so với no_defense
5. Train một lần: SecAlign + model phụ (expert/discriminator/probe/soft-prompt)

In-processing (mẫu 3-4) gần như bắt buộc chạy local (cần truy cập forward/logits), API không xen vào decode được.

---

## 6. Quy ước model (để so sánh công bằng)

Model target (sinh response cuối, đem đi chấm) phải CỐ ĐỊNH; chỉ cơ chế phòng thủ thay đổi.

| Loại                 | Model                                     | Nguồn      |
| -------------------- | ----------------------------------------- | ---------- |
| Target chạy API      | llama-3.1-8b-instant                      | Groq       |
| Target chạy local    | Qwen 1.5B (train lại hay không tuỳ paper) | local host |
| Judge XSTest + JustEval | openai/gpt-oss-20b                     | Groq       |
| Classifier HarmBench | Llama-2-13b-cls (local GPU) / Mistral-7b-val-cls (Kaggle T4) | local/Kaggle |

- Method gọi API infer: dùng model quy ước chung.
- Method train lại (intra): base PHẢI là Qwen 1.5B.
- Method dùng 2 LLM (target + phụ trợ): target cố định; con phụ trợ tự do nhưng phải khai báo.
- Nhóm target-API (llama-3.1-8b) và target-local (Qwen 1.5B) KHÔNG cùng thang so sánh (base khác nhau) — nếu gộp bảng thì tách 2 bảng, mỗi bảng có no_defense riêng.

---

## 7. Cấu trúc repo (chi tiết xem `README.md` gốc)

- `core/` — thư viện chung: `env.py` (pool key), `groq_client.py`, `cost_meter.py`, `datasets.py`, `runner.py`.
- `data/` — `harmbench.csv` (300), `xstest.csv` (250), `justeval.csv` (800).
- `metrics/` — scorer: `xstest.py` (over-refusal), `justeval.py` (utility), `harmbench.py` + `harmbench.ipynb` (ASR).
- `methods/<type>/<TÊN>/` — mỗi method: `method.py` + `repo/` + `outputs/` + README. Có `no_defense` làm mốc.
- `docs/` — bối cảnh (file này, 01, 02, BANG_PHUONG_PHAP).
- `tools/view_outputs.ipynb` — soi output.

---

## 8. Phong cách làm việc mong muốn

- Trả lời tiếng Việt, thuật ngữ kỹ thuật giữ tiếng Anh.
- Code Python để paste vào cell Kaggle / dùng trong repo.
- Critique thẳng, chỉ ra pitfall, không gật đầu cho qua — đây là nghiên cứu, cần chặt chẽ.
- Với việc bám chuẩn benchmark: giữ verbatim phần quyết định con số (classifier, prompt, định nghĩa metric); phần đổi (I/O, engine, model nhỏ hơn) phải khai báo rõ trong báo cáo.
