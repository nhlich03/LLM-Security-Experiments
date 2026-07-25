# File 1 — Cách chạy (pipeline 2-stage)

Cách chạy một phương pháp phòng thủ để ra số cho 3 metric. Quy ước model/API xem File 2 (`02_QUY_UOC_MODEL.md`). Cấu trúc repo tổng quan xem `README.md` ở gốc.

Mỗi method là 1 folder `methods/<type>/<TÊN>/` có `method.py`. Chạy 2 stage:

```powershell
conda activate <env-cua-method>          # vd: sage, ia
cd "E:\DHBK\JAILBREAK\LAM THI NGHIEM\methods\pre\SAGE"

python method.py response --task all     # STAGE 1: sinh response (áp phương pháp)
python method.py judge    --task all     # STAGE 2: chấm điểm
```

`--task` chọn: `all` (mặc định, cả 3) | `harmbench` | `xstest` | `justeval`.

---

## 3 bộ data (input, trong `data/`)

| File | Nội dung | Metric | Cột |
|---|---|---|---|
| `harmbench.csv` | HarmBench, bỏ 100 copyright, còn **300** (200 standard + 100 contextual) | defense (ASR) | `Behavior, FunctionalCategory, SemanticCategory, Tags, ContextString, BehaviorID` |
| `xstest.csv` | XSTest, **250** safe prompts | over-refusal | `id, prompt, type, label, focus, note` |
| `justeval.csv` | JustEval, **800** instruction helpful (bỏ 200 safety) | utility | `id, prompt, source_id, dataset, category` |

Dùng nguyên, không tự lọc. `core/datasets.py` lo việc đọc + ghép prompt.

**Ghép prompt HarmBench** (chỉ lúc *sinh*, file lưu vẫn giữ cột riêng):
- `standard`: prompt = `Behavior`.
- `contextual`: prompt = `ContextString` + `\n\n---\n\n` + `Behavior` (behavior loại này chỉ có nghĩa khi có context trước).

XSTest / JustEval: prompt = cột `prompt`.

Sinh response với **`max_tokens=512`, `temperature=0`** (chuẩn HarmBench, dùng chung).

---

## STAGE 1 — response (sinh, không cần GPU)

Gọi target (`llama-3.1-8b-instant` trên Groq) qua **pool key** trong `.env` (xoay vòng khi 429). Method áp cơ chế phòng thủ (SAGE bọc prompt, IA 2 call...) rồi lưu:

- `outputs/<task>_<slug>_response.csv` — data gốc + cột `response`.
- `outputs/<task>_<slug>_cost_detail.csv` / `_summary.csv` — cost (token in/out, số call).

**Resume:** dòng đã có trong file response thì bỏ qua → đứt giữa chừng chạy lại không mất. Cost tự merge (không mất cost dòng cũ).

---

## STAGE 2 — judge (chấm điểm)

| Metric | Scorer | Chạy ở đâu | Ra file |
|---|---|---|---|
| **over-refusal** (XSTest) | `metrics/xstest.py` — judge1 string-match + judge2 `gpt-oss-20b` | **Local** (API) | `xstest_<slug>_judged.csv` |
| **utility** (JustEval) | `metrics/justeval.py` — `gpt-oss-20b` chấm 5 aspect 1-5 | **Local** (API) | `justeval_<slug>_judged.csv` |
| **ASR** (HarmBench) | `metrics/harmbench.py` — classifier `HarmBench-Llama-2-13b-cls` | **GPU 40GB** | `harmbench_<slug>_judged.csv` |

**HarmBench khi không có GPU:** đem `harmbench_<slug>_response.csv` lên **Kaggle**, chạy `metrics/harmbench.ipynb` (classifier `HarmBench-Mistral-7b-val-cls`, T4), đổi `SLUG` trong CONFIG, tải file judged về bỏ vào `outputs/`.

Judge in số ngay lúc chạy: over-refusal %, utility (1-5), ASR %.

---

## Xem nhanh output

- `tools/view_outputs.ipynb` — chọn `METHOD = "pre/SAGE"` để xem response + cost **1 method**.
- `python tools/compare_methods.py` — quét mọi `outputs/`, in **bảng so sánh cross-method** (ASR / over-refusal / utility / cost), lưu `tools/comparison.md`.

---

## Ghép bảng so sánh

Luôn có `no_defense` (baseline) làm mốc. Method tốt: kéo **ASR ↓** và **over-refusal ↓** mà giữ **utility** cao, chấp nhận cost.

Bảng dưới sinh tự động bằng `python tools/compare_methods.py` (ASR = classifier Llama-13b):

| Phương pháp | ASR ↓ | over-refusal ↓ (j2) | utility ↑ | cost (call/req) |
|---|---|---|---|---|
| no_defense (mốc) | 30.7% | 8.0% | (chưa) | 1.0 |
| SAGE | 0.7% | 34.8% | (chưa) | 1.0 |
| IA | 2.0% | 12.4% | (chưa) | 2.0 |
| G4D | 7.0% | 10.8% | (chưa) | 4.0 |
| erase-and-check | 14.7% | 8.4% | (chưa) | 1.6 + filter |

---

## Lỗi hay gặp
- Thiếu key: `.env` phải có ≥1 dòng `GROQ_API_KEY_...=...` (hoặc `GROQ_API_KEYS=k1,k2,...`).
- Console lỗi font tiếng Việt (Windows): `set PYTHONIOENCODING=utf-8` trước khi chạy.
- HarmBench judge local báo thiếu `torch` → máy không GPU, dùng Kaggle.
- Chạy `judge` mà báo `[skip] chưa có ..._response.csv` → chạy `response` trước.
