# Cách chạy (pipeline 2-stage)

**Đây là chỗ DUY NHẤT mô tả cách chạy.** `README.md` ở gốc chỉ có quickstart rồi trỏ về đây.

Quy ước model xem `CLAUDE.md` §6. Cấu trúc repo xem `README.md` gốc.

Mỗi method là một folder `methods/<type>/<TÊN>/` có `method.py`, chạy 2 stage:

```bash
cd "LAM THI NGHIEM/methods/pre/SAGE"
python method.py response --task all     # STAGE 1: sinh response (áp phương pháp)
python method.py judge    --task all     # STAGE 2: chấm điểm
```

`--task` chọn: `all` (mặc định, cả 3) | `harmbench` | `xstest` | `justeval`.
`--limit N` để smoke test N dòng đầu.

Method nào cần môi trường riêng thì ghi trong README của chính method đó (vd CAT + DeRTa dùng chung venv legacy `methods/intra/CAT/.venv_cat`).

---

## 3 bộ data (input, trong `data/`)

| File | Nội dung | Metric | Cột |
|---|---|---|---|
| `harmbench.csv` | HarmBench, bỏ 100 copyright, còn **300** (200 standard + 100 contextual) | defense (ASR) | `Behavior, FunctionalCategory, SemanticCategory, Tags, ContextString, BehaviorID` |
| `xstest.csv` | XSTest, **250** safe prompts | over-refusal | `id, prompt, type, label, focus, note` |
| `justeval.csv` | JustEval, **800** instruction helpful (bỏ 200 safety) | utility | `id, prompt, source_id, dataset, category` |

Dùng nguyên, **không tự lọc**. `core/datasets.py` lo việc đọc + ghép prompt.

**Ghép prompt HarmBench** (chỉ lúc *sinh*, file lưu vẫn giữ cột riêng):
- `standard`: prompt = `Behavior`
- `contextual`: prompt = `ContextString` + `\n\n---\n\n` + `Behavior` — behavior loại này chỉ có nghĩa khi có context đứng trước

XSTest / JustEval: prompt = cột `prompt`.

Sinh response với **`max_tokens=512`, `temperature=0`** (chuẩn HarmBench, dùng chung cho mọi method).

---

## STAGE 1 — response (sinh)

Method áp cơ chế phòng thủ rồi gọi target. Hai backend:

| Backend | Target | Ai dùng |
|---|---|---|
| `groq` (mặc định) | `llama-3.1-8b-instant` qua **pool key** trong `.env`, xoay vòng khi 429 | pre / post |
| `local` | trọng số HF trên GPU (`core/local_client.py`) | in / intra |

**Resume:** dòng đã có trong file response thì bỏ qua → đứt giữa chừng chạy lại không mất gì. Cost tự merge, không mất cost của dòng cũ.

---

## STAGE 2 — judge (chấm điểm)

| Metric | Scorer | Chạy ở đâu | Ra file |
|---|---|---|---|
| **over-refusal** (XSTest) | `metrics/xstest.py` — judge1 string-match + judge2 `gpt-oss-20b` | API | `xstest_<slug>_judged.csv` |
| **utility** (JustEval) | `metrics/justeval.py` — `gpt-oss-20b` chấm 5 aspect 1-5 | API | `justeval_<slug>_judged.csv` |
| **ASR** (HarmBench) | `metrics/harmbench.py` — classifier `HarmBench-Llama-2-13b-cls` | **GPU 40GB** | `harmbench_<slug>_judged.csv` |

Judge in số ngay lúc chạy: over-refusal %, utility (1-5), ASR %.

### Chấm ASR trên GPU

| # | Việc | Chi tiết |
|---|---|---|
| 1 | Cài lib | `torch`, `transformers`, `accelerate`, `sentencepiece`, `tqdm` |
| 2 | Model | Lần đầu tự tải `cais/HarmBench-Llama-2-13b-cls` (~26GB) |
| 3 | VRAM | fp16 ~26GB, hợp GPU 40GB. **`--batch-size 4`** — batch 8 OOM ở những batch câu dài |

**GPU chạy MIG** (H100 slice 40GB): NVML bị chặn, phải đặt trước khi chạy, nếu không sẽ crash `NVML_SUCCESS == r ... CUDACachingAllocator`:

```bash
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
```

**Máy không GPU:** đem `harmbench_<slug>_response.csv` lên **Kaggle**, chạy `metrics/harmbench.ipynb` (classifier `Mistral-7b-val-cls`, T4), đổi `SLUG` trong CONFIG, tải file judged về bỏ vào `outputs/`. Lưu ý con Mistral chấm **khác** con Llama-13b — chỉ dùng làm fallback, số chính thức phải là Llama-13b.

---

## File trong `outputs/` của mỗi method

Naming nhất quán: **`<task>_<slug>_<kind>.csv`** (task = `harmbench`|`xstest`|`justeval`, slug = tên method viết thường).

| File | Sinh ở đâu | Ý nghĩa |
|---|---|---|
| `<task>_<slug>_response.csv` | stage 1 | data gốc + cột `response` |
| `harmbench_<slug>_judged.csv` | GPU | đã chấm → **ASR** |
| `xstest_<slug>_judged.csv` | API | đã chấm → **over-refusal** |
| `justeval_<slug>_judged.csv` | API | đã chấm 5 aspect → **utility** |
| `<task>_<slug>_cost_detail.csv` | stage 1 | mỗi call một dòng |
| `<task>_<slug>_cost_summary.csv` | stage 1 | gom theo request |

Cột trong file cost: `n_calls` · `api_in_tokens`/`api_out_tokens` (nhóm API) · `local_in_tokens`/`local_out_tokens`/`local_sec` (nhóm local — ghi cả token lẫn giây, xem `PHUONG_PHAP.md` §7) · `train_sec` (một lần).

---

## Xem kết quả

- `python tools/compare_methods.py` — quét mọi `outputs/`, in **bảng so sánh cross-method**, lưu `tools/comparison.md`. **Đây là bảng kết quả chính thức**, tự sinh nên không bao giờ lệch.
- `tools/view_outputs.ipynb` — soi response + cost của **một** method (đặt `METHOD = "pre/SAGE"`).

Bảng nào chép tay trong doc đều có thể đã cũ — số cuối cùng lấy ở `comparison.md`.

Luôn có `no_defense` làm mốc. Method tốt = kéo **ASR ↓** và **over-refusal ↓** mà vẫn giữ **utility** cao, chấp nhận cost.

---

## Thêm phương pháp mới

1. Tạo `methods/<type>/<TÊN>/`, clone repo upstream vào `repo/` để tham chiếu (**không sửa** `repo/`).
2. Viết `method.py` mỏng: khai báo cơ chế (`transform_prompt` single-call **hoặc** `generate` multi-call) rồi gọi `core.runner.run_method(...)`.
3. Thêm `requirements.txt` + `README.md` (method này làm gì + cách chạy lại).
4. Key dùng chung POOL trong `.env`, không khai báo riêng.

---

## Lỗi hay gặp

| Triệu chứng | Nguyên nhân / cách xử |
|---|---|
| Thiếu key | `.env` phải có ≥1 dòng `GROQ_API_KEY_...=...` (hoặc `GROQ_API_KEYS=k1,k2,...`) |
| Console lỗi font tiếng Việt (Windows) | `set PYTHONIOENCODING=utf-8` trước khi chạy |
| `judge --task harmbench` báo thiếu `torch` | máy không GPU → dùng Kaggle |
| `[skip] chưa có ..._response.csv` | chạy `response` trước |
| `NVML_SUCCESS == r ... CUDACachingAllocator` | GPU chạy MIG → đặt `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` |
| `Cannot access gated repo ... meta-llama/*` | dùng mirror `NousResearch/*` (đã là mặc định) hoặc đặt `HF_TOKEN` |
