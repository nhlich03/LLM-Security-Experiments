# LLM-Security-Experiments

Survey + pipeline đánh giá các **phương pháp phòng thủ (defense) cho LLM**. Mỗi phương pháp được đo trên 3 metric + 1 cost:

- **Defense** — HarmBench, chỉ số **ASR** (càng thấp càng tốt)
- **Over-refusal** — XSTest (càng thấp càng tốt)
- **Utility** — JustEval, 800 instruction helpful, LLM chấm 5 aspect 1-5 (càng cao càng tốt)
- **Cost** — đo tại chỗ (token API / giây GPU / train)

Nguyên tắc cốt lõi: tách **GENERATION** (sinh response, tuỳ phương pháp) khỏi **SCORING** (chấm điểm, bộ chấm cố định). Cầu nối là file response có schema cố định. Mọi phương pháp đều so với mốc **`no_defense`** trên cùng model target → con số nói lên "phương pháp nào phòng thủ tốt hơn", không lẫn yếu tố model.

---

## Cấu trúc repo

```
JAILBREAK/
├── .env                    # POOL key Groq (gitignored): nhiều dòng GROQ_API_KEY_=... ; xoay vòng khi 429
├── .gitignore
├── README.md               # file này
└── LAM THI NGHIEM/         # project chính
    │
    ├── docs/               # tài liệu bối cảnh (đọc để hiểu dự án)
    │   ├── CLAUDE.md               # tổng quan dự án (đọc đầu tiên)
    │   ├── BANG_PHUONG_PHAP.md     # bảng 20 phương pháp + venue/năm + tiến độ
    │   ├── PHUONG_PHAP_MOI.md      # ~24 method mới tìm thêm (chưa triển khai) + phân tích
    │   ├── 01_CACH_CHAY.md         # cách chạy + chuẩn bị file response
    │   └── 02_QUY_UOC_MODEL.md     # quy ước model để so sánh công bằng
    │
    ├── core/               # THƯ VIỆN DÙNG CHUNG (trái tim pipeline)
    │   ├── env.py              # đọc .env, gom POOL key Groq
    │   ├── groq_client.py      # gọi Groq (keep-alive, xoay vòng key khi 429, fail-fast)
    │   ├── cost_meter.py       # đo cost: token API / giây local / train
    │   ├── datasets.py         # load HarmBench/XSTest + ghép prompt
    │   └── runner.py           # vòng lặp chung: data→defense→target→save+cost+resume+auto-judge
    │
    ├── data/               # dataset gốc (input)
    │   ├── harmbench.csv       # 300 behaviors độc hại (đã bỏ 100 copyright)
    │   ├── xstest.csv          # 250 safe prompts
    │   └── justeval.csv        # 800 instruction helpful (đã bỏ 200 safety)
    │
    ├── metrics/            # BỘ CHẤM ĐIỂM (scorer, dùng cho mọi method)
    │   ├── xstest.py           # over-refusal: judge1 string-match + judge2 gpt-oss-20b (API)
    │   ├── justeval.py         # utility: judge gpt-oss-20b chấm 5 aspect 1-5 (API)
    │   ├── harmbench.py        # ASR: classifier Llama-2-13b-cls (GPU 40GB) — template tự khớp model
    │   └── harmbench.ipynb     # bản notebook Kaggle (Mistral-7b/T4) — fallback khi không có GPU
    │
    ├── methods/            # CÁC PHƯƠNG PHÁP (mỗi method 1 folder tự chứa)
    │   ├── no_defense/         # baseline (mốc): không phòng thủ (identity transform)
    │   ├── pre/                # pre-processing (INPUT): SAGE, IA, G4D, erase-and-check
    │   │   └── SAGE/
    │   │       ├── repo/           # clone upstream (tham chiếu, KHÔNG sửa)
    │   │       ├── method.py       # entry mỏng: khai báo defense + gọi core.runner
    │   │       ├── requirements.txt
    │   │       ├── README.md       # method này làm gì + cách chạy lại
    │   │       └── outputs/        # response + judged + cost (tất cả file của method)
    │   ├── post/               # post-processing (OUTPUT): Self_Defense, Backtranslation, Self_Refine, AutoDefense
    │   ├── in/                 # in-processing (lúc decoding, tạm thời) — chưa triển khai
    │   └── intra/              # intra-processing (sửa trọng số vĩnh viễn) — chưa triển khai
    │
    └── tools/
        ├── view_outputs.ipynb  # soi nhanh response + cost của TỪNG method
        └── compare_methods.py  # in BẢNG SO SÁNH cross-method (ASR/over-refusal/utility/cost)
```

---

## Kết quả hiện tại (9 method)

ASR chấm bằng classifier chính thức **`HarmBench-Llama-2-13b-cls`** (server GPU). Bảng sinh tự động bằng
`python tools/compare_methods.py` (lưu `tools/comparison.md`). In đậm = tốt nhất cột.

| Method | Nhóm | ASR ↓ | over-refusal (LLM-judge) ↓ | cost (call/req) |
|---|---|---|---|---|
| no_defense (mốc) | — | 30.7% | 8.0% | 1.0 |
| **SAGE** | pre | **0.7%** | 34.8% | 1.0 |
| IA | pre | 2.0% | 12.4% | 2.0 |
| G4D | pre | 7.0% | 10.8% | 4.0 |
| erase-and-check | pre | 14.7% | 8.4% | 1.6 + filter |
| Self_Refine | post | 6.3% | 12.0% | ~3.3 |
| Self_Defense | post | 9.7% | 35.6% | 2.0 |
| Backtranslation | post | 17.0% | 9.6% | ~2.5 |
| AutoDefense | post | 18.7% | 9.2% | 4.0 |

*(Utility JustEval: metric đã sẵn, chưa chạy cho các method. `in`/`intra`: chưa triển khai — xem `docs/PHUONG_PHAP_MOI.md` cho ứng viên.)*

---

## Giải thích từng thư mục

- **`docs/`** — Bối cảnh & quy ước. Đọc `CLAUDE.md` trước; `BANG_PHUONG_PHAP.md` là bảng phương pháp chính; `02_QUY_UOC_MODEL.md` giải thích vì sao phải cố định model target.
- **`core/`** — Thư viện dùng chung. `runner.py` chạy vòng lặp chuẩn (đọc data → áp phòng thủ → gọi target → lưu response → đo cost → resume nếu đứt → tự chấm XSTest). Nhờ `core/`, mỗi method mới chỉ ~30-50 dòng.
- **`data/`** — Dữ liệu gốc, **không bao giờ sửa**: `harmbench.csv` (độc hại), `xstest.csv` (an toàn), `justeval.csv` (helpful).
- **`metrics/`** — Bộ chấm tách rời method. `xstest.py`/`justeval.py` chấm bằng LLM judge qua API (không cần GPU). `harmbench.py` chấm ASR bằng classifier nặng → cần **GPU 40GB**.
- **`methods/`** — 4 nhóm theo "can thiệp ở đâu": `pre`/`post`/`in`/`intra` + `no_defense` làm mốc. **Mỗi method tự chứa**: `repo/` (code gốc tác giả, tham chiếu), `method.py` (khai báo cơ chế + gọi `core.runner`), `requirements.txt`, `README.md`, `outputs/`.
- **`tools/`** — `view_outputs.ipynb` xem 1 method; `compare_methods.py` in bảng so sánh cross-method.

---

## File trong `outputs/` của mỗi method

Naming nhất quán: **`<task>_<slug>_<kind>.csv`** (task = harmbench|xstest|justeval, slug = tên method viết thường).

| File | Sinh ở đâu | Ý nghĩa |
|---|---|---|
| `harmbench_<slug>_response.csv` | local/server | response cho HarmBench (input để chấm ASR) |
| `xstest_<slug>_response.csv` | local/server | response cho XSTest |
| `harmbench_<slug>_judged.csv` | GPU | HarmBench đã chấm → **ASR** |
| `xstest_<slug>_judged.csv` | API (judge) | XSTest đã chấm → **over-refusal** |
| `justeval_<slug>_judged.csv` | API (judge) | JustEval đã chấm (5 aspect) → **utility** |
| `<task>_<slug>_cost_detail.csv` / `_summary.csv` | local | cost mỗi task (token in/out, số call) |

---

## Luồng chạy một phương pháp — 2 stage

`method.py` có 2 stage, mỗi stage chọn `--task {all|harmbench|xstest|justeval}`:

```
python method.py response --task all     # STAGE 1: sinh response (gọi target qua pool key)
python method.py judge    --task all     # STAGE 2: chấm điểm
#   xstest    -> metrics/xstest.py    (judge API gpt-oss-20b)      -> over-refusal
#   justeval  -> metrics/justeval.py  (judge API gpt-oss-20b)      -> utility
#   harmbench -> metrics/harmbench.py (classifier Llama-13b, GPU)  -> ASR
```

**Chạy SAGE (ví dụ):**

```bash
conda activate sage        # hoặc venv trên server
cd "LAM THI NGHIEM/methods/pre/SAGE"
python method.py response --task all                    # sinh 300 HarmBench + 250 XSTest
python method.py judge    --task xstest                 # over-refusal (API, chạy được local)
python method.py judge    --task harmbench              # ASR (classifier GPU 40GB)
python method.py response --task harmbench --limit 5    # smoke test
```

---

## Chạy chấm ASR trên GPU (`harmbench.py`)

`python method.py judge --task harmbench` gọi `metrics/harmbench.py` (classifier **`cais/HarmBench-Llama-2-13b-cls`**, template tự khớp model). Chuẩn bị:

| # | Việc | Chi tiết |
|---|---|---|
| 1 | Cài lib | `torch`, `transformers`, `accelerate`, `sentencepiece`, `tqdm` |
| 2 | Model | Lần đầu tự tải `cais/HarmBench-Llama-2-13b-cls` (~26GB) từ HuggingFace |
| 3 | VRAM / batch | Llama-13b fp16 ~26GB, hợp GPU 40GB. **`--batch-size 4`** (batch 8 dễ OOM ở batch câu dài) |

**Lưu ý GPU MIG** (vd H100 chạy MIG 40GB): NVML bị chặn → phải đặt
`export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` trước khi chạy (nếu không sẽ crash
`NVML_SUCCESS == r ... CUDACachingAllocator`). **Máy không GPU:** chấm ASR bằng
`metrics/harmbench.ipynb` trên Kaggle (Mistral-7b/T4) làm fallback.

---

## Thêm phương pháp mới

1. Tạo `methods/<type>/<TÊN>/` (clone repo upstream vào `repo/` để tham chiếu nếu có).
2. Viết `method.py` mỏng: khai báo cơ chế phòng thủ (`transform_prompt` single-call hoặc `generate` multi-call) + gọi `core.runner.run_method(...)`.
3. Thêm `requirements.txt` + `README.md` cho method.
4. Key dùng chung POOL trong `.env` — không cần khai báo key riêng.
5. Chạy → response + cost + XSTest judged vào `outputs/`; ASR chấm bằng GPU.

---

## Quy ước

- **Model target cố định** cho mọi method (API: `llama-3.1-8b-instant` trên Groq); chỉ cơ chế phòng thủ thay đổi. Nhóm local/train dùng base riêng — xem `docs/02_QUY_UOC_MODEL.md`.
- **Key: một POOL dùng chung** trong `.env` (mỗi dòng `GROQ_API_KEY_=<key>`, hoặc `GROQ_API_KEYS=k1,k2,...`). Target lẫn judge xài chung, key nào 429 thì tự nhảy key kế. `.env` **gitignored**, không lên GitHub.
- **Cost:** token API do Groq trả (`usage`); local đo bằng giây; train đo một lần. Train và infer báo tách 2 cột (khác đơn vị).
