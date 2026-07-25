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
E:\DHBK\JAILBREAK\
├── .env                    # POOL key Groq (gitignored): nhiều dòng GROQ_API_KEY_=... ; xoay vòng khi 429
├── .gitignore
├── README.md               # file này
├── test/                   # thư mục nháp (gitignored)
└── LAM THI NGHIEM/         # project chính
    │
    ├── docs/               # tài liệu bối cảnh (đọc để hiểu dự án)
    │   ├── CLAUDE.md            # tổng quan dự án (đọc đầu tiên)
    │   ├── BANG_PHUONG_PHAP.md  # bảng 18 phương pháp + lộ trình P1..P5
    │   ├── 01_CACH_CHAY.md      # cách chạy + chuẩn bị file response
    │   └── 02_QUY_UOC_MODEL.md  # quy ước model để so sánh công bằng
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
    │   ├── xstest.py           # over-refusal: judge1 string-match + judge2 gpt-oss-20b (LOCAL/API)
    │   ├── justeval.py         # utility: judge gpt-oss-20b chấm 5 aspect 1-5 (LOCAL/API)
    │   ├── xstest.ipynb        # bản notebook Kaggle tham chiếu
    │   ├── harmbench.ipynb     # ASR (test hiện tại): Mistral-7b — CHẠY KAGGLE T4
    │   └── harmbench.py        # ASR (chạy thật): Llama-2-13b-cls — máy GPU 40GB (template tự khớp model)
    │
    ├── methods/            # CÁC PHƯƠNG PHÁP (mỗi method 1 folder tự chứa)
    │   ├── no_defense/         # baseline (mốc): không phòng thủ (identity transform)
    │   │   ├── method.py
    │   │   ├── outputs/
    │   │   └── reference/      # 2 notebook Kaggle gốc (đã convert thành method.py)
    │   ├── pre/                # pre-processing (can thiệp INPUT): SAGE, IA, G4D, erase-and-check
    │   │   └── SAGE/
    │   │       ├── repo/           # clone upstream (KHÔNG sửa)
    │   │       ├── method.py       # entry mỏng: khai báo defense + gọi core.runner
    │   │       ├── requirements.txt
    │   │       ├── README.md       # method này làm gì + cách chạy lại
    │   │       └── outputs/        # response + judged + cost (tất cả file của method)
    │   ├── post/               # post-processing (can thiệp OUTPUT)
    │   ├── in/                 # in-processing (can thiệp lúc decoding, tạm thời)
    │   └── intra/              # intra-processing (sửa trọng số vĩnh viễn)
    │
    ├── results/            # KẾT QUẢ tổng hợp để so sánh cross-method (bảng cuối)
    │
    └── tools/
        ├── view_outputs.ipynb  # soi nhanh response + cost của TỪNG method
        └── compare_methods.py  # in BẢNG SO SÁNH cross-method (ASR/over-refusal/utility/cost)
```

---

## Giải thích từng thư mục

- **`docs/`** — Bối cảnh & quy ước của dự án. Không có code. Đọc `CLAUDE.md` trước để nắm mục tiêu; `02_QUY_UOC_MODEL.md` giải thích vì sao phải cố định model target.

- **`core/`** — Thư viện dùng chung, là "khung xương" mà mọi phương pháp tái sử dụng để **không phải viết lại** phần gọi API / đo cost / đọc data. Trong đó `runner.py` là quan trọng nhất: nó chạy vòng lặp chuẩn (đọc data → áp cơ chế phòng thủ → gọi target → lưu response → đo cost → resume nếu đứt → tự chấm XSTest). Nhờ `core/`, mỗi method mới chỉ cần ~30 dòng.

- **`data/`** — Dữ liệu đầu vào gốc, **chỉ 2 file** và **không bao giờ bị sửa**: `harmbench.csv` (prompt độc hại để test phòng thủ) và `xstest.csv` (prompt an toàn để test từ chối oan). Mọi phương pháp đọc chung 2 file này → cùng đề bài.

- **`metrics/`** — Bộ chấm điểm, tách rời khỏi phương pháp (một method sinh response, metric chấm response đó). `xstest.py` (over-refusal) và `justeval.py` (utility, 5 aspect) chấm bằng LLM judge trên máy (gọi API, không cần GPU). `harmbench.py`/`.ipynb` chấm ASR bằng classifier nặng → cần **GPU** (local 40GB hoặc Kaggle).

- **`methods/`** — Nơi ở của các phương pháp phòng thủ, chia 4 nhóm theo "can thiệp ở đâu": `pre` (input) · `post` (output) · `in` (lúc decoding) · `intra` (sửa trọng số). Cộng thêm `no_defense` là mốc so sánh. **Mỗi method là 1 folder tự chứa**: `repo/` (code gốc của tác giả, giữ nguyên), `method.py` (điểm chạy, chỉ khai báo cơ chế + gọi `core.runner`), `requirements.txt`, `README.md`, và `outputs/` (mọi file method sinh ra).

- **`results/`** — Nơi gom **số cuối** của nhiều method lại thành bảng so sánh (ASR / over-refusal / cost). File chi tiết của từng method nằm ở `outputs/` của method đó; `results/` chỉ chứa bản tổng hợp để nhìn toàn cảnh.

- **`tools/`** — Tiện ích phụ. `view_outputs.ipynb`: chọn **1 method** rồi xem nhanh response + cost. `compare_methods.py`: chạy `python tools/compare_methods.py` → quét mọi `outputs/`, in **bảng so sánh cross-method** (ASR / over-refusal / utility / cost) và lưu `tools/comparison.md`.

---

## File trong `outputs/` của mỗi method

Naming nhất quán: **`<task>_<slug>_<kind>.csv`** (task = harmbench|xstest|justeval, slug = tên method viết thường).

| File | Sinh ở đâu | Ý nghĩa |
|---|---|---|
| `harmbench_<slug>_response.csv` | local | response cho HarmBench (input để chấm ASR) |
| `xstest_<slug>_response.csv` | local | response cho XSTest |
| `justeval_<slug>_response.csv` | local | response cho JustEval (utility) |
| `harmbench_<slug>_judged.csv` | GPU/Kaggle | HarmBench đã chấm → **ASR** |
| `xstest_<slug>_judged.csv` | local (judge) | XSTest đã chấm → **over-refusal** |
| `justeval_<slug>_judged.csv` | local (judge) | JustEval đã chấm (5 aspect) → **utility** |
| `<task>_<slug>_cost_detail.csv` / `_summary.csv` | local | cost mỗi task (token in/out, số call) |

---

## Luồng chạy một phương pháp — 2 stage

`method.py` có 2 stage, mỗi stage chọn `--task {all|harmbench|xstest|justeval}`:

```
python method.py response --task all     # STAGE 1: sinh response (goi target qua pool key)
                                          #   -> outputs/<task>_<slug>_response.csv + cost

python method.py judge    --task all     # STAGE 2: cham diem
                                          #   xstest   -> metrics/xstest.py   (judge API gpt-oss-20b) -> over-refusal
                                          #   harmbench-> metrics/harmbench.py (classifier Llama-13b, GPU) -> ASR
```

**Chạy SAGE (ví dụ):**

```powershell
conda activate sage
cd "E:\DHBK\JAILBREAK\LAM THI NGHIEM\methods\pre\SAGE"

python method.py response --task all        # sinh 300 HarmBench + 250 XSTest
python method.py judge    --task xstest        # cham over-refusal (API, chạy được local)
python method.py judge    --task harmbench     # cham ASR (classifier GPU 40GB)

python method.py response --task harmbench --limit 5   # smoke test
```

> Máy KHÔNG GPU (hiện tại): chạy `response` + `judge --task xstest` ở local; còn ASR thì
> đem `harmbench_<slug>_response.csv` lên Kaggle chạy `metrics/harmbench.ipynb` (Mistral-7b/T4).
> `judge --task harmbench` bằng `harmbench.py` (Llama-13b) chỉ chạy khi có GPU 40GB.

## Thêm phương pháp mới

1. Tạo `methods/<type>/<TÊN>/` (clone repo upstream vào `repo/` nếu có).
2. Viết `method.py` mỏng: import cơ chế phòng thủ + gọi `core.runner.run_method(...)`.
3. Tạo conda env riêng + `requirements.txt` + `README.md` cho method.
4. (Key dùng chung pool trong `.env` — không cần khai báo key riêng cho method.)
5. Chạy → response + cost + XSTest judged vào `outputs/`; HarmBench đem lên Kaggle.

## Quy ước

- **Model target cố định** cho mọi method (API: `llama-3.1-8b-instant` trên Groq); chỉ cơ chế phòng thủ thay đổi. Chi tiết: `docs/02_QUY_UOC_MODEL.md`.
- **Key: một POOL dùng chung** trong `.env` — liệt kê nhiều key (mỗi dòng `GROQ_API_KEY_=<key>`, hoặc `GROQ_API_KEYS=key1,key2,...`). Cả target lẫn judge xài chung pool, tuần tự; key nào dính 429 thì tự nhảy sang key kế tiếp. Không phân biệt vai trò key.
- **Cost:** token API do Groq trả (mình chỉ đọc `usage`); local đo bằng giây; train đo một lần. Train và infer báo tách 2 cột (khác đơn vị).

## Chạy trên máy GPU 40GB — chấm ASR bằng `harmbench.py`

Trên máy GPU, `python method.py judge --task harmbench` sẽ gọi `metrics/harmbench.py`
(classifier **Llama-2-13b-cls**, template tự khớp model) — không cần Kaggle nữa. Chuẩn bị:

| # | Việc | Chi tiết |
|---|---|---|
| 1 | **Cài lib** vào conda env | `torch`, `transformers`, `accelerate`, `sentencepiece` (+ `tqdm`) — thêm vào `requirements.txt`. |
| 2 | **Model classifier** | Lần đầu tự tải `cais/HarmBench-Llama-2-13b-cls` (~26GB) từ HuggingFace (cần dung lượng + mạng; set `HF_TOKEN` nếu bị rate-limit). |
| 3 | **VRAM / batch** | Llama-13b fp16 ~26GB → hợp GPU 40GB; chỉnh `cls_batch_size` (mặc định 8) theo card. Muốn nhẹ hơn: `--cls-model cais/HarmBench-Mistral-7b-val-cls` (template tự đổi sang Mistral). |

Chỉ vậy — không đổi cấu trúc. `harmbench.py` đã đặt sẵn `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
Máy không GPU: chấm ASR bằng `metrics/harmbench.ipynb` trên Kaggle (Mistral-7b/T4) như hiện tại.
#   L L M - S e c u r i t y - E x p e r i m e n t s  
 