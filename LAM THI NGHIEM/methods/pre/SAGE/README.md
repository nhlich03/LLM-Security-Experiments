# SAGE — pre-processing defense (P1)

> **Trạng thái:** đã chạy full. ASR (HarmBench, **Llama-13b chính thức**) = **0.7%** · over-refusal (XSTest judge2) = **34.8%** · cost ~1 call/req. (Số Mistral-7b Kaggle cũ = 16.3%; Utility JustEval: chưa chạy.)

## 1. Phương pháp này là gì

**SAGE = Self-Aware Guard Enhancement** (ACL 2025 Findings, NJUNLP) — repo: https://github.com/NJUNLP/SAGE

Cơ chế: **bọc** prompt người dùng bằng một instruction gộp (Semantic Analysis + Task Structure Analysis + Response Protocol) rồi cho target model sinh **một lần**. Training-free, single-call → mẫu cost "1 call target / request".

- **Loại:** pre-processing (can thiệp ở INPUT).
- **Lõi phương pháp = cái prompt** trong `repo/defense_prompts.py`, hàm `make_sage_prompt`. Script import **verbatim**, không sửa 1 ký tự.
- **Không có model phụ trợ.** Chỉ có target.

## 2. Mình đang làm gì trong folder này

Repo gốc của họ **chỉ có prompt + data, không có code chạy**. Nên `method.py` mỏng: khai báo `transform_prompt=make_sage_prompt` rồi gọi `core.runner` — phần sinh/chấm/cost dùng chung `core/`.

`method.py` chỉ lo **GENERATION** (stage `response`). Chấm điểm (stage `judge`) do `core.runner` gọi các scorer trong `metrics/` (xstest, justeval = local; harmbench = GPU/Kaggle).

```
SAGE/
├── repo/            # clone NJUNLP/SAGE (defense_prompts.py, data/) — KHÔNG sửa
├── method.py        # entry mỏng: make_sage_prompt + core.runner
├── requirements.txt
├── outputs/         # response + judged + cost
└── README.md        # file này
```

## 3. Cách chạy (khi bật lại)

**Môi trường:** conda env `sage` (python 3.10, có `pandas` + `requests` + `ipykernel`; KHÔNG cần `groq`). Tạo lại nếu chưa có:
```powershell
conda create -n sage python=3.10 -y
conda run -n sage python -m pip install -r requirements.txt --no-cache-dir   # requirements.txt nam ngay trong folder SAGE
```

**Key:** dùng chung **pool key** trong `.env` (nhiều dòng `GROQ_API_KEY_=...`); `core.runner` tự nạp pool, xoay vòng khi 429. Không cần key riêng cho SAGE.

2 stage, mỗi stage chọn `--task {all|harmbench|xstest|justeval}`:
```powershell
conda activate sage
cd "e:/DHBK/JAILBREAK/LAM THI NGHIEM/methods/pre/SAGE"

# STAGE 1 - sinh response
python method.py response --task all               # 300 HarmBench + 250 XSTest + 800 JustEval
python method.py response --task harmbench --limit 5   # smoke test

# STAGE 2 - cham diem
python method.py judge --task xstest                # over-refusal (API, local)
python method.py judge --task justeval              # utility (API, local)
python method.py judge --task harmbench             # ASR (classifier Llama-13b, GPU 40GB)
```

- **Resume được:** dòng đã có trong file response thì bỏ qua → đứt giữa chừng chạy lại không mất.
- **XSTest judge** gọi `metrics/xstest.py` (gpt-oss-20b, chung pool key) → `xstest_sage_judged.csv` + over-refusal.
- **HarmBench judge**: local cần GPU 40GB (`harmbench.py`, Llama-13b). Máy không GPU → đem `harmbench_sage_response.csv` lên Kaggle chạy `metrics/harmbench.ipynb` (Mistral-7b/T4).
- Windows lỗi font: `set PYTHONIOENCODING=utf-8`.

## 4. Output → làm gì tiếp

File sinh ra trong `outputs/`:

Naming nhất quán: `<task>_<slug>_<kind>.csv` (slug=`sage`).

| File                                              | Dùng cho                                                                   |
| ------------------------------------------------- | -------------------------------------------------------------------------- |
| `harmbench_sage_response.csv`                     | thả lên **Kaggle** chạy `metrics/harmbench.ipynb` → ra **ASR**             |
| `harmbench_sage_judged.csv`                       | (từ Kaggle) HarmBench đã chấm — đặt lại vào đây cho đủ bộ                   |
| `xstest_sage_response.csv` / `_judged.csv`        | XSTest thô / đã chấm → **over-refusal**                                    |
| `justeval_sage_response.csv` / `_judged.csv`      | JustEval thô / đã chấm (5 aspect) → **utility**                            |
| `<task>_sage_cost_detail.csv` / `_summary.csv`    | cost mỗi task (token in/out, số call)                                      |

## 5. Bảng khai báo (để so sánh công bằng — CLAUDE.md §6)

| Mục                            | Giá trị                                                                                 |
| ------------------------------ | --------------------------------------------------------------------------------------- |
| Tên phương pháp                | SAGE                                                                                    |
| Model target (sinh response)   | `llama-3.1-8b-instant` (Groq) — target chung                                            |
| Có gọi API infer không?        | Có, Groq, model như trên. temperature=0, max_tokens=512                                 |
| Có train lại không?            | Không (training-free)                                                                   |
| Có model phụ trợ không?        | Không                                                                                   |
| Degree of freedom mình tự chọn | Chuỗi SAGE đưa vào **một `user` message duy nhất** (repo không quy định system vs user) |

## 6. Điểm cần để ý (pitfall)

- Với XSTest, SAGE hay chèn preamble *"Security Check 1: Semantic Analysis…"* trước câu trả lời. Judge over-refusal có thể hiểu nhầm, và preamble ăn vào `max_tokens=512` → có thể đội over-refusal giả. Đặc tính của SAGE, cần nêu khi so với `no_defense`.
