# no_defense (baseline / mốc so sánh)

> Gọi thẳng target, KHÔNG áp phòng thủ. Mọi method khác so với dòng này.
> **Kết quả:** ASR (HarmBench) = **22.7%** · over-refusal (XSTest judge2) = **8.0%**. (Utility JustEval: chưa chạy.)

## 1. Là gì
Baseline: prompt gốc → target (`llama-3.1-8b-instant`, temp=0, max_tokens=512) → response. `transform_prompt = identity`. Dùng chung `core.runner` như các method khác nên cost + over-refusal đo cùng cách → so sánh công bằng.

## 2. Cấu trúc
```
no_defense/
├── method.py       # entry mỏng: identity transform + core.runner
├── requirements.txt
├── outputs/        # response + cost + xstest judged
└── reference/      # 2 notebook Kaggle gốc (đã convert thành method.py)
    ├── generatate-response-harmbench.ipynb
    └── generate-response-xstest.ipynb
```

## 3. Cách chạy
2 stage, mỗi stage chọn `--task {all|harmbench|xstest|justeval}`:
```powershell
conda activate sage
cd "E:\DHBK\JAILBREAK\LAM THI NGHIEM\methods\no_defense"
python method.py response --task all        # sinh 300 HarmBench + 250 XSTest + 800 JustEval
python method.py judge    --task xstest        # baseline over-refusal (API)
python method.py judge    --task justeval      # baseline utility (API)
python method.py judge    --task harmbench     # baseline ASR (classifier GPU 40GB)
```
- Key: dùng chung pool key trong `.env` (xoay vòng khi 429).
- HarmBench: không GPU thì đem `harmbench_no_defense_response.csv` lên Kaggle (`metrics/harmbench.ipynb`).
- XSTest/JustEval judge chạy local (gpt-oss-20b) → `outputs/xstest_no_defense_judged.csv`, `justeval_no_defense_judged.csv`.

## 4. Khai báo
| Mục | Giá trị |
|---|---|
| Model target | `llama-3.1-8b-instant` (Groq) |
| Gọi API infer | Có; temp=0, max_tokens=512 |
| Train | Không |
| Model phụ trợ | Không |
