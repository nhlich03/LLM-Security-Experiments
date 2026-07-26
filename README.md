# LLM-Security-Experiments

Survey + pipeline đánh giá các **phương pháp phòng thủ (defense) cho LLM**. Mỗi phương pháp được đo trên 3 metric + 1 cost:

- **Defense** — HarmBench, chỉ số **ASR** (càng thấp càng tốt)
- **Over-refusal** — XSTest (càng thấp càng tốt)
- **Utility** — JustEval, 800 instruction helpful, LLM chấm 5 aspect 1-5 (càng cao càng tốt)
- **Cost** — đo tại chỗ (token API · token + giây GPU cho nhóm local · train tách riêng)

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
    ├── docs/               # 4 file, đọc theo thứ tự này
    │   ├── CLAUDE.md               # 1. bối cảnh: taxonomy · 3 metric · cost · quy ước model
    │   ├── PHUONG_PHAP.md          # 2. TẤT CẢ phương pháp: bảng theo nhóm + kết quả + ưu tiên
    │   ├── 01_CACH_CHAY.md         # 3. cách chạy (chỗ duy nhất)
    │   └── Tom_Tat_Model.md        # model trong từng paper
    │
    ├── core/               # THƯ VIỆN DÙNG CHUNG (trái tim pipeline)
    │   ├── env.py              # đọc .env, gom POOL key Groq
    │   ├── groq_client.py      # gọi Groq (keep-alive, xoay vòng key khi 429, fail-fast)
    │   ├── local_client.py     # target chạy LOCAL trên GPU, cùng interface .chat() với Groq
    │   ├── cost_meter.py       # đo cost: token API / token+giây local / train
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
    │   ├── no_defense/         # mốc cho bảng API (Groq)
    │   ├── no_defense_local/   # mốc cho bảng LOCAL (trọng số trên GPU)
    │   ├── pre/                # INPUT: SAGE, IA, G4D, erase-and-check
    │   │   └── SAGE/
    │   │       ├── repo/           # clone upstream (tham chiếu, KHÔNG sửa)
    │   │       ├── method.py       # entry mỏng: khai báo defense + gọi core.runner
    │   │       ├── requirements.txt
    │   │       ├── README.md       # method này làm gì + cách chạy lại
    │   │       └── outputs/        # response + judged + cost (tất cả file của method)
    │   ├── post/               # OUTPUT: Self_Defense, Backtranslation, Self_Refine, AutoDefense
    │   ├── in/                 # lúc decoding, tạm thời: SafeDecoding, JBShield
    │   └── intra/              # sửa trọng số vĩnh viễn: CAT, CircuitBreakers, DeRTa
    │
    └── tools/
        ├── view_outputs.ipynb  # soi nhanh response + cost của TỪNG method
        └── compare_methods.py  # in BẢNG SO SÁNH cross-method (ASR/over-refusal/utility/cost)
```

---

## Kết quả hiện tại (9 method)

ASR chấm bằng classifier chính thức **`HarmBench-Llama-2-13b-cls`** (server GPU). Bảng sinh tự động bằng
`python tools/compare_methods.py` (lưu `tools/comparison.md`). In đậm = tốt nhất cột.

| Method | Nhóm | ASR ↓ | over-refusal (LLM-judge) ↓ | cost: call/req · tok in/out · train |
|---|---|---|---|---|
| no_defense (mốc) | — | 30.7% | 8.0% | 1.0 · 107/234 · 0 |
| **SAGE** | pre | **0.7%** | 34.8% | 1.0 · 251/116 · 0 |
| IA | pre | 2.0% | 12.4% | 2.0 · 523/282 · 0 |
| G4D | pre | 7.0% | 10.8% | 4.0 · 928/664 · 0 |
| erase-and-check | pre | 14.7% | 8.4% | 1.6 · 76/184 · +0.056s filter |
| Self_Refine | post | 6.3% | 12.0% | 3.3 · 1604/797 · 0 |
| Self_Defense | post | 9.7% | 35.6% | 2.0 · 410/352 · 0 |
| Backtranslation | post | 17.0% | 9.6% | 2.5 · 435/533 · 0 |
| AutoDefense | post | 18.7% | 9.2% | 4.0 · 2821/842 · 0 |

**Cost** (đủ cột trong `tools/comparison.md`): `call/req` · `tok_in/req`, `tok_out/req` (token API) · `Ltok_in/req`, `Ltok_out/req`, `local_s/req` (nhóm local ghi **cả token lẫn giây**) · `train_s` (một lần). Train và infer **báo tách** vì khác đơn vị. Ô `-` nghĩa là method không dùng kênh đo đó.

### Nhóm LOCAL (in/intra) — đã code + smoke test, chưa chạy full

5 method chạy trên GPU server bằng checkpoint tác giả: **SafeDecoding · JBShield** (in) · **CAT · Circuit Breakers · DeRTa** (intra), cộng `no_defense_local` làm mốc. Cả 5 đều đã chạy thật và đã train lại thử trên Llama-3. Chi tiết + caveat: `docs/PHUONG_PHAP.md` §5.

*(Utility JustEval: metric đã sẵn, chưa chạy cho method nào.)*

---

## Giải thích từng thư mục

- **`docs/`** — 4 file. `CLAUDE.md` (bối cảnh + quy ước model) → `PHUONG_PHAP.md` (mọi phương pháp: bảng theo nhóm pre/post/in/intra, kết quả đã chạy, thứ tự ưu tiên làm tiếp) → `01_CACH_CHAY.md` (cách chạy). `Tom_Tat_Model.md` là bản riêng cho thầy.
- **`core/`** — Thư viện dùng chung. `runner.py` chạy vòng lặp chuẩn (đọc data → áp phòng thủ → gọi target → lưu response → đo cost → resume nếu đứt → tự chấm XSTest). Nhờ `core/`, mỗi method mới chỉ ~30-50 dòng.
- **`data/`** — Dữ liệu gốc, **không bao giờ sửa**: `harmbench.csv` (độc hại), `xstest.csv` (an toàn), `justeval.csv` (helpful).
- **`metrics/`** — Bộ chấm tách rời method. `xstest.py`/`justeval.py` chấm bằng LLM judge qua API (không cần GPU). `harmbench.py` chấm ASR bằng classifier nặng → cần **GPU 40GB**.
- **`methods/`** — 4 nhóm theo "can thiệp ở đâu": `pre`/`post`/`in`/`intra` + `no_defense` làm mốc. **Mỗi method tự chứa**: `repo/` (code gốc tác giả, tham chiếu), `method.py` (khai báo cơ chế + gọi `core.runner`), `requirements.txt`, `README.md`, `outputs/`.
- **`tools/`** — `view_outputs.ipynb` xem 1 method; `compare_methods.py` in bảng so sánh cross-method.

---

## Chạy thử một phương pháp

```bash
cd "LAM THI NGHIEM/methods/pre/SAGE"
python method.py response --task harmbench --limit 5   # smoke test
python method.py response --task all                   # sinh 300 + 250 + 800
python method.py judge    --task xstest                # over-refusal (API, chạy local được)
python method.py judge    --task harmbench             # ASR (classifier 13B, cần GPU 40GB)
```

→ **Chi tiết đầy đủ ở `LAM THI NGHIEM/docs/01_CACH_CHAY.md`**: 3 bộ data, quy tắc ghép prompt, cơ chế resume, file trong `outputs/`, cách chấm ASR trên GPU (kể cả lưu ý MIG), cách thêm method mới, danh sách lỗi hay gặp.

Bảng kết quả chính thức là `tools/comparison.md`, sinh bằng `python tools/compare_methods.py`.

---

## Quy ước

- **Model target cố định**, chỉ cơ chế phòng thủ thay đổi. Hai bảng, hai target: API = `llama-3.1-8b-instant` (Groq), local = `Meta-Llama-3-8B-Instruct` (GPU). Hai bảng **không cùng thang** → tách riêng, mỗi bảng có `no_defense` của nó. Chi tiết `docs/CLAUDE.md` §6.
- **Key: một POOL dùng chung** trong `.env` (mỗi dòng `GROQ_API_KEY_=<key>`, hoặc `GROQ_API_KEYS=k1,k2,...`). Target lẫn judge xài chung, key nào 429 thì tự nhảy key kế. `.env` **gitignored**, không lên GitHub.
- **Cost:** token API do Groq trả (`usage`); local đo bằng giây; train đo một lần. Train và infer báo tách 2 cột (khác đơn vị).
