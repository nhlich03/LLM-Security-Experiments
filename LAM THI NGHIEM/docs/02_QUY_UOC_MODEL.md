# File 2 — Quy ước model / API (để so sánh công bằng)

Chúng ta so sánh nhiều **phương pháp phòng thủ** khác nhau. Mỗi paper/phương pháp có thể tự chọn model, tự train, tự gọi API khác nhau. Nếu để mỗi người một kiểu, kết quả sẽ **không so sánh được** — vì lúc đó ta đang so *model* chứ không phải so *phương pháp phòng thủ*.

Vì vậy tất cả phương pháp phải tuân theo các quy ước dưới đây.

---

## Nguyên tắc gốc: cố định model target

**Model target** = con LLM sinh ra **response cuối cùng** (chính là thứ đem đi chấm HarmBench/XSTest).

> Model target phải **giống hệt nhau** ở mọi phương pháp. Chỉ có *cơ chế phòng thủ* được thay đổi.

Nếu phương pháp A dùng Qwen còn phương pháp B dùng Llama thì bảng so sánh vô nghĩa. Cố định target là điều kiện tiên quyết để con số nói lên "phương pháp nào phòng thủ tốt hơn".

**Model target chung của cả nghiên cứu:** `llama-3.1-8b-instant` (Groq) — cho nhóm method chạy API. *(Nhóm local/train sau này base = Qwen2.5-1.5B-Instruct; tách bảng riêng.)*

---

## 3 trường hợp và quy ước tương ứng

### TH1 — Phương pháp gọi API để infer

Nếu cơ chế phòng thủ cần gọi API một LLM (ví dụ: dùng một LLM để viết lại prompt, phân loại prompt, kiểm duyệt output...), thì **dùng đúng API và model quy ước chung**, không mỗi người một API.

- API dùng chung: `Groq`
- Model target: `llama-3.1-8b-instant`. Judge (XSTest + JustEval): `openai/gpt-oss-20b`. Classifier ASR (HarmBench): `cais/HarmBench-Llama-2-13b-cls` (Kaggle test: `Mistral-7b-val-cls`).
- Tham số target: temperature `0`, max_tokens `512`. Key: **pool dùng chung** trong `.env` (nhiều `GROQ_API_KEY_...`), xoay vòng khi 429.

Lý do: cùng một cơ chế nhưng gọi GPT-4 sẽ khác hẳn gọi model 8B. Cố định để chi phí/hành vi nhất quán.

### TH2 — Phương pháp dùng local model (thường train lại)

Nếu phương pháp fine-tune / train lại model (ví dụ adversarial training, safety tuning...), thì:

- **Base model để train lại phải là model target chung** ở trên — không đổi sang base khác.
- Sau khi train, chính model đã train đó trở thành model target sinh response.
- Ghi rõ: cấu hình train (LoRA/full, data, số step...) là *phần của phương pháp*, được phép khác nhau giữa các paper.

Lý do: cùng xuất phát từ một base thì mới biết phương pháp train nào cải thiện phòng thủ tốt hơn.

Model local dùng chung (base): `Qwen2.5-1.5B-Instruct` *(dùng cho nhóm method local/train — P4-P5; chưa triển khai)*

### TH3 — Phương pháp dùng 2 LLM (target + phụ trợ)

Một số phương pháp dùng 2 con: 1 con **target** sinh response, 1 con **phụ trợ** hỗ trợ target (ví dụ lọc prompt trước khi đưa vào, kiểm duyệt output sau khi sinh, làm guard model...).

- **Con target**: phải là model target chung (theo nguyên tắc gốc) — không được đổi.
- **Con phụ trợ**: là *một phần của phương pháp*, do paper tự chọn. Được phép khác nhau giữa các phương pháp, nhưng **phải ghi rõ** dùng con gì (tên, size, chạy local hay API).

Lý do: con phụ trợ chính là "cái hay" mà phương pháp đóng góp, nên để nó tự do. Nhưng con target phải cố định để response cuối so sánh được.

---

## Bảng khai báo (mỗi phương pháp điền 1 dòng)

Khi nộp kết quả, mỗi phương pháp ghi kèm bảng này để minh bạch:

| Mục | Giá trị |
|---|---|
| Tên phương pháp | |
| Model target (sinh response) | *(phải = target chung)* |
| Có gọi API infer không? Nếu có: API + model | |
| Có train lại không? Nếu có: base + cấu hình | |
| Có model phụ trợ không? Nếu có: tên + size + local/API | |
| Ghi chú khác | |

---

## Tóm tắt một câu

Giữ **model target cố định** cho mọi phương pháp; những gì thuộc về *cơ chế phòng thủ* (model phụ trợ, cách train, cách gọi API) thì tuân quy ước chung ở trên và khai báo minh bạch. Có vậy bảng so sánh cuối mới phản ánh đúng "phương pháp nào tốt hơn".
