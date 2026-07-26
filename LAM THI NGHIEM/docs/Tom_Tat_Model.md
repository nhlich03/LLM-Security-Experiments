# Chọn model cho survey

## 1. Model trong paper gốc

**Target** = model sinh câu trả lời cuối, đem đi chấm. **Phụ trợ** = model khác mà cơ chế phòng thủ cần thêm.

| Nhóm | Method | Venue | Target model mã nguồn mở | Target model đóng (API) | Model phụ trợ | Phụ trợ dùng để làm gì | Đang dùng |
|---|---|---|---|---|---|---|---|
| pre | SAGE | Findings ACL 2025 | Gemma-2-9B-IT · Qwen2.5-7B-Instruct · Llama-3.1-8B-Instruct | GPT-4o-mini · GPT-4o · Claude-3.5-Sonnet | không | — | `llama-3.1-8b-instant` |
| pre | IA | COLING 2025 | Vicuna-7B/13B · ChatGLM-6B · Llama2-7B-Chat · Llama3-8B-Instruct | GPT-3.5 | chính target | Phân tích ý định thật của câu hỏi → **đoạn phân tích được ghép vào hội thoại** làm ngữ cảnh, rồi target trả lời ở lượt 2. Không chặn request nào | `llama-3.1-8b-instant` (cả 2 lượt) |
| pre | G4D | Findings NAACL 2025 | Vicuna-v1.5-13B | GPT-4o-mini | GPT-4o-mini (3 agent) | 3 lượt gọi riêng: phát hiện ý định · viết lại câu hỏi (chỉ khi bị nghi ngờ) · phân tích an toàn → cả 3 kết quả **ghép thành đoạn "guidance" chèn vào prompt** gửi cho target. Không chặn request | `llama-3.1-8b-instant` (cả 3 agent + target)<br>→ **thay GPT-4o-mini** |
| pre | erase-and-check | arXiv 2023 | Llama-2 | — | DistilBERT (**66M tham số**, đã fine-tune) | Chấm từng biến thể prompt là harmful hay an toàn → chỉ cần **một** biến thể bị gắn harmful là **chặn thẳng, KHÔNG gọi target**, trả về câu từ chối cố định | target `llama-3.1-8b-instant`<br>filter **DistilBERT** local (giữ như paper) |
| post | LLM Self Defense | arXiv 2023 | Llama 2 | GPT-3.5 | chính target | Đọc câu trả lời của target rồi phán có hại hay không → nếu có hại thì **vứt câu trả lời, thay bằng câu từ chối cố định**; nếu không thì giữ nguyên | `llama-3.1-8b-instant` (cả 2 lượt) |
| post | Self-Refine | NeurIPS 2023 | Vicuna-13B | GPT-3.5 · GPT-4 · text-davinci-003 | chính target | Phê bình câu trả lời rồi viết lại bản mới → bản mới **thay thế** bản cũ, lặp k vòng. Không chặn request | `llama-3.1-8b-instant` (mọi lượt) |
| post | Backtranslation | Findings ACL 2024 | Llama-2-13B-Chat · Vicuna-13B | GPT-3.5-turbo · GPT-4 | Vicuna-13B | Suy ngược từ câu trả lời ra câu hỏi gốc → **hỏi lại target bằng câu hỏi suy ngược đó**; nếu target từ chối câu hỏi này thì câu trả lời ban đầu bị **vứt, thay bằng từ chối** | `llama-3.1-8b-instant` (cả 3 lượt)<br>→ **thay Vicuna-13B** |
| post | AutoDefense | arXiv 2024 | — | GPT-3.5 (con được bảo vệ) | LLaMA-2-13B | Nhiều agent cùng đọc câu trả lời của target rồi bỏ phiếu → nếu kết luận có hại thì **thay câu trả lời bằng từ chối**; nếu không thì giữ nguyên | `llama-3.1-8b-instant` (victim + mọi agent)<br>→ **thay LLaMA-2-13B** |
| in | SafeDecoding | ACL 2024 | Vicuna-7B · Llama-2-7B-chat · Guanaco-7B · Falcon-7B · Dolphin-7B | — | expert = target + LoRA | Đưa ra phân phối xác suất token thiên về từ chối → **trộn với phân phối của target ở 2 token đầu** để lái ngay từ đầu câu; từ token thứ 3 target sinh tiếp bình thường | **local** `Llama-2-7b-chat` + expert LoRA của tác giả<br>*(đã thử cả Llama-3-8B + expert tự train)* |
| in | JBShield | USENIX Sec 2025 | Mistral-7B-v0.2 · Llama-2-7b-chat · Llama-3-8B-Instruct · Vicuna-7B/13B-v1.5 | — | không | — | **local** `Meta-Llama-3-8B-Instruct` |
| intra | CAT / CAPO | NeurIPS 2024 | Gemma-2B · Phi-3-Mini · Mistral-7B · Zephyr-7B · Llama2-7B | — | không | — | **local** `ContinuousAT/Llama3-8B-IT-CAT` |
| intra | Circuit Breakers | NeurIPS 2024 | Mistral-7B-Instruct-v2 · Llama-3-8B-Instruct | — | không | — | **local** `GraySwanAI/Llama-3-8B-Instruct-RR` |
| intra | DeRTa | ACL 2025 | Llama-3-8B · Llama-3-70B (+Instruct) | — | không | — | **local** `Meta-Llama-3-8B-Instruct` + LoRA `Youliang/llama3-8b-instruct-lora-derta-100step` |

---

## 2. Model local nào chạy được nhiều phương pháp nhất


| Mức | ý nghĩa |
|---|---|
| **Pretrain sẵn** | Tác giả đã đăng sẵn model/adapter đã train cho đúng base này lên HuggingFace → tải về chạy, không train gì |
| **Tự train** | **Không** có pretrain cho base này, **nhưng** repo có đủ **code train + dữ liệu train** → chạy script của họ, tốn giờ GPU nhưng không phải tự viết gì |
| **Phải tự viết** | Thiếu code hoặc thiếu dữ liệu cho base này → phải tự bổ sung trước rồi mới train được |

| Base model | SafeDecoding | JBShield | CAT | Circuit Breakers | DeRTa | Pretrain sẵn | Tự train | Phải tự viết |
|---|---|---|---|---|---|:--:|:--:|:--:|
| **Llama-3-8B-Instruct** | Tự train<br><sub>expert LoRA, **16 giây** — đã làm xong</sub> | Pretrain sẵn<br><sub>có sẵn dữ liệu calibration</sub> | Pretrain sẵn<br><sub>`Llama3-8B-IT-CAT`</sub> | Pretrain sẵn<br><sub>`Llama-3-8B-Instruct-RR`</sub> | Pretrain sẵn<br><sub>LoRA 8B</sub> | **4** | 1 | 0 |
| Llama-2-7B-chat | Pretrain sẵn<br><sub>expert LoRA trong repo</sub> | Pretrain sẵn | Pretrain sẵn<br><sub>`Llama-2-7B-CAT`</sub> | Tự train | Phải tự viết | **3** | 1 | 1 |
| Mistral-7B-Instruct-v0.2 | Tự train | Pretrain sẵn | Tự train<br><sub>có trong paper</sub> | Pretrain sẵn<br><sub>`Mistral-7B-Instruct-RR`</sub> | Phải tự viết | **2** | 2 | 1 |
| Vicuna-7B / 13B-v1.5 | Pretrain sẵn | Pretrain sẵn | Phải tự viết | Tự train | Phải tự viết | **2** | 1 | 2 |
| Qwen2.5-7B | Tự train | Phải tự viết | Phải tự viết | Tự train | Phải tự viết | **0** | 2 | 3 |

---

## 3. Model mà Groq hỗ trợ

Đây là toàn bộ model đang có trên Groq 

| Model ID | Tốc độ (token/s) | Giá / 1M token | Rate limit | Context | Max output |
|---|---|---|---|---|---|
| `llama-3.1-8b-instant` — Meta Llama 3.1 8B | 560 | $0.05 vào · $0.08 ra | 250K TPM · 1K RPM | 131,072 | 131,072 |
| `llama-3.3-70b-versatile` — Meta Llama 3.3 70B | 280 | $0.59 vào · $0.79 ra | 300K TPM · 1K RPM | 131,072 | 32,768 |
| `openai/gpt-oss-120b` — GPT OSS 120B | 500 | $0.15 vào · $0.60 ra | 250K TPM · 1K RPM | 131,072 | 65,536 |
| `openai/gpt-oss-20b` — GPT OSS 20B | 1000 | $0.075 vào · $0.30 ra | 250K TPM · 1K RPM | 131,072 | 65,536 |
| `whisper-large-v3` | — | $0.111 / giờ | 200K ASH · 300 RPM | — | — |
| `whisper-large-v3-turbo` | — | $0.04 / giờ | 400K ASH · 400 RPM | — | — |


Hiện tại nếu gọi api thì đều free, nhưng limit. Nhưng hiện tại đang có 15 key api nên xoay vòng chạy được.
---

## 4. Các câu hỏi cần chốt về chọn model

Nhóm theo chủ đề. Mỗi câu kèm một dòng bối cảnh để biết vì sao phải hỏi.

### A. Chọn model TARGET

**A1.** Target cho nhóm chạy API chốt ở **8B** hay nâng lên **70B**?
> Hiện `llama-3.1-8b-instant`. Groq có `llama-3.3-70b-versatile`, đắt gấp ~10 và chậm gấp đôi.

**A2.** Nếu nâng lên 70B, chấp nhận việc **mất khoảng trống so sánh** không?
> 70B đã align tốt sẵn → ASR của `no_defense` sẽ tụt từ 30.7% xuống khoảng 10–15%, các phương pháp dồn cục quanh 2–5% và khó phân biệt nhau.

**A3.** Nếu giữ 8B, trả lời thế nào cho câu "kết quả chỉ đúng với model yếu"?
> 8B align nông nên phương pháp nào cũng trông hiệu quả hơn thực tế. Có cần chạy spot-check 70B cho vài bài để chứng minh thứ hạng không đảo?

**A4.** Target API là **Llama-3.1**-8B còn target local là **Llama-3.0**-8B — hai bảng có được đặt cạnh nhau không?
> Đây là hai model khác nhau, không phải một con chạy hai chế độ. Phải chốt local là 3.0 vì **cả 5 checkpoint intra đều dựng trên 3.0**, đổi sang 3.1 là mất sạch.

**A5.** Có cần chạy `no_defense` local trên **cả 3.0 lẫn 3.1** để đo chênh lệch giữa hai bảng không?
> Có 3 mốc thì mới tách được "chênh do phiên bản model" khỏi "chênh do API vs local".

**A6.** SafeDecoding đang chạy **Llama-2-7b** (vì repo có sẵn expert LoRA cho nó) — chấp nhận, hay bắt buộc kéo về Llama-3?
> Đã train được expert Llama-3 (16 giây). Nhưng nếu dùng expert tự train thì lệch khỏi bản của tác giả.

**A7.** Hai bảng API và local trình bày **tách hẳn**, hay gộp một bảng có chú thích?

### B. Model phụ trợ loại A — LLM chat đa dụng

*(G4D 3 agent · Backtranslation · AutoDefense · SelfDefend shadow · IA/Self_Defense/Self_Refine)*

**B1.** Phụ trợ loại A có **bắt buộc dùng đúng target model** không, kể cả khi paper dùng con mạnh hơn hẳn?
> G4D dùng **GPT-4o-mini** cho cả 3 agent, mình thay bằng `llama-3.1-8b` → đang làm yếu method đi thật.

**B2.** Target là **local** mà phụ trợ loại A gọi **API** (hoặc ngược lại) thì có chấp nhận không?
> Chưa xảy ra, nhưng sẽ xảy ra khi thêm SafeInfer / Aligner.

**B3.** Paper dùng phụ trợ **13B đời cũ**, mình thay bằng **8B đời mới** — cái cần khớp là **số tham số** hay **thế hệ model**?
> Backtranslation dùng Vicuna-13B (2023), AutoDefense dùng Llama-2-13B (2023). Llama-3.1-8B ít tham số hơn nhưng gần như chắc chắn theo chỉ dẫn tốt hơn.

**B4.** Có cần chạy **ablation phụ trợ** (cùng method, đổi aux 8B ↔ 70B) để tách phần đóng góp của aux khỏi cơ chế không?
> Không có số này thì không biết ASR đẹp là do cơ chế hay do con aux thông minh.

### C. Model phụ trợ loại B — component chuyên biệt, CHÍNH LÀ đóng góp của paper

*(erase-and-check DistilBERT · WildGuard 7B · Llama Guard · ShieldGemma · Aligner · SafeDecoding expert)*

**C1.** Loại này có được **đụng vào** không, hay bắt buộc giữ nguyên như tác giả phát hành?
> Train lại WildGuard thì nó không còn là WildGuard — giá trị của nó nằm ở bộ WildGuardMix. Tương tự trọng số DistilBERT của erase-and-check chính là thứ chứng minh tính chất certified.

**C2.** Nếu giữ nguyên, method đó có còn nằm **cùng bảng** với các method khác không, khi nó mang theo một model hoàn toàn khác họ?
> WildGuard là Mistral-7B, target là Llama-3.1-8B.

**C3.** Quy tắc phân biệt là gì: component **gắn với target** thì train lại, component **độc lập** thì giữ nguyên?
> SafeDecoding expert = target + LoRA → đổi target là buộc phải train lại. Còn WildGuard/DistilBERT thì độc lập hoàn toàn với target.

**C4.** Có tính **cost train của component** vào bảng không?
> Mình không train chúng (tải về dùng), nhưng tác giả có train. Nếu không tính thì method loại này trông rẻ hơn thực tế.

**C5.** Component chuyên biệt **lớn ngang hoặc hơn target** thì so sánh có công bằng không?
> WildGuard 7B ~ target 8B; Aligner 7B. Về mặt tổng tài nguyên, method này dùng gấp đôi.

### D. Phụ trợ lớn hơn target

**D1.** Khi phụ trợ **mạnh hơn** target, method thực chất thành *"một model to giám sát một model nhỏ"* — đây có phải điểm yếu của chính paper cần nêu trong survey không?
> Dính 4 bài: Backtranslation (Vicuna-13B), AutoDefense (Llama-2-13B), JBShield (biến thể vicuna-13b), G4D (paper để target là Vicuna-13B).

### E. Model chấm điểm (không phải phụ trợ, nhưng vẫn là model)

**E1.** Judge và classifier **lớn hơn target** thì có sao không?
> Judge XSTest/JustEval = `gpt-oss-20b`, classifier HarmBench = `Llama-2-13b-cls`, target chỉ 8B.

**E2.** Nếu nâng target lên 70B thì judge/classifier có phải nâng theo không?
> Hay giữ nguyên để mọi dòng trong bảng vẫn chấm bằng cùng một thước?

### F. Trình bày trong báo cáo

**F1.** Bảng kết quả có cần thêm cột **"model trong paper"** bên cạnh **"model mình dùng"** để người đọc tự trừ hao không?

**F2.** Sai khác về model khai báo ở đâu — chú thích dưới bảng, hay một mục riêng?
