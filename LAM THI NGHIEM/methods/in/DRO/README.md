# DRO — Directed Representation Optimization

**Nhóm:** in · **Venue:** ICML 2024
**Paper:** [Prompt-Driven LLM Safeguarding via Directed Representation Optimization](https://arxiv.org/abs/2401.18018) · **Repo gốc:** https://github.com/chujiezheng/LLM-Safeguard (clone trong `repo/`)

## Method làm gì

Quan sát khởi đầu: model **đã "biết"** phân biệt harmful/harmless trong không gian biểu diễn — chiếu hidden state của prompt độc và prompt sạch lên đúng hướng thì chúng đã tách ra rồi, chỉ là **ranh giới chưa đủ rộng**.

DRO không dạy model điều gì mới. Nó tối ưu một **soft prompt liên tục** (embedding, không phải chữ) sao cho:
- biểu diễn của prompt **harmful** bị đẩy **theo** hướng từ chối
- biểu diễn của prompt **harmless** bị đẩy **ngược lại**

Deploy = prepend soft prompt đó vào input → **không có overhead decoding**, ~1×T.

## ⚠️ Bắt buộc train trước mới chạy được

**Upstream không phát hành soft prompt đã train.** Khác với 3 bài intra (có checkpoint tải về là chạy), ở đây không train thì không có gì để nạp.

Train là pipeline **3 stage**, tất cả đều cần GPU:

| Stage | Script | Làm gì |
|---|---|---|
| 1 | `forward.py` | dump hidden state token cuối cho 100 prompt harmful (`data/custom.txt`) + 100 harmless (`data_harmless/custom.txt`) |
| 2 | `estimate.py` | từ hidden state đó ước lượng **hướng từ chối** và **hướng harmful** (PCA basis + ranh giới logistic) |
| 3 | `train.py` | tối ưu soft prompt (khởi tạo từ embedding của system prompt mặc định), AdamW lr=1e-3 |

```bash
python train_smoke.py                 # smoke: 20 query mỗi bên
python train_smoke.py --full          # 100+100 như upstream
python train_smoke.py --stage forward # chạy lại một stage
```

Ra file `work/trained_prompts/<model>/type.all_length.default.safetensors`, rồi:

```bash
python method.py response --task harmbench
python method.py response --task xstest
python method.py judge    --task xstest
python method.py judge    --task harmbench
```

`method.py` tự tìm soft prompt ở đường mặc định; override bằng `DRO_SOFT_PROMPT=<path>`.

## Hai thứ upstream không chạy được với Llama-3, đã vá vào bản copy

`repo/` **giữ nguyên**, `train_smoke.py` sinh bản đã vá trong `work/`.

**A. Llama-3 không có trong whitelist.** `train.py`, `forward.py`, `generate.py`, `forward_with_soft.py`, `train_unlikelihood.py` đều kết thúc dãy `elif` bằng:

```python
else:
    raise ValueError(f"Unsupported or untuned model: {model_name}")
```

Chỉ nhận Llama-2 / CodeLlama / Orca-2 / Mistral / Vicuna / OpenChat. Cùng kiểu lỗi với CAT (`NotImplementedError`). → thêm nhánh Llama-3.

**B. Không có chat template Llama-3.** Mấy file đó **ép ghi đè** template của tokenizer từ file `.jinja`, mà `chat_templates/` không có bản Llama-3. Tự viết template thì dễ sai tinh vi mà nó lại quyết định từng vị trí token. → Vá cho phép `"chat_template": null` thì **giữ template built-in của tokenizer** — bản đó chính là template chính thức của Meta, trung thành hơn là tự viết.

## Cách nạp soft prompt lúc infer

Lấy verbatim logic từ `repo/code/generate.py::process_soft_prompt_as_word_embedding`:

1. Thêm N token giả `<soft_prompt_0..N-1>` vào tokenizer
2. Mở rộng bảng embedding, ghi soft prompt đã train vào đúng N ô mới
3. Prepend một **system message** gồm đúng N token đó

Nhờ vậy soft prompt đi qua chat template bình thường, không phải hack `inputs_embeds`. `lm_head` không đổi → các token giả **không bao giờ bị sinh ra**, đúng ý upstream.

## ⚠️ Vấn đề phân loại: borderline in/pre

Soft prompt = embedding đã train rồi prepend vào input → xét chặt thì **cùng họ với RPO** (pre, optimized-prompt). Giữ ở IN theo định nghĩa "in = can thiệp tầng biểu diễn/decoding". Chi tiết ở `docs/PHUONG_PHAP.md` §5.1.

Chuyển sang PRE thì **không có bài thay thế** — 3 ứng viên IN còn lại (InferAligner, SafeInt, Jailbreak Antidote) đều repo rỗng.

## Trạng thái — 🔧 ĐÃ CODE, đang train trên server

Sau khi có soft prompt sẽ chạy `--task harmbench` + `--task xstest` đầy đủ.

## Lưu ý khi báo cáo

- **Soft prompt tự train**, không phải bản của tác giả (họ không phát hành) → số không so trực tiếp với paper được.
- Data train là `custom.txt` **có sẵn trong repo** (100 harmful + 100 harmless) → phần này trung thành.
- Bản smoke cắt còn 20+20 query, phải chạy `--full` mới dùng cho bảng kết quả chính.
