# AutoDefense — Multi-Agent LLM Defense (post-processing, P3)

> target trả lời → **3 agent** (Intention → OriginalPrompt → Judge) đọc response → Judge phán VALID/INVALID; INVALID → thay bằng từ chối.

## 1. Là gì
Paper: [AutoDefense](https://arxiv.org/pdf/2403.04783) · repo: [XHMY/AutoDefense](https://github.com/XHMY/AutoDefense)

Config: **3-agent detector** (`ex-3`, bản headline). Prompt verbatim từ `data/prompt/defense_prompts.json`. Mọi call cùng target `llama-3.1-8b`:
0. **victim**: target trả lời prompt gốc → response (con bị "soi").
1. **IntentionAnalyzer**: phân tích ý định của response.
2. **OriginalPromptAnalyzer**: suy ra 3 prompt gốc khả dĩ.
3. **Judge**: đọc 2 phân tích + response → `Judgment: VALID/INVALID`.

VALID → giữ response gốc; INVALID / "I apologize" / không có token → **REFUSAL** (fail-closed). *(Coordinator/Output agent trong repo là router có sẵn, không gọi LLM → bỏ.)*

## 2. Cách chạy
```bash
python method.py response --task harmbench      # 4 call/prompt
python method.py response --task xstest
python method.py judge    --task xstest
python method.py judge    --task harmbench
```

## 3. Khai báo
| Mục | Giá trị |
|---|---|
| Model target | `llama-3.1-8b-instant` (Groq) |
| Gọi API infer | Có, **4 call/request** (victim + 3 agent); temp=0, max_tokens=512 |
| Train | Không |
| Model phụ trợ | Không (3 agent = chính target với system prompt khác nhau; gốc cho phép model khác) |
| Config | 3-agent `ex-3` (headline); parse verdict = substring VALID/INVALID |
