"""
Groq chat client over plain HTTP with a KEY POOL + rotation.

One Session (keep-alive). Given a list of API keys, it uses them in order and,
whenever a key hits 429 (rate limit), immediately rotates to the next key
instead of sleeping. Only when EVERY key is rate-limited in a row does it back
off briefly, then cycles again. Invalid keys (401/403) are dropped from the pool.

chat() returns (text, resp_obj); resp_obj.usage has prompt/completion tokens,
compatible with cost_meter.record_api().
"""

import time
from types import SimpleNamespace

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqClient:
    def __init__(self, keys, model, temperature=0.0, max_tokens=512,
                 token_param="max_tokens", extra=None, max_all_limited_cycles=20):
        self.keys = [k for k in keys if k]
        if not self.keys:
            raise RuntimeError("GroqClient: empty key pool.")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.token_param = token_param        # "max_tokens" (chat) or "max_completion_tokens" (reasoning)
        self.extra = extra or {}
        self.max_all_limited_cycles = max_all_limited_cycles
        self.idx = 0
        self.session = requests.Session()

    def _payload(self, messages):
        p = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if self.max_tokens:
            p[self.token_param] = self.max_tokens
        p.update(self.extra)
        return p

    def chat(self, user_content, system=None, **extra):
        """Single user message (+ optional system). Returns (text, resp_obj)."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_content})
        return self.chat_messages(messages, **extra)

    def chat_messages(self, messages, **extra):
        """Arbitrary messages list (multi-turn, e.g. user/assistant/user for IA).
        Same key-pool rotation. Returns (text, resp_obj)."""
        payload = self._payload(messages)
        payload.update(extra)

        rl_streak = 0          # consecutive 429s across keys
        all_limited_cycles = 0
        while True:
            key = self.keys[self.idx]
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            try:
                r = self.session.post(GROQ_URL, headers=headers, json=payload, timeout=120)
            except requests.RequestException as e:
                print(f"    [net] {type(e).__name__}: {e}; retry 3s...", flush=True)
                time.sleep(3)
                continue

            code = r.status_code
            if code == 429:
                # This key is rate-limited -> jump to the next key immediately.
                self.idx = (self.idx + 1) % len(self.keys)
                rl_streak += 1
                if rl_streak >= len(self.keys):
                    # every key limited in a row -> short backoff, then keep cycling
                    all_limited_cycles += 1
                    if all_limited_cycles > self.max_all_limited_cycles:
                        raise RuntimeError("All keys rate-limited for too long.")
                    wait = min(2 * all_limited_cycles, 20)
                    print(f"    [keys] all {len(self.keys)} limited, waiting {wait}s...", flush=True)
                    time.sleep(wait)
                    rl_streak = 0
                continue
            if code in (401, 403):
                # bad/expired key -> drop it from the pool and try another
                bad = self.keys.pop(self.idx)
                print(f"    [keys] dropping invalid key (...{bad[-4:]}), {len(self.keys)} left", flush=True)
                if not self.keys:
                    raise RuntimeError("All keys invalid (401/403).")
                self.idx %= len(self.keys)
                continue
            if code >= 500:
                print(f"    [retry] HTTP {code}; 3s...", flush=True)
                time.sleep(3)
                continue
            if code >= 400:
                raise RuntimeError(f"Groq HTTP {code}: {r.text[:200]}")

            data = r.json()
            text = data["choices"][0]["message"]["content"]
            u = data.get("usage", {}) or {}
            usage = SimpleNamespace(
                prompt_tokens=u.get("prompt_tokens", 0),
                completion_tokens=u.get("completion_tokens", 0),
            )
            return (text if text is not None else ""), SimpleNamespace(usage=usage)
