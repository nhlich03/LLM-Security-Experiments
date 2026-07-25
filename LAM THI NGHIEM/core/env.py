"""Shared .env loading + Groq key pool (no python-dotenv needed)."""

import os
import re


# ----- Load every .env found walking up from start_dir (setdefault = first wins) -----
def load_env(start_dir, max_up=6):
    loaded = []
    for up in range(0, max_up + 1):
        path = os.path.abspath(os.path.join(start_dir, *([".."] * up), ".env"))
        if os.path.exists(path) and path not in loaded:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            loaded.append(path)
    return loaded


def get_key(name):
    return os.getenv(name)


# ----- Groq key POOL: all keys gathered into one list, used with rotation -----
# Parsed straight from the .env FILE(S) so that MANY lines sharing the same name
# (e.g. several `GROQ_API_KEY_=...`) are all collected — os.environ would keep
# only one. Sources (combined, de-duped by value, order preserved):
#   1) GROQ_API_KEYS = "key1,key2,key3"   (comma / whitespace separated)
#   2) every line whose name starts with GROQ_API_KEY (any suffix, repeats OK)
def load_keys(start_dir, max_up=6):
    load_env(start_dir)                     # still populate os.environ for other vars
    keys, seen_paths = [], set()
    for up in range(0, max_up + 1):
        path = os.path.abspath(os.path.join(start_dir, *([".."] * up), ".env"))
        if not os.path.exists(path) or path in seen_paths:
            continue
        seen_paths.add(path)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, val = line.split("=", 1)
                name, val = name.strip(), val.strip().strip('"').strip("'")
                if not val:
                    continue
                if name == "GROQ_API_KEYS":
                    keys += [p for p in re.split(r"[,\s]+", val) if p]
                elif name.startswith("GROQ_API_KEY"):
                    keys.append(val)
    seen, pool = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            pool.append(k)
    return pool
