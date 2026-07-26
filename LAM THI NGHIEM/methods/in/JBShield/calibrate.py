"""
JBShield - step 1: concept calibration (produces the vectors JBShield-M needs).

JBShield does not train anything, but it cannot run cold either: it first has to LEARN,
per model, where the two concepts live:
  - toxic concept     : present in both harmful and jailbreak prompts
  - jailbreak concept : present only in jailbreak prompts; this is what flips the model
                        from refusing to complying

Upstream does that inside `repo/detection.py::detection(model_name, update_vectors=True)`,
which writes into `repo/vectors/<model_name>/`:
    mean_harmful_embedding.pt / mean_harmless_embedding.pt
    calibration_safety_vector.pt / calibration_jailbreak_vector_<attack>.pt
    thershold_safety_<attack>.pt / thershold_jailbreak_<attack>.pt   (sic, upstream typo)
    delta_safety.pt / delta_jailbreak_<attack>.pt
    layer_indexs.pt          (critical layer per concept)
Those files are NOT in the repo, so this must run once before method.py.

Calibration data ships with the repo (`repo/data/`): harmful/harmless calibration splits
plus jailbreak prompts for 9 attacks (gcg, autodan, saa, drattack, pair, puzzler, ijp,
base64, zulu) x 5 models. Nothing to download.

Upstream hardcodes local model directories in `repo/config.py` (e.g. "./models/Llama-2-7b-chat-hf").
We patch that dict at import time so a plain HF repo id works instead.

Cost: embeddings for the calibration sets + SVD per layer. Minutes, not hours. GPU needed.

Usage:
  python calibrate.py --model llama-3
  JBS_MODEL_PATH=meta-llama/Meta-Llama-3-8B-Instruct python calibrate.py --model llama-3

`--model` must be one of upstream's five keys, because the jailbreak prompt files are named
after them: mistral | llama-2 | llama-3 | vicuna-7b | vicuna-13b.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "repo")

# Upstream key -> the HF repo id we actually want to load
DEFAULT_HF_PATHS = {
    "llama-3": "NousResearch/Meta-Llama-3-8B-Instruct",   # mirror: meta-llama/* is gated
    "llama-2": "NousResearch/Llama-2-7b-chat-hf",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "vicuna-7b": "lmsys/vicuna-7b-v1.5",
    "vicuna-13b": "lmsys/vicuna-13b-v1.5",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3", choices=list(DEFAULT_HF_PATHS),
                    help="upstream model key (jailbreak prompt files are named after it)")
    ap.add_argument("--model-path", default=None,
                    help="HF repo id / local dir to load instead of upstream's ./models/... path")
    a = ap.parse_args()

    model_path = (a.model_path or os.environ.get("JBS_MODEL_PATH")
                  or DEFAULT_HF_PATHS[a.model])

    # Upstream resolves every relative path against the repo root, so run from there.
    os.chdir(REPO)
    sys.path.insert(0, REPO)

    import config
    config.model_paths[a.model] = model_path
    print(f"[calibrate] {a.model} -> {model_path}")

    os.makedirs(os.path.join(REPO, "vectors", a.model), exist_ok=True)
    os.makedirs(os.path.join(REPO, "logs"), exist_ok=True)

    from detection import detection      # noqa: E402  (needs cwd + patched config first)

    # update_vectors=True is the whole point: it persists the concept vectors.
    # The same call also prints JBShield-D detection accuracy per attack - free extra number.
    detection(a.model, update_vectors=True)

    out = os.path.join(REPO, "vectors", a.model)
    print(f"\n[done] vectors written to {out}")
    print("Files:", sorted(os.listdir(out))[:6], "...")
    print(f"\nNext:  JBS_MODEL_KEY={a.model} python method.py response --limit 5")


if __name__ == "__main__":
    main()
