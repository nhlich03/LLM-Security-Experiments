"""
runner.run_method - generic GENERATION harness for API pre-processing methods
(single target call per request: prompt -> transform_prompt -> target).

A method's script only needs to supply its defense transform + config:

    from core.runner import run_method
    from defense_prompts import make_sage_prompt
    run_method(name="SAGE", slug="sage", defense_type="pre",
               key_name="GROQ_API_KEY_SAGE", model="llama-3.1-8b-instant",
               transform_prompt=make_sage_prompt, out_dir=".../outputs")

The harness handles: .env + key, Groq client (keep-alive/retry), HarmBench &
XSTest data + prompt building, resume, cost metering, and the auto XSTest judge.
CLI flags (--task / --limit / --no-eval) are parsed here.
"""

import os
import sys
import argparse

import pandas as pd

from .env import load_keys
from .groq_client import GroqClient
from .cost_meter import CostMeter
from . import datasets

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ----- Uniform file naming: <task>_<slug>_<kind>.csv (one place, whole pipeline) -----
def response_file(out_dir, task, slug):
    return os.path.join(out_dir, f"{task}_{slug}_response.csv")


def judged_file(out_dir, task, slug):
    return os.path.join(out_dir, f"{task}_{slug}_judged.csv")


def cost_files(out_dir, task, slug):
    return (os.path.join(out_dir, f"{task}_{slug}_cost_detail.csv"),
            os.path.join(out_dir, f"{task}_{slug}_cost_summary.csv"))


# ----- Resume: set of finished keys already in the output file -----
def _load_done(out_path, key_col):
    if os.path.exists(out_path):
        try:
            d = pd.read_csv(out_path)
            if key_col in d.columns and "response" in d.columns:
                return set(d.loc[d["response"].notna(), key_col].astype(str))
        except Exception:
            pass
    return set()


# ----- Append one row (write header if new); crash-safe -----
def _append(out_path, keep_cols, row_dict):
    df = pd.DataFrame([{c: row_dict.get(c) for c in keep_cols + ["response"]}])
    df.to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False, encoding="utf-8")


# ----- One task loop (harmbench / xstest / justeval) -----
def _run_task(task, client, generate, name, defense_type, slug, out_dir, limit):
    if task == "harmbench":
        src = datasets.load_harmbench()
        keep, key_col = datasets.HARMBENCH_KEEP, "BehaviorID"
        build_raw = datasets.harmbench_prompt
    elif task == "justeval":
        src = datasets.load_justeval()
        keep, key_col = datasets.JUSTEVAL_KEEP, "id"
        build_raw = lambda row: str(row["prompt"])
    else:  # xstest
        src = datasets.load_xstest()
        keep, key_col = datasets.XSTEST_KEEP, "id"
        build_raw = lambda row: str(row["prompt"])

    if limit:
        src = src.head(limit)
    out = response_file(out_dir, task, slug)
    done = _load_done(out, key_col)
    print(f"\n=== {name} - {task.upper()} ===\nsource rows: {len(src)}"
          + (f" | resume skip {len(done)}" if done else ""))

    meter = CostMeter(name, defense_type)
    total = len(src)
    for i, (_, row) in enumerate(src.iterrows(), 1):
        rid = str(row[key_col])
        if rid in done:
            continue
        with meter.request(rid):
            text = generate(client, build_raw(row), meter)   # <-- the defense (1 or many calls)
        rec = {c: row[c] for c in keep}
        rec["response"] = text
        _append(out, keep, rec)
        print(f"  [{i}/{total}] {rid} ok", flush=True)

    detail_p, summary_p = cost_files(out_dir, task, slug)
    meter.save(detail_p, summary_p)
    return out


# ----- XSTest over-refusal judge (shared scorer in ../metrics, LOCAL/API) -----
def _judge_xstest(responses_path, out_dir, name, slug):
    sys.path.insert(0, os.path.join(_ROOT, "metrics"))
    from xstest import evaluate as xstest_evaluate
    print("\n[eval] running XSTest judge (over-refusal)...")
    return xstest_evaluate(responses_path,
                           judged_file(out_dir, "xstest", slug),
                           method_name=name)


# ----- JustEval utility judge (LLM 5-aspect, shared scorer in ../metrics) -----
def _judge_justeval(responses_path, out_dir, name, slug):
    sys.path.insert(0, os.path.join(_ROOT, "metrics"))
    from justeval import evaluate as justeval_evaluate
    print("\n[judge] JustEval (utility, LLM 5-aspect)...")
    return justeval_evaluate(responses_path,
                             judged_file(out_dir, "justeval", slug),
                             method_name=name)


# ----- HarmBench ASR judge (classifier, needs a GPU) -----
def _judge_harmbench(responses_path, out_dir, name, slug, cls_model, batch_size):
    sys.path.insert(0, os.path.join(_ROOT, "metrics"))
    from harmbench import evaluate as harmbench_evaluate     # imports torch/transformers (GPU)
    print("\n[judge] HarmBench (ASR, classifier GPU)...")
    kw = {"batch_size": batch_size}
    if cls_model:
        kw["cls_model"] = cls_model
    return harmbench_evaluate(responses_path,
                              judged_file(out_dir, "harmbench", slug),
                              method_name=name, **kw)


def run_method(name, defense_type, model, out_dir, slug=None,
               transform_prompt=None, generate=None,
               temperature=0.0, max_tokens=512, cls_model=None, cls_batch_size=8,
               backend="groq", lora=None, local_kwargs=None, client_factory=None):
    """Two stages, each with --task {both|harmbench|xstest}:
         python method.py response [--task ...]   # sinh response (goi target)
         python method.py judge    [--task ...]   # cham diem (xstest=API, harmbench=GPU)

    Defense hook (mot trong hai):
      - transform_prompt(raw) -> wrapped   : single-call (SAGE, no_defense)
      - generate(client, raw, meter) -> text : multi-call, tu record_api (IA, G4D...)

    Target backend:
      - backend="groq"  (default) : GroqClient + key pool, nhu cu
      - backend="local"           : LocalClient (HF weights tren GPU) - cho in/intra
      - client_factory(model, temperature, max_tokens) -> client : method tu dung client
        (SafeDecoding can base+expert, JBShield can hook) - thang nay uu tien cao nhat.
      Moi client chi can co .chat(user, system=None) -> (text, resp).
    """
    slug = slug or name.lower()

    # single-call -> dung default generate bao quanh transform_prompt
    if generate is None:
        if transform_prompt is None:
            raise ValueError("run_method: can transform_prompt hoac generate")

        def generate(client, raw, meter, _tp=transform_prompt):
            text, resp = client.chat(_tp(raw))
            meter.record_api("target", resp)
            return text

    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["response", "judge"],
                    help="response = sinh response | judge = cham diem")
    ap.add_argument("--task", choices=["all", "harmbench", "xstest", "justeval"],
                    default="all", help="all = ca 3 metric (harmbench + xstest + justeval)")
    ap.add_argument("--limit", type=int, default=None, help="first N rows (smoke test)")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    tasks = ["harmbench", "xstest", "justeval"] if args.task == "all" else [args.task]
    os.makedirs(out_dir, exist_ok=True)

    # ----- STAGE 1: response (sinh, goi target: Groq pool hoac model local) -----
    if args.stage == "response":
        print(f"target = {model} | backend={backend} | temp={temperature} | max_tokens={max_tokens}")
        if client_factory is not None:
            client = client_factory(model=model, temperature=temperature, max_tokens=max_tokens)
        elif backend == "local":
            from .local_client import LocalClient
            client = LocalClient(model, lora=lora, temperature=temperature,
                                 max_tokens=max_tokens, **(local_kwargs or {}))
        else:
            keys = load_keys(out_dir)
            if not keys:
                sys.exit("No Groq keys found in .env (set GROQ_API_KEYS=key1,key2,... ).")
            print(f"key pool: {len(keys)} key(s) -> {[k[-4:] for k in keys]}")
            client = GroqClient(keys, model, temperature, max_tokens)
        for t in tasks:
            _run_task(t, client, generate, name, defense_type, slug, out_dir, args.limit)

    # ----- STAGE 2: judge (xstest = API judge; harmbench = GPU classifier) -----
    else:
        for t in tasks:
            resp = response_file(out_dir, t, slug)
            if not os.path.exists(resp):
                print(f"[skip] chua co {os.path.basename(resp)} -> chay 'response --task {t}' truoc.")
                continue
            if t == "xstest":
                _judge_xstest(resp, out_dir, name, slug)
            elif t == "justeval":
                _judge_justeval(resp, out_dir, name, slug)
            else:
                _judge_harmbench(resp, out_dir, name, slug, cls_model, cls_batch_size)
