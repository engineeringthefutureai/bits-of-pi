"""Follow-up S: sampled recitations of π against the forecast from per-digit certainty.

For each variant: a teacher-forced pass over the E2 π text keeps the top candidates at every
digit (tf/<model>/<variant>/sampling/pi_top.parquet), then `n` samples per sampler setting are drawn from
the same header with /completion (runs/<model>/<variant>/S.jsonl). A sample is streamed and cut at
its first wrong digit, since nothing after it is scored.

usage: sampling.py --model M [--variants a,b] [--device level_zero:1] [--keep]
"""

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from common import INPUTS, RUNS, TF, Server, append_jsonl, load_config, read_jsonl
from pipeline import delete_quant_source, ensure_file, log, reference_file
from run_tf import digit_token_index, dump, load, raw_dir


def pi_input():
    return next(i for i in read_jsonl(INPUTS / "E2.jsonl") if i["input_id"] == "pi")


def top_candidates(cfg, model, variant, path, device):
    """Top-n log-probabilities at each digit position, from a fresh dump (or the kept reference one)."""
    out = TF / model / variant / "sampling" / "pi_top.parquet"   # subfolder: is_done counts E2 files
    if out.exists():
        return
    inp, n = pi_input(), cfg["sampling"]["top_n"]
    prefix = raw_dir(model, variant) / "pi"
    fresh = not prefix.with_suffix(".tokens.json").exists()
    if fresh:
        prefix = raw_dir(model, variant + "_top") / "pi"
        dump(cfg, path, inp["header"] + inp["digits"], prefix, device)
    meta, lp = load(prefix)
    idx = digit_token_index(meta["pieces"], inp["header"], inp["digits"])
    tokens = np.array(meta["tokens"])
    ids = np.full((len(idx), n), -1, dtype=np.int64)
    lps = np.full((len(idx), n), np.nan)
    correct = np.full(len(idx), -1, dtype=np.int64)
    for k, t in enumerate(idx):
        if t <= 0:
            continue
        row = np.asarray(lp[t - 1], dtype=np.float64)
        top = np.argpartition(-row, n)[:n]
        top = top[np.argsort(-row[top])]
        ids[k], lps[k], correct[k] = top, row[top], tokens[t]
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"pos": np.arange(len(idx)), "correct_token": correct,
                             "top_ids": list(ids), "top_logprobs": list(lps)}), out)
    if fresh:
        shutil.rmtree(prefix.parent, ignore_errors=True)
    log(f"[{variant}] top-{n} candidates at {int((idx > 0).sum())} digit positions")


def sample_one(url, header, digits, body):
    """Stream one sample; stop at the first character that is not the next digit of π."""
    text = ""
    with requests.post(url + "/completion", json={**body, "prompt": header, "stream": True},
                       stream=True, timeout=3600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data: "):
                continue
            d = json.loads(line[6:])
            text += d.get("content", "")
            k = next((i for i, c in enumerate(text) if i >= len(digits) or c != digits[i]), None)
            if k is not None or d.get("stop"):
                return text, k if k is not None else len(text)
    return text, len(text)


def samples_todo(cfg, model, variant):
    s = cfg["sampling"]
    done = {(r["setting"], r["seed"]) for r in read_jsonl(RUNS / model / variant / "S.jsonl")}
    return [(name, seed) for name in s["settings"] for seed in range(1, s["n"] + 1) if (name, seed) not in done]


def run_samples(cfg, model, variant, path, device):
    s = cfg["sampling"]
    out = RUNS / model / variant / "S.jsonl"
    todo = samples_todo(cfg, model, variant)
    if not todo:
        return
    inp = pi_input()
    cfg = {**cfg, "server": {**cfg["server"], "args": list(cfg["server"]["args"])}}
    a = cfg["server"]["args"]
    a[a.index("--parallel") + 1] = str(s["parallel"])
    a[a.index("--ctx-size") + 1] = str(s["parallel"] * 1100)

    def job(item):
        name, seed = item
        body = {**cfg["request"], **s["settings"][name], "min_p": 0, "seed": seed,
                "n_predict": len(inp["digits"]) + 5}
        text, k = sample_one(srv.url, inp["header"], inp["digits"], body)
        return {"model": model, "variant": variant, "setting": name, "seed": seed,
                "params": body, "output": text, "prefix": k}

    with Server(cfg, path, device, log_path=RUNS / model / variant / "server_sampling.log") as srv:
        with ThreadPoolExecutor(s["parallel"]) as pool:
            for k, rec in enumerate(pool.map(job, todo), 1):
                append_jsonl(out, rec)
                if k % 16 == 0 or k == len(todo):
                    log(f"[{variant}] {k}/{len(todo)} samples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants")
    ap.add_argument("--device")
    ap.add_argument("--port", type=int)
    ap.add_argument("--keep", action="store_true", help="keep DIY quant files")
    args = ap.parse_args()
    cfg = load_config()
    if args.port:
        cfg["server"]["port"] = args.port
    wanted = args.variants.split(",") if args.variants else cfg["sampling"]["variants"][args.model]
    variants = {v["name"]: v for v in cfg["models"][args.model]["variants"]}
    for name in wanted:
        v = variants[name]
        if (TF / args.model / name / "sampling" / "pi_top.parquet").exists() and not samples_todo(cfg, args.model, name):
            log(f"[{name}] sampling already complete")
            continue
        path, _ = ensure_file(cfg, args.model, v, args.device)
        top_candidates(cfg, args.model, name, path, args.device)
        run_samples(cfg, args.model, name, path, args.device)
        if v["source"] == "quantize" and not args.keep:
            os.remove(path)
            log(f"[{name}] deleted {path}")
    delete_quant_source(cfg, args.model)


if __name__ == "__main__":
    main()
