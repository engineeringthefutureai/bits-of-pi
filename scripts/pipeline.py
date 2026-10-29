"""Download or quantize -> E0 -> generate -> teacher-forced pass -> delete, one variant at a time.

usage:
  pipeline.py --model qwen3.5-4b                     # every variant in config order
  pipeline.py --model qwen3.5-4b --variants BF16,Q8_0
  pipeline.py --model qwen3.5-4b --device level_zero:1 --port 18081 --variants ...   # second GPU

The reference variant runs first and its GGUF is kept (it is the source for every DIY quant and
the E2 baseline). Every other model file is deleted after its runs, unless --keep.
Completed work is skipped, so the pipeline can be stopped and restarted at any point.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys

from huggingface_hub import hf_hub_download

import e0
import run_gen
import run_tf
from common import (DATA, GEN_EXPS, INPUTS, MODELS, RUNS, TF, ResourceSampler, backend_env, bin_path,
                    load_config, model_n_params, read_jsonl)


def log(msg):
    print(f"{dt.datetime.now():%H:%M:%S} {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def hf_file(mcfg, model, filename):
    dest = MODELS / model / "hf"
    return hf_hub_download(mcfg["hf_repo"], filename, local_dir=dest)


def reference_file(cfg, model):
    mcfg = cfg["models"][model]
    ref = next(v for v in mcfg["variants"] if v["name"] == mcfg["reference"])
    return MODELS / model / "hf" / ref["file"]


def imatrix_file(cfg, model, kind, device):
    mcfg = cfg["models"][model]
    if kind == "unsloth":
        src = hf_file(mcfg, model, mcfg["unsloth_imatrix"])
        link = MODELS / model / "imatrix_unsloth.gguf"
        if not link.exists():
            link.symlink_to(os.path.abspath(src))
        return link
    path = MODELS / model / "imatrix.gguf"
    if not path.exists():
        c = cfg["calibration"]
        log(f"computing imatrix for {model} from data/calibration.txt")
        subprocess.run([str(bin_path(cfg, "llama-imatrix")), "-m", str(reference_file(cfg, model)),
                        "-f", str(DATA / "calibration.txt"), "-o", str(path),
                        "--chunks", str(c["chunks"]), "-c", str(c["ctx"]), "-ngl", "999"],
                       env=backend_env(cfg, device), check=True,
                       stdout=open(MODELS / model / "imatrix.log", "w"), stderr=subprocess.STDOUT)
    return path


def ensure_file(cfg, model, v, device):
    mcfg = cfg["models"][model]
    if v["source"] == "hf":
        return hf_file(mcfg, model, v["file"]), {"hf_repo": mcfg["hf_repo"], "hf_file": v["file"]}
    out = MODELS / model / "quants" / f"{v['name']}.gguf"
    info = {"quant_type": v["type"], "imatrix": v.get("imatrix", "none")}
    cmd = [str(bin_path(cfg, "llama-quantize"))]
    if v.get("imatrix", "none") != "none":
        im = imatrix_file(cfg, model, v["imatrix"], device)
        cmd += ["--imatrix", str(im)]
        info["imatrix_sha256"] = sha256(im)
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        log(f"[{v['name']}] quantizing {v['type']}")
        subprocess.run(cmd + [str(reference_file(cfg, model)), str(tmp), v["type"]],
                       env=backend_env(cfg, device), check=True,
                       stdout=open(out.with_suffix(".log"), "w"), stderr=subprocess.STDOUT)
        tmp.rename(out)
    return out, info


def bench(cfg, model_file, out_path, device):
    """llama-bench with the server's arithmetic settings: prompt processing at 512 and 4096 tokens,
    decode of 128 tokens, 3 repetitions each. Server timings on short prompts understate SYCL
    throughput (fixed per-request overhead), so speed is reported from this instead."""
    out = subprocess.run([str(bin_path(cfg, "llama-bench")), "-m", str(model_file), "-ngl", "999",
                          "-p", "512,4096", "-n", "128", "-r", "3", "-b", "2048", "-ub", "512",
                          "-ctk", "f16", "-ctv", "f16", "-fa", "off", "-o", "json"],
                         env=backend_env(cfg, device), capture_output=True, text=True, check=True).stdout
    rows = json.loads(out)
    keep = ("n_prompt", "n_gen", "avg_ts", "stddev_ts", "model_size", "model_n_params", "backends", "gpu_info", "build_commit")
    out_path.write_text(json.dumps([{k: r.get(k) for k in keep} for r in rows], indent=1) + "\n")
    return rows


def is_done(cfg, model, v):
    vdir = RUNS / model / v["name"]
    if not (vdir / "meta.json").exists() or not (vdir / "bench.json").exists():
        return False
    for e in GEN_EXPS:
        if len(read_jsonl(vdir / f"{e}.jsonl")) < len(read_jsonl(INPUTS / f"{e}.jsonl")):
            return False
    n_tf = len(read_jsonl(INPUTS / "E2.jsonl"))
    return len(list((TF / model / v["name"]).glob("*.parquet"))) == n_tf


def write_meta(cfg, model, v, path, info):
    meta_path = RUNS / model / v["name"] / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    params_path = RUNS / model / "params.json"
    if not params_path.exists():
        params_path.parent.mkdir(parents=True, exist_ok=True)
        params_path.write_text(json.dumps({"n_params": model_n_params(reference_file(cfg, model))}) + "\n")
    n_params = json.loads(params_path.read_text())["n_params"]
    size = os.path.getsize(path)
    meta = {"model": model, "variant": v["name"], "group": v["group"], "source": v["source"],
            "file_size": size, "n_params": n_params, "bpw": round(size * 8 / n_params, 3),
            "sha256": sha256(path), "llama_cpp_commit": cfg["llama_cpp"]["commit"],
            "backend": cfg["llama_cpp"]["backend"], **info}
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def process(cfg, model, v, args):
    name = v["name"]
    reference = cfg["models"][model]["reference"]
    if is_done(cfg, model, v) and not (name in (reference, "Q8_0")
                                       and not (RUNS / model / name / "E0.json").exists()):
        log(f"[{name}] already complete")
        return
    path, info = ensure_file(cfg, model, v, args.device)
    meta = write_meta(cfg, model, v, path, info)
    log(f"[{name}] {meta['bpw']} bpw, {meta['file_size'] / 1e9:.2f} GB")

    if name in (reference, "Q8_0") and not (RUNS / model / name / "E0.json").exists():
        if not e0.run(cfg, meta, path, args.device, log) and not args.ignore_e0:
            sys.exit(f"E0 failed for {name}; see runs/{model}/{name}/E0.json")

    vdir = RUNS / model / name
    device = args.device or cfg["llama_cpp"]["device"]
    if run_gen.pending(model, name, GEN_EXPS):
        server_info = {}
        csv_name = "resources_rerun.csv" if (vdir / "resources.json").exists() else "resources.csv"
        with ResourceSampler(device, vdir / csv_name) as sampler:
            run_gen.run(cfg, meta, path, GEN_EXPS, args.device, log,
                        on_stop=lambda srv: server_info.update(peak_rss_mib=srv.peak_rss_mib))
        res = {"device": device, **sampler.summary(), "host_peak_rss_mib": server_info.get("peak_rss_mib")}
        res_name = "resources_rerun.json" if (vdir / "resources.json").exists() else "resources.json"
        (vdir / res_name).write_text(json.dumps(res, indent=1) + "\n")
        log(f"[{name}] resources: VRAM +{res['vram_delta_mib']} MiB (peak {res['vram_peak_mib']}), "
            f"host RSS {res['host_peak_rss_mib']} MiB, power {res['power_mean_w']} W")
    if not args.skip_tf:
        run_tf.run(cfg, meta, path, args.device, log)
    if not (vdir / "bench.json").exists():
        rows = bench(cfg, path, vdir / "bench.json", device)
        log(f"[{name}] bench: " + ", ".join(
            f"{'pp' + str(r['n_prompt']) if r['n_prompt'] else 'tg' + str(r['n_gen'])} {r['avg_ts']:.0f} t/s" for r in rows))

    if name != reference and not args.keep:
        os.remove(path)
        log(f"[{name}] deleted {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", help="comma-separated subset, in config order by default")
    ap.add_argument("--device", help="ONEAPI_DEVICE_SELECTOR, e.g. level_zero:1")
    ap.add_argument("--port", type=int)
    ap.add_argument("--keep", action="store_true", help="keep model files after their runs")
    ap.add_argument("--skip-tf", action="store_true")
    ap.add_argument("--ignore-e0", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    if args.port:
        cfg["server"]["port"] = args.port
    variants = cfg["models"][args.model]["variants"]
    if args.variants:
        wanted = args.variants.split(",")
        unknown = set(wanted) - {v["name"] for v in variants}
        if unknown:
            sys.exit(f"unknown variants: {sorted(unknown)}")
        variants = [v for v in variants if v["name"] in wanted]
    reference = cfg["models"][args.model]["reference"]
    variants.sort(key=lambda v: v["name"] != reference)
    for v in variants:
        process(cfg, args.model, v, args)


if __name__ == "__main__":
    main()
