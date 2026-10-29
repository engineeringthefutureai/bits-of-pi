"""E2: teacher-forced pass with tools/bin/tf_dump, reduced to per-digit-position records.

Full log-probabilities ([n_tokens, n_vocab] float32, ~1 GB per text) are written to
models/<model>/tf_raw/<variant>/ and deleted after reduction, except for the reference
variant, whose distributions every other variant's KL divergence is measured against.
"""

import argparse
import json
import shutil
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common import INPUTS, MODELS, ROOT, TF, backend_env, load_config, read_jsonl

CHUNK = 64


def raw_dir(model, variant):
    return MODELS / model / "tf_raw" / variant


def dump(cfg, model_file, text, prefix, device=None):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    txt = prefix.with_suffix(".txt")
    txt.write_text(text)
    subprocess.run([str(ROOT / "tools/bin/tf_dump"), "-m", str(model_file), "-f", str(txt), "-o", str(prefix)],
                   env=backend_env(cfg, device), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load(prefix):
    meta = json.loads(prefix.with_suffix(".tokens.json").read_text())
    lp = np.memmap(prefix.with_suffix(".logprobs.f32"), dtype=np.float32, mode="r",
                   shape=(len(meta["tokens"]), meta["n_vocab"]))
    return meta, lp


def digit_token_index(pieces, header, digits):
    """For each digit k, the token index that starts exactly at it and is exactly that digit, else -1."""
    starts, pos = {}, 0
    for t, piece in enumerate(pieces):
        starts[pos] = (t, piece)
        pos += len(piece)
    idx = np.full(len(digits), -1)
    for k, d in enumerate(digits):
        t_piece = starts.get(len(header) + k)
        if t_piece and t_piece[1] == d:
            idx[k] = t_piece[0]
    return idx


def reduce_text(inp, prefix, ref_prefix):
    meta, lp = load(prefix)
    tok_idx = digit_token_index(meta["pieces"], inp["header"], inp["digits"])
    ref_lp = None
    if ref_prefix is not None:
        ref_meta, ref_lp = load(ref_prefix)
        assert ref_meta["tokens"] == meta["tokens"], "tokenization differs from the reference"

    ok = np.where(tok_idx > 0)[0]
    tokens = np.array(meta["tokens"])
    cols = {k: np.full(len(inp["digits"]), np.nan) for k in
            ("logprob", "rank", "margin", "top1_token", "kl_vs_ref", "top1_matches_ref")}
    for c0 in range(0, len(ok), CHUNK):
        ks = ok[c0:c0 + CHUNK]
        rows = np.asarray(lp[tok_idx[ks] - 1], dtype=np.float64)
        correct = tokens[tok_idx[ks]]
        lpc = rows[np.arange(len(ks)), correct]
        top2 = np.argpartition(-rows, 1, axis=1)[:, :2]
        v2 = rows[np.arange(len(ks))[:, None], top2]
        top1 = top2[np.arange(len(ks)), v2.argmax(1)]
        # margin = correct minus the best competitor: runner-up if correct is top-1, else top-1
        best_other = np.where(top1 == correct, v2.min(1), v2.max(1))
        cols["logprob"][ks] = lpc
        cols["rank"][ks] = (rows > lpc[:, None]).sum(1)
        cols["margin"][ks] = lpc - best_other
        cols["top1_token"][ks] = top1
        if ref_lp is not None:
            ref_rows = np.asarray(ref_lp[tok_idx[ks] - 1], dtype=np.float64)
            cols["kl_vs_ref"][ks] = (np.exp(ref_rows) * (ref_rows - rows)).sum(1)
            cols["top1_matches_ref"][ks] = ref_rows.argmax(1) == top1

    table = pa.table({"pos": np.arange(len(inp["digits"])), "digit": list(inp["digits"]),
                      "scorable": tok_idx > 0, **cols})
    return table, int((tok_idx <= 0).sum())


def run(cfg, meta, model_file, device=None, log=print):
    model, variant = meta["model"], meta["variant"]
    reference = cfg["models"][model]["reference"]
    out_dir = TF / model / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    for inp in read_jsonl(INPUTS / "E2.jsonl"):
        out = out_dir / f"{inp['input_id']}.parquet"
        if out.exists():
            continue
        prefix = raw_dir(model, variant) / inp["input_id"]
        if not prefix.with_suffix(".logprobs.f32").exists():
            dump(cfg, model_file, inp["header"] + inp["digits"], prefix, device)
        ref_prefix = None if variant == reference else raw_dir(model, reference) / inp["input_id"]
        if ref_prefix is not None and not ref_prefix.with_suffix(".logprobs.f32").exists():
            raise FileNotFoundError(f"reference logprobs missing: run the {reference} variant first")
        table, n_bad = reduce_text(inp, prefix, ref_prefix)
        pq.write_table(table, out)
        log(f"[{variant}] E2 {inp['input_id']}: {table.num_rows} positions, {n_bad} unscorable")
    if variant != reference:
        shutil.rmtree(raw_dir(model, variant), ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--device")
    args = ap.parse_args()
    cfg = load_config()
    meta = json.loads((ROOT / "runs" / args.model / args.variant / "meta.json").read_text())
    run(cfg, meta, args.file, args.device)


if __name__ == "__main__":
    main()
