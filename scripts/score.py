"""Scoring: pure functions for extraction and alignment, plus a CLI that scores every run record.

usage: score.py [--model M]   writes scores/<model>.jsonl, one row per request
"""

import argparse
import json
import re

from common import GEN_EXPS, RUNS, SCORES, append_jsonl, load_config, read_jsonl

WORDS = {w: str(i) for i, w in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"])}
WORD_OF = {d: w for w, d in WORDS.items()}


# ---------- extraction ----------

def extract_recall(text):
    """E1: the leading digit run of a raw continuation (stops at the first non-digit).

    Returns (strict, lenient): lenient also skips whitespace inside the run, so a model that
    wraps lines every 50 digits is not scored as stopping at 50.
    """
    strict = re.match(r"\d*", text).group(0)
    lenient = re.sub(r"\s", "", re.match(r"[\d\s]*", text).group(0))
    return strict, lenient


def extract_digits(text, int_part=None):
    """E3/E5: every digit in the output, after dropping a leading integer part.

    Models often drop the point ("314159..."), so "<int_part>" is stripped with or without it.
    This is unambiguous because the first decimal of pi (1) and e (7) differs from the integer part.
    """
    text = text.strip()
    if int_part is not None:
        text = re.sub(rf"^{re.escape(int_part)}\s*\.?", "", text)
    return re.sub(r"\D", "", text)


def extract_words(text, int_part=None):
    """E4/E6: map digit words to digits. Returns (digits, parse_errors).

    For E6, a leading "<integer word> point" (e.g. "three point") is dropped first.
    """
    toks = text.split()
    norm = [t.strip(".,;:!?\"'()[]").lower() for t in toks[:2]]
    if int_part is not None and norm[:2] == [WORD_OF[int_part], "point"]:
        toks = toks[2:]
    elif norm[:1] == ["point"]:
        toks = toks[1:]
    digits, errors = [], []
    for tok in toks:
        w = tok.strip(".,;:!?\"'()[]").lower()
        if w in WORDS:
            digits.append(WORDS[w])
        elif w:
            errors.append(tok)
    return "".join(digits), errors


# ---------- alignment ----------

def align(ref, hyp):
    """Levenshtein alignment of hyp against ref.

    Returns a list of (op, ref_pos, hyp_pos) with op in {"=", "S", "D", "I"}:
    match, substitution, deletion (ref digit missing from hyp), insertion (extra hyp digit).
    ref_pos for an insertion is the ref position it precedes. Common prefix and suffix are
    stripped first, so near-perfect copies of long strings stay cheap.
    """
    p = 0
    while p < len(ref) and p < len(hyp) and ref[p] == hyp[p]:
        p += 1
    s = 0
    while s < len(ref) - p and s < len(hyp) - p and ref[-1 - s] == hyp[-1 - s]:
        s += 1
    r, h = ref[p:len(ref) - s], hyp[p:len(hyp) - s]

    n, m = len(r), len(h)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri, row, prev = r[i - 1], dp[i], dp[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j - 1] + (ri != h[j - 1]), prev[j] + 1, row[j - 1] + 1)

    mid, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            mid.append(("=" if r[i - 1] == h[j - 1] else "S", p + i - 1, p + j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            mid.append(("D", p + i - 1, p + j))
            i -= 1
        else:
            mid.append(("I", p + i, p + j - 1))
            j -= 1
    mid.reverse()

    ops = [("=", k, k) for k in range(p)] + mid
    off_r, off_h = len(ref) - s, len(hyp) - s
    ops += [("=", off_r + k, off_h + k) for k in range(s)]
    return ops


def in_run(ref, pos, min_len=3):
    """True if ref position pos (or, at a boundary, its neighbour) lies in a run of >= min_len equal digits."""
    def run_len(k):
        if not 0 <= k < len(ref):
            return 0
        a = b = k
        while a > 0 and ref[a - 1] == ref[k]:
            a -= 1
        while b < len(ref) - 1 and ref[b + 1] == ref[k]:
            b += 1
        return b - a + 1
    return run_len(pos) >= min_len or run_len(pos - 1) >= min_len


def prefix_len(ref, hyp):
    """Number of leading digits of hyp that match ref: the position of the first wrong digit."""
    n = 0
    for a, b in zip(ref, hyp):
        if a != b:
            break
        n += 1
    return n


def copy_metrics(ref, hyp):
    ops = align(ref, hyp)
    errors = [{"op": op, "ref_pos": rp, "in_run": in_run(ref, rp)} for op, rp, _ in ops if op != "="]
    n_s = sum(e["op"] == "S" for e in errors)
    n_i = sum(e["op"] == "I" for e in errors)
    n_d = sum(e["op"] == "D" for e in errors)
    return {
        "ref_len": len(ref), "hyp_len": len(hyp),
        "prefix": prefix_len(ref, hyp),
        "edit_distance": len(errors), "subs": n_s, "ins": n_i, "dels": n_d,
        "errors_per_100": 100 * len(errors) / len(ref),
        "errors_in_run": sum(e["in_run"] for e in errors),
        "exact": len(errors) == 0,
        "errors": errors,
    }, ops


def planted_outcome(ops, hyp, pos, true_digit, planted_digit):
    """E5: what the model wrote at the planted position."""
    for op, rp, hp in ops:
        if rp == pos and op in ("=", "S"):
            d = hyp[hp]
            return "planted" if d == planted_digit else "memory" if d == true_digit else "other"
        if rp == pos and op == "D":
            return "other"
    return "other"


# ---------- per-record scoring ----------

def score_record(rec, inp, ref_output=None):
    """Score one run record against its input. ref_output: the reference variant's raw output."""
    out = {k: rec[k] for k in ("model", "variant", "bpw", "exp", "input_id", "finish_reason", "tokens_out")}
    out["truncated"] = rec["finish_reason"] == "length"
    out["meta"] = inp.get("meta", {})
    ref = inp["ref"]
    exp = rec["exp"]

    if exp == "E1":
        strict, lenient = extract_recall(rec["output"])
        out.update(prefix=prefix_len(ref, strict), prefix_lenient=prefix_len(ref, lenient),
                   hyp_len=len(strict), stop_char=rec["output"][len(strict):len(strict) + 1])
        if ref_output is not None:
            out["agree_ref_prefix"] = prefix_len(extract_recall(ref_output)[0], strict)
        return out

    if exp in ("E4", "E4b", "E6"):
        hyp, parse_errors = extract_words(rec["output"], inp.get("int_part"))
        out["parse_errors"] = len(parse_errors)
        out["parse_error_sample"] = parse_errors[:5]
    else:
        hyp = extract_digits(rec["output"], inp.get("int_part"))

    if exp == "E6":
        # recall: only the prefix is meaningful; after the first error the model is off-script
        out.update(prefix=prefix_len(ref, hyp), hyp_len=len(hyp))
        return out

    metrics, ops = copy_metrics(ref, hyp)
    out.update(metrics)
    if exp == "E5":
        m = inp["meta"]
        out["outcome"] = planted_outcome(ops, hyp, m["pos"], m["true_digit"], m["planted_digit"])
    return out


def score_model(model, cfg):
    reference = cfg["models"][model]["reference"]
    inputs = {}
    for e in GEN_EXPS:
        for inp in read_jsonl(RUNS.parent / "inputs" / f"{e}.jsonl"):
            inputs[(e, inp["input_id"])] = inp
    ref_runs = {}
    for e in ("E1",):
        for r in read_jsonl(RUNS / model / reference / f"{e}.jsonl"):
            ref_runs[(e, r["input_id"])] = r["output"]

    out_path = SCORES / f"{model}.jsonl"
    out_path.unlink(missing_ok=True)
    n = 0
    for vdir in sorted(d for d in (RUNS / model).iterdir() if d.is_dir()):
        for e in GEN_EXPS:
            for rec in read_jsonl(vdir / f"{e}.jsonl"):
                row = score_record(rec, inputs[(e, rec["input_id"])], ref_runs.get((e, rec["input_id"])))
                append_jsonl(out_path, row)
                n += 1
    print(f"{model}: scored {n} records -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    args = ap.parse_args()
    cfg = load_config()
    for model in [args.model] if args.model else cfg["models"]:
        if (RUNS / model).exists():
            score_model(model, cfg)


if __name__ == "__main__":
    main()
