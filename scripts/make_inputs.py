"""Generate reference digits, random strings, the calibration text and every experiment input.

Run once; outputs are committed. Re-running with the same seed reproduces them byte for byte.
"""

import hashlib
import json
import random
import re

import mpmath

from common import DATA, INPUTS, ROOT, load_config

N_REF = 1200

CONSTANTS = {
    "pi":    {"int": "3", "value": lambda: mpmath.pi,
              "e1": ["Pi to 1000 decimal places:\n3.", "The decimal expansion of pi is 3.", "π = 3."],
              "e2_header": "Pi to 1000 decimal places: 3.", "name": "pi"},
    "e":     {"int": "2", "value": lambda: mpmath.e,
              "e1": ["e to 1000 decimal places:\n2.", "The decimal expansion of e is 2.", "e = 2."],
              "e2_header": "e to 1000 decimal places: 2.", "name": "e (Euler's number)"},
    "sqrt2": {"int": "1", "value": lambda: mpmath.sqrt(2),
              "e1": ["The square root of 2 to 1000 decimal places:\n1.",
                     "The decimal expansion of the square root of 2 is 1.", "√2 = 1."],
              "e2_header": "The square root of 2 to 1000 decimal places: 1.", "name": "the square root of 2"},
}

COPY_PROMPT = ("Repeat the following digits exactly. Output only the digits, "
               "with no spaces or other text.\n\n{digits}")
WORDS_PROMPT = ("Write each of the following digits as an English word (zero, one, two, …), "
                "separated by single spaces. Output nothing else.\n\n{digits}")
# E4b: E4's prompt is read by Qwen 3.5 4B as "say this number" ("twenty trillion two hundred...");
# this version asks for one word per digit explicitly and shows an example.
WORDS_PROMPT_B = ("Spell out each digit below separately as an English word (zero, one, two, three, four, five, "
                  "six, seven, eight, nine), one word per digit, separated by single spaces. "
                  "Do not read the digits as a number. For example, 507 becomes: five zero seven. "
                  "Output nothing else.\n\n{digits}")
RECALL_WORDS_PROMPT = ("Write the first {n} decimal places of {name} (the digits after \"{int}.\") "
                       "as English words (zero, one, two, …), separated by single spaces. Output nothing else.")


def decimals(value_fn, n):
    """First n decimal digits after the point, computed with guard digits and checked at two precisions."""
    out = []
    for dps in (n + 30, n + 80):
        mpmath.mp.dps = dps
        s = mpmath.nstr(value_fn(), dps - 5, strip_zeros=False)
        out.append(s.split(".")[1][:n])
    assert out[0] == out[1], "precision mismatch"
    return out[0]


def random_digits(rng, n):
    return "".join(rng.choice("0123456789") for _ in range(n))


def run_heavy(rng, n, n_runs=8):
    """Random digits with n_runs runs of 4-7 identical digits inserted at random places."""
    runs = [str(rng.randrange(10)) * rng.randint(4, 7) for _ in range(n_runs)]
    base = random_digits(rng, n - sum(map(len, runs)))
    cuts = sorted(rng.sample(range(1, len(base)), n_runs))
    parts, prev = [], 0
    for cut, run in zip(cuts, runs):
        parts += [base[prev:cut], run]
        prev = cut
    parts.append(base[prev:])
    s = "".join(parts)
    assert len(s) == n
    return s


def copy_tokens(n):
    return int(1.3 * n) + 50


def word_tokens(n):
    return 3 * n + 50


def main():
    cfg = load_config()
    rng = random.Random(cfg["seed"])
    DATA.mkdir(exist_ok=True)
    INPUTS.mkdir(exist_ok=True)

    refs = {k: decimals(c["value"], N_REF) for k, c in CONSTANTS.items()}
    assert refs["pi"].startswith("14159265358979323846")
    assert refs["e"].startswith("71828182845904523536")
    assert refs["sqrt2"].startswith("41421356237309504880")
    for k, v in refs.items():
        (DATA / f"ref_{k}.txt").write_text(v + "\n")

    strings = {}
    for n in (100, 300, 1000):
        for i in range(10):
            strings[f"len{n}_s{i:02d}"] = random_digits(rng, n)
    for i in range(10):
        strings[f"run300_s{i:02d}"] = run_heavy(rng, 300)
    tf_random = random_digits(rng, 1000)
    (DATA / "random_strings.json").write_text(json.dumps(
        {"seed": cfg["seed"], "strings": strings, "tf_random": tf_random}, indent=1) + "\n")

    exps = {e: [] for e in ("E1", "E2", "E3", "E4", "E4b", "E5", "E6")}

    for k, c in CONSTANTS.items():
        for j, prompt in enumerate(c["e1"]):
            exps["E1"].append({"exp": "E1", "input_id": f"{k}_p{j}", "endpoint": "completion",
                               "prompt": prompt, "ref": refs[k], "int_part": c["int"],
                               "n_predict": 1200, "meta": {"constant": k, "phrasing": j}})

    for k, c in CONSTANTS.items():
        exps["E2"].append({"exp": "E2", "input_id": k, "header": c["e2_header"], "digits": refs[k][:1000]})
    exps["E2"].append({"exp": "E2", "input_id": "random", "header": "1000 random digits: ", "digits": tf_random})

    for sid, s in strings.items():
        exps["E3"].append({"exp": "E3", "input_id": sid, "endpoint": "chat",
                           "prompt": COPY_PROMPT.format(digits=s), "ref": s, "int_part": None,
                           "n_predict": copy_tokens(len(s)), "meta": {"length": len(s), "run_heavy": sid.startswith("run")}})
        if not sid.startswith("len1000"):
            exps["E4"].append({"exp": "E4", "input_id": sid, "endpoint": "chat",
                               "prompt": WORDS_PROMPT.format(digits=s), "ref": s, "int_part": None,
                               "n_predict": word_tokens(len(s)), "meta": {"length": len(s), "run_heavy": sid.startswith("run")}})
            exps["E4b"].append({"exp": "E4b", "input_id": sid, "endpoint": "chat",
                                "prompt": WORDS_PROMPT_B.format(digits=s), "ref": s, "int_part": None,
                                "n_predict": word_tokens(len(s)), "meta": {"length": len(s), "run_heavy": sid.startswith("run")}})

    for k in ("pi", "e"):
        c = CONSTANTS[k]
        assert refs[k][0] != c["int"], "integer-part stripping in score.py relies on this"
        true = refs[k][:300]
        for pos in (10, 50, 150, 290):
            planted = str((int(true[pos]) + 5) % 10)
            seq = true[:pos] + planted + true[pos + 1:]
            exps["E5"].append({"exp": "E5", "input_id": f"{k}_p{pos:03d}", "endpoint": "chat",
                               "prompt": COPY_PROMPT.format(digits=f"{c['int']}.{seq}"), "ref": seq,
                               "int_part": c["int"], "n_predict": copy_tokens(301),
                               "meta": {"constant": k, "pos": pos, "true_digit": true[pos], "planted_digit": planted}})

    n6 = 500
    for k, c in CONSTANTS.items():
        exps["E6"].append({"exp": "E6", "input_id": k, "endpoint": "chat",
                           "prompt": RECALL_WORDS_PROMPT.format(n=n6, name=c["name"], int=c["int"]),
                           "ref": refs[k][:n6], "int_part": c["int"], "n_predict": word_tokens(n6),
                           "meta": {"constant": k}})

    for e, rows in exps.items():
        with open(INPUTS / f"{e}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{e}: {len(rows)} inputs")

    make_calibration(cfg)


def make_calibration(cfg):
    """Generic calibration text with no digits of the constants and no digit runs of 3 or more."""
    c = cfg["calibration"]
    src = (ROOT / c["source"]).read_text(encoding="utf-8")
    kept, total = [], 0
    for line in src.splitlines():
        if not line.strip() or re.search(r"\d{3,}|\d[.,]\d", line):
            continue
        kept.append(line)
        total += len(line) + 1
        if total >= c["max_chars"]:
            break
    text = "\n".join(kept) + "\n"
    assert not re.search(r"\d{3,}", text)
    for needle in ("14159", "71828", "41421", "3.14", "2.718", "1.414"):
        assert needle not in text
    (DATA / "calibration.txt").write_text(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    (DATA / "calibration.sha256").write_text(f"{sha}  calibration.txt\n")
    print(f"calibration: {len(text)} chars, sha256 {sha}")


if __name__ == "__main__":
    main()
