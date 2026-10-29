"""E0 sanity checks for one model file: digit tokenization, thinking off, determinism.

Writes runs/<model>/<variant>/E0.json. The Q8_0-vs-BF16 check is read from the scored E1 runs.
"""

import json
import subprocess

from common import INPUTS, RUNS, Server, backend_env, bin_path, read_jsonl


def tokenize_check(cfg, model_file, device):
    pi = (RUNS.parent / "data" / "ref_pi.txt").read_text().strip()[:200]
    text = "3." + pi
    out = subprocess.run([str(bin_path(cfg, "llama-tokenize")), "-m", str(model_file), "-p", text,
                          "--ids", "--log-disable"],
                         env=backend_env(cfg, device), capture_output=True, text=True, check=True).stdout
    ids = json.loads(out.strip().splitlines()[-1])
    return {"text_chars": len(text), "n_tokens": len(ids), "token_ids": ids,
            "single_digit_tokens": len(ids) == len(text) or len(ids) == len(text) + 1}


def run(cfg, meta, model_file, device=None, log=print):
    device = device or cfg["llama_cpp"]["device"]
    out_dir = RUNS / meta["model"] / meta["variant"]
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"tokenize": tokenize_check(cfg, model_file, device)}

    e1 = read_jsonl(INPUTS / "E1.jsonl")[0]
    e3 = read_jsonl(INPUTS / "E3.jsonl")[0]
    outputs = {"E1": [], "E3": []}
    thinking = None
    for session in range(2):
        with Server(cfg, model_file, device, log_path=out_dir / "server_e0.log") as srv:
            srv.chat("Say OK.", 4)
            for _ in range(2 if session == 0 else 1):
                outputs["E1"].append(srv.run(e1)[1]["output"])
                res = srv.run(e3)[1]
                outputs["E3"].append(res["output"])
                if thinking is None:
                    thinking = {"reasoning_empty": res["reasoning"] == "",
                                "no_think_tag": "<think>" not in res["output"],
                                "output_head": res["output"][:120]}
    result["thinking_off"] = thinking
    result["determinism"] = {e: {"identical": len(set(o)) == 1, "n_runs": len(o)} for e, o in outputs.items()}
    result["pass"] = (result["tokenize"]["single_digit_tokens"]
                      and thinking["reasoning_empty"] and thinking["no_think_tag"]
                      and all(d["identical"] for d in result["determinism"].values()))
    (out_dir / "E0.json").write_text(json.dumps(result, indent=1) + "\n")
    log(f"[{meta['variant']}] E0 {'PASS' if result['pass'] else 'FAIL'}: "
        f"tokens={result['tokenize']['n_tokens']}/{result['tokenize']['text_chars']} chars, "
        f"thinking={thinking}, determinism={result['determinism']}")
    return result["pass"]
