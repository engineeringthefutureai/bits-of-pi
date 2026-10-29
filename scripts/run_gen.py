"""Generation experiments (E1, E3-E6) for one model file, against one warm llama-server.

Resumable: inputs whose record already exists in runs/<model>/<variant>/<exp>.jsonl are skipped.
Records are only ever appended, never edited.
"""

import argparse
import json

from common import GEN_EXPS, INPUTS, RUNS, Server, append_jsonl, load_config, read_jsonl


def pending(model, variant, exps):
    todo = []
    for e in exps:
        done = {r["input_id"] for r in read_jsonl(RUNS / model / variant / f"{e}.jsonl")}
        todo += [i for i in read_jsonl(INPUTS / f"{e}.jsonl") if i["input_id"] not in done]
    return todo


def make_record(cfg, meta, inp, body, res, device):
    return {
        "model": meta["model"], "variant": meta["variant"], "bpw": meta["bpw"],
        "exp": inp["exp"], "input_id": inp["input_id"],
        "prompt": inp["prompt"], "params": body,
        "output": res["output"], "reasoning": res.get("reasoning", ""),
        "tokens_in": res["tokens_in"], "tokens_out": res["tokens_out"], "finish_reason": res["finish_reason"],
        "timings": res.get("timings"),
        "llama_cpp_commit": cfg["llama_cpp"]["commit"], "backend": cfg["llama_cpp"]["backend"], "device": device,
    }


def run(cfg, meta, model_file, exps=GEN_EXPS, device=None, log=print, on_stop=None):
    model, variant = meta["model"], meta["variant"]
    todo = pending(model, variant, exps)
    if not todo:
        log(f"[{variant}] generation: nothing to do")
        return
    device = device or cfg["llama_cpp"]["device"]
    out_dir = RUNS / model / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    with Server(cfg, model_file, device, log_path=out_dir / "server.log") as srv:
        srv.chat("Say OK.", 4)  # throwaway warm-up request
        for k, inp in enumerate(todo, 1):
            body, res = srv.run(inp)
            append_jsonl(out_dir / f"{inp['exp']}.jsonl", make_record(cfg, meta, inp, body, res, device))
            log(f"[{variant}] {k}/{len(todo)} {inp['exp']} {inp['input_id']} "
                f"out={res['tokens_out']} {res['finish_reason']}")
    if on_stop:
        on_stop(srv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--exps", default=",".join(GEN_EXPS))
    ap.add_argument("--device")
    args = ap.parse_args()
    cfg = load_config()
    meta = json.loads((RUNS / args.model / args.variant / "meta.json").read_text())
    run(cfg, meta, args.file, args.exps.split(","), args.device)


if __name__ == "__main__":
    main()
