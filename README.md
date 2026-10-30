# bits-of-pi

*Quantization and digit recall: how many digits of π survive each bit removed.*

**Results: [engineeringthefutureai.github.io/bits-of-pi](https://engineeringthefutureai.github.io/bits-of-pi/)**

How does weight quantization damage exact, low-margin token decisions? This repo measures it
with digit sequences on Qwen 3.5/3.8 GGUF models (28 variants of the 4B, 19 of the 9B, 17 of the 27B), from BF16 down to
~2 bits per weight, run through llama.cpp.

Three abilities are tested separately:

- **Recall:** reciting digits of π, e and √2 stored in the weights.
- **Copying:** reproducing random digits given in the prompt, verbatim or spelled as words.
- **Arbitration:** copying π with one planted wrong digit: prompt or memory?

A teacher-forced pass measures per-position logit margins and KL divergence from BF16, to
explain *where* recall breaks, not just *when*.

The full test plan, metrics, comparison method and hypotheses are in **[PLAN.md](PLAN.md)**.

## Layout

```text
config.yaml          models, variants, server flags, pinned llama.cpp commit
PLAN.md              test plan and comparison methodology
data/                reference digits (mpmath), random strings, calibration hash
inputs/<exp>.jsonl   every prompt, generated once from a fixed seed
runs/<model>/<variant>/              per-variant summaries: file, bench, resources, E0
                                     (raw request records stay local, see below)
tf/<model>/<variant>/<text>.parquet   E2 per-digit-position records
scores/              scored rows and headline tables
figures/             one figure per hypothesis, plus H5_model_size.png across models
docs/                GitHub Pages site (index.html + copies of its figures)
scripts/             make_inputs, pipeline, run_gen, run_tf, e0, score, analyze
tools/tf_dump.cpp    teacher-forced logit dump, linked to the pinned llama.cpp build
tests/               unit tests for extraction and alignment
```

## Reproducing

Requirements: a llama.cpp checkout at commit `bed0a8566` built with the backend named in
`config.yaml` (SYCL here; Vulkan or CUDA work by editing `llama_cpp.build`/`backend`), symlinked
as `./llama.cpp`, and Python 3.11+.

```bash
ln -s /path/to/llama.cpp llama.cpp
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
tools/build.sh                         # builds tools/bin/tf_dump against the pinned build
.venv/bin/python scripts/make_inputs.py
.venv/bin/python -m pytest tests

.venv/bin/python scripts/pipeline.py --model qwen3.5-4b       # download/quantize -> run -> delete
.venv/bin/python scripts/score.py --model qwen3.5-4b
.venv/bin/python scripts/analyze.py --model qwen3.5-4b
.venv/bin/python scripts/build_site.py                 # refresh docs/figures for the site
```

Raw logs (per-request records in `runs/**/*.jsonl`, per-second resource samples, server
output) are written locally and gitignored. Published data starts at the scored rows in
`scores/` and the per-position E2 records in `tf/`.

The pipeline is resumable: stop it at any point and rerun the same command. With two GPUs,
run a second pipeline on the other device and port with a disjoint `--variants` list
(after the reference variant has finished):

```bash
.venv/bin/python scripts/pipeline.py --model qwen3.5-4b --device level_zero:1 --port 18081 --variants ...
```

## Status

Phase 1 (Qwen 3.5 4B): **[REPORT.md](REPORT.md)**. Phase 2 (Qwen 3.5 9B, and the 4B/9B
comparison): **[REPORT_9B.md](REPORT_9B.md)**. Phase 3 (Qwen 3.8 27B, and the three-size
comparison): **[REPORT_27B.md](REPORT_27B.md)**.

## License

Code: MIT. Calibration text is derived from WikiText-2 (CC BY-SA 3.0) and is regenerated
locally rather than committed.
