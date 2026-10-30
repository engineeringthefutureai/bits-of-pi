# Test plan and comparison methodology

This plan turns the original design doc into a concrete run. Where it departs from that
doc, the change and its reason are listed under [Decisions and deviations](#decisions-and-deviations).

## What we measure

The probe is a digit sequence: one right answer per position, checked mechanically. Digits
are single tokens in the Qwen 3.5 tokenizer (checked in E0: 202 tokens for "3." + 200 digits),
so **digit position = token position** throughout.

The experiments form a 2 × 2 grid plus two probes that look inside the model:

|                       | digits out        | words out              |
|-----------------------|-------------------|------------------------|
| **from memory**       | E1 free recall    | E6 recall as words     |
| **from the prompt**   | E3 verbatim copy  | E4 spell as words      |

- **E2 teacher-forced margins:** feed the true digits and read logits at every position. No
  generation, so one early error can't derail the rest; this is the mechanism behind E1.
- **E5 planted error:** copy π or e with one digit changed; does the model copy the prompt or
  "correct" it from memory? This is arbitration between the two rows of the grid.
- **E0 sanity checks:** tokenization, thinking off, byte-identical determinism (twice on one
  server, once after a restart), Q8_0 ≈ BF16.

## Is the metric "first wrong token"?

Yes for recall, no for copying.

| Exp | Primary metric | Why |
|---|---|---|
| E1, E6 | **correct prefix length**: position of the first wrong digit | After the first error a recalling model is off its memorized path; anything after it is noise, not partial credit. |
| E1 | **agreement prefix** with the reference variant's own output | Separates "quantization changed the answer" from "the reference was wrong too". Sharper than the correct prefix when the reference itself fails early (e.g. √2). |
| E3, E4, E5 | **edit distance per 100 digits** from a Levenshtein alignment, split into substitutions, insertions and deletions, each flagged if inside a run of ≥ 3 equal digits | A copy that skips one digit at position 40 is 1 error, not 260. The first-error position is still recorded (`prefix`). |
| E5 | outcome at the planted position: `planted` / `memory` / `other` | The question is which source won, not how many digits matched. |
| E2 | per position: log p(correct), rank, **margin** (correct − best competitor), top-1 flip vs reference, **KL(reference ‖ variant)** | Margin is how much noise a position tolerates; KL is how much noise a quant adds. H2 predicts errors where the second exceeds the first. |

For E1, the strict prefix stops at the first non-digit (spec). A lenient prefix that ignores
whitespace is also recorded, so a model that wraps lines every 50 digits isn't scored as
stopping at 50; the two are reported side by side if they ever differ.

## Variants (Phase 1, Qwen 3.5 4B)

Two arms, all from the same BF16 GGUF (unsloth's, `Qwen3.5-4B-BF16.gguf`):

| Arm | Variants | Purpose |
|---|---|---|
| Reference | BF16 | Baseline for every comparison and the E2 KL target |
| DIY ladder (no imatrix) | Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K | H1: where is the cliff, with nothing protecting the weights |
| DIY scheme | Q4_0 | H4: legacy vs K-quant at the same size |
| DIY + own imatrix | Q4_K_M_imat, Q3_K_M_imat, Q2_K_imat, IQ4_XS, IQ3_XXS, IQ2_XS | H4: smarter rounding, calibrated on digit-free text |
| DIY + unsloth imatrix | Q4_K_M_usimat, Q3_K_M_usimat | Separates unsloth's calibration data from their per-tensor recipe |
| unsloth prequantized | Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_0, IQ4_XS, Q3_K_M, UD-Q4_K_XL, UD-Q3_K_XL, UD-Q2_K_XL, UD-IQ3_XXS, UD-IQ2_XXS | What people actually run |

28 variants. Bits per weight is measured per file as `file size × 8 / parameters` (parameters
summed over every tensor of the BF16 file) and is the x-axis everywhere.

Each non-reference file is downloaded or quantized, run, reduced, then **deleted**; the BF16
GGUF, both imatrices and the reference's raw E2 log-probabilities stay until the model is done.

## Inputs

All generated once by `scripts/make_inputs.py` from seed 20261027; committed in `inputs/`.

| Exp | Endpoint | Inputs | Max tokens |
|---|---|---|---|
| E1 | `/completion` | π, e, √2 × 3 phrasings = 9 | 1,200 |
| E2 | `tf_dump` (logits) | 1,000 decimals of π, e, √2 + 1,000 random digits = 4 | — |
| E3 | chat | random strings: 10 each at 100 / 300 / 1,000 digits + 10 run-heavy at 300 = 40 | 1.3 × n + 50 |
| E4 | chat | the 100- and 300-digit strings + run-heavy = 30 | 3 × n + 50 |
| E5 | chat | first 300 decimals of π and e, one digit replaced by (d + 5) mod 10 at 10 / 50 / 150 / 290 = 8 | 441 |
| E6 | chat | first 500 decimals of π, e, √2 as words = 3 | 1,550 |

Reference digits come from `mpmath` at two precisions that must agree; the strict E1 scores of
the Q4_K_M smoke test (326 correct digits of π) matched them digit for digit.

## Run protocol

As in the spec: one pinned llama.cpp build (`bed0a8566`, SYCL, one Arc Pro B60), one warm
`llama-server` per file, greedy decoding (`temperature 0, top_k 1`, every penalty off,
`cache_prompt: false`), `--parallel 1`, f16 KV cache, flash attention **off** (pinned, not
`auto`, so it can't switch per quant type). Thinking is off twice over: `--reasoning off` on the
server and `chat_template_kwargs.enable_thinking: false` per request; E0 verifies it.

Every request appends one JSON line to `runs/<model>/<variant>/<exp>.jsonl` (a raw log, kept
locally and never published) with the prompt,
the full request body, the raw output, token counts, finish reason, commit, backend and device.
Runs are resumable and records are append-only; the one later change was stripping wall-clock
timestamps before publication.

## Resources and speed (side measurements)

Not a benchmark, but enough to answer "what hardware runs this quant, and what does it cost":

| Measure | Source | Stored in |
|---|---|---|
| VRAM needed | `xpu-smi` device memory, peak during the run minus the idle baseline before the server starts; includes weights, the 8,192-token f16 KV cache and compute buffers | `runs/<m>/<v>/resources.json` |
| Host RAM | llama-server peak resident memory (`VmHWM`) | `resources.json` |
| Power, memory-bandwidth utilization | `xpu-smi`, 1 Hz | `resources.csv`, means in `resources.json` |
| Decode and prompt speed | llama-server `timings` on every response; median over requests with ≥ 100 tokens | each run record |

Caveats: both GPUs run different variants at once and quantization runs on the CPU in
between, so speeds are good to roughly ±10%, fine for comparing quants, not for publishing as
benchmarks. VRAM is per device and unaffected. The server's host-RAM prompt cache is disabled
(`--cache-ram 0`) so host memory reflects the model, not cached prompts.

## Comparison methodology

All variants see identical inputs, so every comparison is **paired by input**.

1. **Curves against bits per weight (H1, H2, H5).** One line per arm. The cliff is the
   highest-bpw ladder variant whose E1 median prefix falls below half the reference's. The
   noise level is mean KL on the π text; the prediction is that the cliff sits where KL
   reaches the margins typical of the positions just before the reference's recall ends.
2. **Matched pairs at equal type (H4).** For each quant type, the four builds of it:
   DIY plain → DIY + own imatrix → DIY + unsloth imatrix → unsloth file. Each step changes one
   thing: imatrix on, calibration data, recipe. Plus Q4_0 vs Q4_K_M for the scheme. Reported
   as the difference in E1 median prefix and E2 mean KL, with the per-input sign (how many of
   the 9 E1 prompts got better, worse or stayed the same).
3. **Reference agreement.** DIY Q8_0 and unsloth Q8_0 should be essentially identical
   (no imatrix applies to Q8_0); a gap means a pipeline or converter difference, not quantization.
4. **Uncertainty.** Greedy runs are deterministic, so there's no run-to-run noise; spread comes
   from inputs. E1 reports median and min–max over 9 prompts; E3/E4 report mean errors per
   100 digits with a bootstrap 95% interval over inputs; E5 reports counts out of 8 (too few
   for rates, so read as a trend across bpw).
5. **Copy failure shape (H3).** Errors split into substitution, insertion and deletion, and
   inside vs outside runs; H3 predicts insertions and deletions inside runs, in long strings,
   at low bits.

Each hypothesis ends as **confirmed / rejected / unclear** with the figure that decides it:

| Hypothesis | Deciding figure | Confirmed if |
|---|---|---|
| H1 recall cliff | `figures/<m>_H1_recall.png` | ladder prefix flat to ~Q4, then falls below half the reference at Q3 or Q2 |
| H2 margins explain flips | `figures/<m>_H2_margins.png`, `_H2_kl.png` | top-1 flips concentrate at low-margin positions; KL grows ~4× per bit removed |
| H3 copying robust | `figures/<m>_H3_E3.png`, `_H3_E4.png` | E3 near zero down to Q3; errors mostly ins/del in runs |
| H4 scheme matters | headline table pairs | K > Q4_0 and imatrix > plain, both widest at ≤ 3 bits |
| H5 size buys headroom | `figures/H5_model_size.png` | larger model's cliff at fewer bits |
| H6 arbitration shifts | `figures/<m>_H6_planted.png` | `memory` share rises as bpw falls |

## Phases

- **Phase 0, setup:** done. Scripts, inputs, `tf_dump`, tests, smoke test on unsloth Q4_K_M.
- **Phase 1, 4B:** BF16 + Q8_0 first (E0 gate), then the remaining 27 variants;
  headline table + H1/H2 figures; **stop for review**.
- **Phase 2, 9B:** 19 variants: unsloth's BF16 (reference) and 12 published files, plus a
  plain no-imatrix ladder (Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K) and Q2_K with our imatrix. The
  plain ladder gives H5 a like-for-like comparison across sizes; the imatrix/recipe breakdown
  (H4) was answered on the 4B and is not repeated. The original E4 is skipped (E4b replaces it).
- **Phase 3, 27B (Qwen 3.8 27B):** 17 variants. BF16 (54.7 GB) does not fit in 2 × 24 GB, so
  unsloth's Q8_0 (made from the BF16 release, not the FP8 one) is the reference and runs split
  over both GPUs, as do Q6_K and UD-Q6_K. The plain ladder (Q6_K to Q2_K) is quantized from
  unsloth's BF16, as on the smaller models; Q2_K_imat uses an imatrix computed from the Q8_0
  reference. unsloth's 27B K-quant files are all dynamic (UD) mixes: UD-Q6_K, UD-Q5_K_M,
  UD-Q4_K_XL, UD-Q4_K_M, Q4_0, UD-IQ4_XS, UD-Q3_K_XL, UD-IQ3_XXS, UD-Q2_K_XL, UD-IQ2_XXS.
  E4 is skipped. Speed for split variants is two-GPU speed and is not comparable to single-GPU
  numbers. The 27B loads without mmap (`mmap: false` in `config.yaml`): a 22–29 GB mapped file
  on a 30 GB host triggered the OOM killer. Done: [REPORT_27B.md](REPORT_27B.md).
- **Phase 4:** cross-model figures and the hypothesis report.

## Decisions and deviations

| Topic | Spec | This run | Why |
|---|---|---|---|
| Backend | unspecified | **SYCL**, one B60 | Same results as Vulkan and CPU on a greedy π test (326 correct digits each, deterministic), ~1.7× faster than Vulkan |
| Quant files | all DIY from BF16 safetensors | 4B: unsloth files **and** DIY from unsloth's BF16 GGUF | Grounds results in files people run; DIY arm keeps the controlled contrasts |
| Q2_K, IQ2_XS in unsloth arm | — | UD-Q2_K_XL, UD-IQ2_XXS | Not published by unsloth |
| E2 tool | `llama-cpp-python` | `tools/tf_dump.cpp` linked to the same build | Same commit and backend as the server by construction; no Python-binding version drift |
| E6 | — | added | Completes the memory × context × digits/words grid. Possibly at floor for the 4B (unsloth Q4_K_M counts 0–9 instead of reciting); kept if BF16 does better, else reported as floor |
| Calibration | open question | wikitext-2 train, lines with any 3+ digit run or decimal number dropped, first 600k chars | Widely used for imatrix; digit-free by construction; SHA-256 in `data/calibration.sha256` |
| 28B | Qwen 3.8 28B | Qwen3.8-27B; reference is unsloth's Q8_0 | That is the model available. BF16 does not fit in 2 × 24 GB; the FP8 release was not used, so every 27B file comes from the same BF16 GGUF lineage |
| Flash attention | "identical across variants" | pinned `off` | `auto` could differ by quant type |
