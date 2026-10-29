# Phase 1 report: Qwen 3.5 4B

Kiryl Dubarenka

28 variants of one model, from BF16 (16 bits per weight) down to 2.9 bits per weight, run
through the same llama.cpp build (`bed0a8566`, SYCL, one Arc Pro B60) with greedy decoding.
Every number here comes from the scored rows in `scores/`, the per-position E2 records in
`tf/` and the per-variant summaries in `runs/`, via `scripts/analyze.py`. Raw request logs are
kept locally and not published. The full per-variant table is
[`scores/qwen3.5-4b_headline.md`](scores/qwen3.5-4b_headline.md).

Bits per weight (bpw) is measured from each file (`size × 8 / parameters`). For this small
model it runs about 0.3–0.4 higher than the usual labels, because the large vocabulary
(token embedding and output layers) stays at higher precision: Q4_K_M is 5.15 bpw here, not 4.8.

## Summary

| Hypothesis | Verdict | Deciding evidence |
|---|---|---|
| H1 recall cliff near 3 bits | **Rejected as stated** | Recall of π declines from 6.6 bpw down, with no flat stretch to 4 bits ([H1](figures/qwen3.5-4b_H1_recall.png)) |
| H2 margins explain flips | **Confirmed** | Recall stops cluster on BF16's lowest-margin digits; noise grows 7.6× per bit on π ([H2](figures/qwen3.5-4b_H2_margins.png), [KL](figures/qwen3.5-4b_H2_kl.png)) |
| H3 copying is robust; failures are skips in runs | **Confirmed** | Verbatim copy is near perfect down to 4.3 bpw for every build; errors are deletions inside runs, even at BF16 ([E3](figures/qwen3.5-4b_H3_E3.png), [E4b](figures/qwen3.5-4b_H3_E4b.png)) |
| H4 scheme matters at equal size | **Confirmed, with a twist** | K beats Q4_0 by 3.3× on π noise; a digit-free imatrix cuts generic noise but not π noise; unsloth's recipe beats every DIY build ([H4](figures/qwen3.5-4b_H4_builds.png)) |
| H5 size buys headroom | Not tested yet | Needs Phases 2–3 |
| H6 low bits revert to memory | **Rejected, reversed** | Lower-bit models copy the planted digit *more* faithfully ([H6](figures/qwen3.5-4b_H6_planted.png)) |

Practical reading for this model on this card:

- **Q8_0 is free.** Indistinguishable from BF16 on every task (KL 0.0001), 1.7× faster decode,
  43% less VRAM.
- **For exact recall, stay at Q6_K or above.** Each bit removed multiplies the noise on
  memorized digits by ~7.6×; π recall halves somewhere between Q6_K and Q5_K_M.
- **For copying from the prompt, almost anything works** down to Q3_K_M, and down to Q2_K with
  an imatrix. Plain Q2_K and our IQ3_XXS/IQ2_XS builds break.
- **Prefer unsloth's file over a stock quant of the same type**: at Q4_K_M and Q3_K_M it has
  about half the noise on π for the same size and speed.
- **Avoid IQ types on SYCL**: IQ3_XXS decodes at 35–38 tokens/s, slower than BF16.

## H1: recall declines early and steadily

E1 asks for π, e and √2 from memory in three phrasings each; the score is the position of the
first wrong digit. BF16 recites **1,000 correct digits of π** (then stops, because the prompt
asked for 1,000), 192 of e and 105 of √2. π is the constant with runway, so it carries H1.

| DIY ladder (no imatrix) | bpw | π digits (median of 3) | KL vs BF16 on π | KL on random digits |
|---|---|---|---|---|
| BF16 (reference) | 16.0 | 1000 | 0 | 0 |
| Q8_0 | 8.5 | 1000 | 0.00013 | 0.00008 |
| Q6_K | 6.6 | 813 | 0.0011 | 0.0012 |
| Q5_K_M | 5.9 | 428 | 0.0075 | 0.0012 |
| Q4_K_M | 5.2 | 123 | 0.039 | 0.0033 |
| Q3_K_M | 4.3 | 88 | 0.35 | 0.0059 |
| Q2_K | 3.6 | 0 | 2.30 | 0.045 |

There is no plateau down to 4 bits. Recall falls below half the reference already at Q5_K_M
(5.9 bpw). The decline in KL is smooth; the steps in E1 come from a few specific digits, not a
threshold in bits (H2).

Recall is far more fragile than generic digit prediction: at Q4_K_M, noise on the memorized π
text is **12× higher** than on random digits, and it grows 7.6× per bit removed against 3.2× on
random digits.

E1 is noisy between neighbouring quants. Our Q6_K recites 813 digits and unsloth's Q6_K 326,
with almost the same KL (0.0011 vs 0.0013): both sit at the threshold of digit 326, and pass or
fail on it. KL is the steadier measure; E1 shows the consequence.

## H2: errors start where margins are thinnest

E2 feeds the true 1,000 digits of π and reads the logits at every position. The margin is the
log-probability gap between the right digit and the best competitor.

- BF16 gets all 1,000 teacher-forced positions right. Its median margin is 9.2 nats, but a few
  digits are knife-edge: position 813 (0.10 nats), 326 (1.34), 10 (1.40).
- Free-recall stops pile up on those digits: of 77 π recalls that stopped early, the most common
  stop positions are 326 (8 runs), 88 (7), 821, 10, 87 and 123 (4 each). The median stop lies at
  the 12th percentile of BF16's margins.
- At high bits the errors are only at the thinnest margins: Q6_K's two teacher-forced errors are
  at positions 813 and 821 (0.1th percentile of margin). At Q4_K_M the 14 errors sit at the
  7.6th percentile. At Q3_K_M (104 errors) and below, noise exceeds most margins and errors
  spread across all positions.

The H2 figure shows this directly: each row's dots start at the thin-margin columns and fill
in as bits drop.

## H3: copying holds, and its failures are skips inside runs

**Verbatim copy (E3)**, 40 random strings of 100–1,000 digits: BF16 makes 3 errors in
17,000 digits, all single insertions. Every build at 4.3 bpw or above copies 1,000-digit strings
perfectly. Copying breaks for three builds only:

| Variant | bpw | Errors per 100 digits | On 1,000-digit strings | Kind |
|---|---|---|---|---|
| Q2_K (no imatrix) | 3.6 | 59 | 99 | mostly deletions; the model gives up |
| IQ3_XXS (our imatrix) | 3.6 | 9.0 | 36 | deletions of whole chunks |
| IQ2_XS (our imatrix) | 3.1 | 12 | 39 | mixed |

The same types survive in other builds: Q2_K with an imatrix makes 0.06 errors per 100, and
unsloth's UD-IQ3_XXS 0.03.

**Spelling as words (E4b)**: even BF16 drops digits inside runs. Run-heavy strings get 2.2
errors per 100 digits against 0.3 on plain random strings, and 75 of BF16's 80 errors are
deletions inside runs of 3+ equal digits ("eight eight eight eight eight" written one short).
That pattern holds unchanged down to about 4.3 bpw. Below that, failures turn into
**insertions**: the model loops, repeating a word too many times (Q2_K_imat: 208 errors per
100 digits on run-heavy strings, 5,297 insertions). Substitutions stay rare everywhere except
the builds that collapse.

The original E4 prompt failed as a test: Qwen 3.5 4B read "write each digit as a word" as
"say this number" ("twenty trillion two hundred…"), even at BF16. Its records are kept; E4b
with an explicit example ("507 becomes: five zero seven") replaces it.

## H4: scheme, imatrix and recipe

Each row of the [H4 figure](figures/qwen3.5-4b_H4_builds.png) is one quant type, built several
ways. KL on π measures noise on memorized content; KL on random digits measures generic noise.

| Build of Q4_K_M | π digits | KL on π | KL on random |
|---|---|---|---|
| plain | 123 | 0.039 | 0.0033 |
| + our digit-free imatrix | 123 | 0.049 | 0.0016 |
| + unsloth's imatrix | 220 | 0.041 | 0.0024 |
| unsloth's file | 326 | 0.022 | 0.0019 |
| unsloth UD-Q4_K_XL (+6% size) | 326 | 0.014 | 0.0013 |

| Build of Q3_K_M | π digits | KL on π | KL on random |
|---|---|---|---|
| plain | 88 | 0.35 | 0.0059 |
| + our imatrix | 88 | 0.33 | 0.0045 |
| + unsloth's imatrix | 88 | 0.29 | 0.0028 |
| unsloth's file | 87 | 0.17 | 0.0027 |
| unsloth UD-Q3_K_XL (+8% size) | 168 | 0.10 | 0.0025 |

- **Scheme:** at the same size class, legacy Q4_0 has 3.3× the π noise of Q4_K_M
  (0.131 vs 0.039). Confirmed.
- **Imatrix:** our imatrix, built from text with no digit strings, halves generic noise at
  Q4_K_M but does not protect π (KL on π even rises, 0.039 → 0.049). An importance matrix
  protects what its calibration text exercises, which is exactly why the spec kept digits out of
  it. At 2–3 bits it matters for everything: it is the difference between Q2_K that cannot copy
  (59 errors per 100) and one that can (0.06).
- **Recipe beats calibration:** stock Q4_K_M with unsloth's own imatrix gets only a little of
  unsloth's advantage (π KL 0.041 vs their 0.022). Most of the gain comes from their per-tensor
  type choices, not their calibration data. UD-XL files go further, at a cost in size and decode
  speed (below).

## H6: low bits weaken memory, they don't strengthen it

E5 asks for a copy of π or e with one digit changed. Of the 8 planted positions, only π position
10 separates the variants: BF16 and nearly every build at 4.6 bpw or more "correct" it to the
memorized digit, while nearly every build at 4.4 bpw or below copies the prompt faithfully.
Quantization weakens the pull of memory rather than strengthening it, consistent with H1.
Exceptions: our Q4_K_M_imat and unsloth Q4_0 copy faithfully at ~5 bpw; our IQ3_XXS reverts at
position 10 and unsloth UD-IQ3_XXS at positions 10 and 50; plain Q2_K fails to copy at all.

## Resources and speed

All on one Arc Pro B60 (24 GB), llama.cpp SYCL, 8,192-token f16 KV cache, flash attention off.
Decode speed is the median the server reported on the actual test requests; it agrees with
`llama-bench` tg128 within 0–5%. Prompt processing is `llama-bench` at 4,096 tokens (short-prompt
server timings understate SYCL throughput, so they are not used).

| Variant | File GB | VRAM GiB | Decode tok/s | Prompt tok/s (4k) | Power W |
|---|---|---|---|---|---|
| BF16 | 8.42 | 8.6 | 38 | 1,087 | 120 |
| Q8_0 | 4.48 | 4.9 | 65 | 2,964 | 127 |
| Q6_K | 3.46 | 4.0 | 76 | 2,984 | 147 |
| Q5_K_M | 3.07 | 3.6 | 81 | 3,092 | 142 |
| Q4_K_M | 2.71 | 3.3 | 89 | 2,807 | 138 |
| us_UD-Q4_K_XL | 2.91 | 3.5 | 78 | 2,838 | 132 |
| IQ4_XS | 2.51 | 3.1 | 65 | 2,223 | 130 |
| Q3_K_M | 2.26 | 2.9 | 85 | 2,332 | 139 |
| us_UD-IQ3_XXS | 1.95 | 2.6 | 35 | 1,674 | 113 |
| Q2_K | 1.92 | 2.5 | 91 | 2,617 | 136 |
| us_UD-IQ2_XXS | 1.52 | 2.2 | 53 | 1,693 | 129 |

- **VRAM = file size + ~770 MiB** for every variant: the KV cache, recurrent state and compute
  buffers don't depend on the weights. Any 4 GB card holds every quant up to Q5_K_M with an 8k
  context; BF16 needs ~9 GB.
- **Decode stops speeding up below ~5 bpw.** 38 → 65 → 89 tokens/s from BF16 to Q8_0 to
  Q4_K_M, then flat to Q2_K. A 4B model on this card is no longer limited by weight reads below
  4–5 bits, so smaller quants buy memory, not speed.
- **IQ types are slow on SYCL**: their codebook lookups cost more than the bytes they save.
  IQ3_XXS (35–38 tok/s) decodes slower than BF16. UD-XL mixes cost 12–28% decode against plain
  K-quants of similar size.
- **Host RAM** tracks the file size (the model is memory-mapped), so it is not an extra
  requirement on top of VRAM.

## Problems found and how they were handled

| Issue | Effect | Handling |
|---|---|---|
| E4 prompt read as "say this number" | E4 measured instruction following, not copying | Kept as recorded; added E4b with an explicit example (approved), rerun on all 28 variants |
| E6 (recall as words) at the floor | BF16 counts "zero one two three…" instead of reciting | Reported as a 4B capability floor |
| E1 "e to 1000 decimal places: 2." | BF16 answers with 1,200 zeros | Real model behaviour; reported per constant, π carries H1 |
| E5 copies drop the decimal point | First runs scored a spurious error | Extraction fixed before any real run, with a unit test |
| Server host-RAM prompt cache | Inflated host memory; possible state reuse | Disabled (`--cache-ram 0`) and the whole study rerun |

Determinism (E0) passed on BF16 and Q8_0: identical bytes on repeated requests, after a
server restart, and across the two GPUs.

## Next

Phase 2 (Qwen 3.5 9B, BF16 safetensors local) with the same pipeline, then the 27B. H5 needs
both. Open for review before that:

- Keep E6 for larger models, or drop it?
- Add `--no-mmap` so host RAM measures what the model really needs?
- The calibration question has a sharper form now: a digit-free imatrix leaves recall
  unprotected. Is a "generic text plus some numbers" calibration the fairer choice for H4?
