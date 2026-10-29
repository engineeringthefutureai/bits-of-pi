# Phase 2 report: Qwen 3.5 9B, and what model size buys

Kiryl Dubarenka

Phase 1 (4B): [REPORT.md](REPORT.md). Web version:
[engineeringthefutureai.github.io/bits-of-pi](https://engineeringthefutureai.github.io/bits-of-pi/)

19 variants of the 9B, from BF16 down to 2.9 bits per weight. The setup is the same as Phase 1:
same llama.cpp build (`bed0a8566`, SYCL, one Arc Pro B60 per variant), same inputs, greedy
decoding. The variant set covers the comparisons that matter for H5:

- unsloth's BF16 (the reference) and 12 of unsloth's published files.
- A plain no-imatrix ladder (Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K), built exactly as on the 4B.
- Q2_K with our imatrix.

The H4 build breakdown (own imatrix, unsloth's imatrix) is not repeated. The original E4 prompt
is skipped (E4b replaces it). Determinism (E0) passed on BF16 and unsloth Q8_0. The full table
is [`scores/qwen3.5-9b_headline.md`](scores/qwen3.5-9b_headline.md).

## Summary

| Hypothesis | 9B verdict | Deciding evidence |
|---|---|---|
| H1 recall cliff near 3 bits | **Closer, still not a plateau** | π recall stays above 1,000 digits to 5.8 bpw and halves between 5.0 and 4.1 bpw ([H1](figures/qwen3.5-9b_H1_recall.png)) |
| H2 margins explain flips | **Confirmed, weaker form** | No knife-edge digits; recall stops still sit at the 6th percentile of BF16's margins ([H2](figures/qwen3.5-9b_H2_margins.png)) |
| H3 copying is robust | **Confirmed, stronger** | Every build except plain Q2_K and UD-IQ2_XXS copies near-perfectly, and the E4b looping the 4B showed at 3.6–4.3 bpw is gone ([E3](figures/qwen3.5-9b_H3_E3.png), [E4b](figures/qwen3.5-9b_H3_E4b.png)) |
| H4 recipe matters | **Confirmed again** | unsloth's files keep less π noise than our plain ladder at Q4/Q3, and an imatrix again rescues Q2_K ([H4](figures/qwen3.5-9b_H4_builds.png)) |
| **H5 size buys headroom** | **Confirmed per bit, not per byte** | Recall halves at Q3_K_M (4.1 bpw) instead of Q5_K_M (5.9 bpw); at equal file size the 4B at 8 bits recalls more ([H5](figures/H5_model_size.png)) |
| H6 low bits revert to memory | **Nothing to revert** | 9B BF16 already copies every planted digit; so does every build except the two that cannot copy ([H6](figures/qwen3.5-9b_H6_planted.png)) |

Practical reading for the 9B on this card:

- **Q8_0 is free**: KL on π 0.000007, about 1.9× faster than BF16, half the VRAM.
- **Q6_K and Q5_K_M keep 1,000+ digits of π.** Q4_K_M keeps 700–790 digits. Below 4.5 bpw,
  recall drops to about 130.
- **Copying survives almost everything** down to 3.6 bpw, imatrix or not, except plain Q2_K.
- **If memory is the constraint and recall is the task, a smaller model at 8 bits beats a
  larger one at 3–4 bits.** At about 4.5 GB, 4B Q8_0 recites 1,000 digits, and 9B Q3_K_M
  recites 129.
- **IQ types remain slow on SYCL**: UD-IQ3_XXS decodes at 20 tokens/s, slower than BF16 (22).

## H5: what the bigger model buys

The plan defines the cliff as the highest-bpw ladder variant whose median π recall falls below
half the reference's.

| Plain ladder | 4B bpw | 4B π digits | 4B KL on π | 9B bpw | 9B π digits | 9B KL on π |
|---|---|---|---|---|---|---|
| BF16 (reference) | 16.0 | 1000 | 0 | 16.0 | 1180 | 0 |
| Q8_0 | 8.5 | 1000 | 0.00013 | 8.5 † | 1200 | 0.000007 |
| Q6_K | 6.6 | 813 | 0.0011 | 6.6 | 1056 | 0.00010 |
| Q5_K_M | 5.9 | **428** | 0.0075 | 5.8 | 1090 | 0.00039 |
| Q4_K_M | 5.2 | 123 | 0.039 | 5.0 | 788 | 0.0058 |
| Q3_K_M | 4.3 | 88 | 0.35 | 4.1 | **129** | 0.13 |
| Q2_K | 3.6 | 0 | 2.30 | 3.4 | 0 | 2.39 |

† unsloth's Q8_0. On the 4B, our Q8_0 and unsloth's were indistinguishable (KL 0.000133 vs 0.000131).

Bold marks the cliff: Q5_K_M on the 4B, Q3_K_M on the 9B, about 1.7 bits further down. **H5
is confirmed.** The shape of the gain matters, though:

- **The 9B starts with far less noise, but loses its lead faster.** At 8–6 bits its KL on π is
  10–19× below the 4B's. As bits drop that noise grows 12× per bit, against 7.6× on the 4B
  (fit over Q8_0–Q2_K on both). The two curves meet at Q2_K, at 2.3–2.4 nats. Across all 12
  unsloth files the 4B/9B ratio of π noise falls from 15–19× at Q8_0–Q5_K_M, to 3.6–14× at
  4.2–5.3 bpw, to 2.4× at UD-IQ3_XXS and 1.2–1.5× at UD-Q2_K_XL and UD-IQ2_XXS. So size buys headroom at moderate quantization, and at
  2–3 bits that headroom is almost gone.
- **For the 9B, memorized digits are no longer the fragile case at high bits.** Noise on π sits
  below noise on random digits down to Q4_K_M (0.0058 vs 0.0096) and only overtakes it at
  Q3_K_M. On the 4B, π noise was already 12× higher at Q4_K_M. The 9B holds π more confidently:
  median margin 10.3 nats, and only one position under 2 nats (the 4B had nine).
- **Per byte, the comparison reverses.** H5 compares at equal bits per weight. A user picking a
  file to fit in memory compares at equal bytes, where the bigger model has fewer bits per
  weight. At about 4.5 GB, 4B Q8_0 recites 1,000 digits and 9B Q3_K_M or UD-Q3_K_XL recite
  129. The 9B matches the 4B's 1,000 digits only from 6.5 GB (Q5_K_M). The 4B at BF16, 8.4 GB,
  is never needed for this task. The top-right panel of the [H5 figure](figures/H5_model_size.png) shows this. Exact
  recall needs many bits per weight more than it needs many weights. Copying points the same
  way: the 4B copies near-perfectly from 2.3 GB (Q3_K_M), a size no working 9B build reaches
  (H3).

## H1: recall on the 9B

E1 asks for π, e and √2 in three phrasings each. BF16 recites π **past the 1,200-digit
generation cap** in one phrasing and to 1,180 in another. In the third ("to 1000 decimal
places") it stops correctly at 1,000. So the 9B reference is a lower bound, and Q8_0 matches it.

| Constant | 9B BF16, by phrasing | 4B BF16, by phrasing |
|---|---|---|
| π | 1000 (stops as asked) / 1200+ / 1180 | 1000 / 1000 / 821 |
| e | list format / **192** / 15 (Python code) | 1,200 zeros / **192** / 211 |
| √2 | 219 / 202 / 203 | 176 / 105 / 102 |

Both models know exactly 192 digits of e from the plain phrasing. The 9B's headline "e = 15" is
a prompt artifact: `e to 1000 decimal places: 2.` is continued as a numbered list
(`2. 1000 decimal places: 3. 10000…`), and `e = 2.` as Python code (`e = 2.718281828459045`).
Only π has the runway to carry H1.

π recall moves in steps: 1,056, 722 and 129 recur across unrelated builds. Each step is one
specific digit where noise first overturns the right answer. At equal type, the plain ladder
and unsloth's files sit on the same steps (Q4_K_M: 788 vs 722).

## H2: no knife-edges, same mechanism

BF16 gets all 1,000 teacher-forced positions of π right. The 4B had digits decided by 0.1 nats.
The 9B's thinnest margin is 1.6 nats (position 10), and its median is 10.3. The margin also
rises with depth into π, from a mean of 6.6 nats over the first 100 digits to 11.2 after digit
300: context reinforces recall.

With no knife-edges, quants survive noise that broke the 4B. Every variant except plain Q2_K
gets position 10 right. Where recall does fail, it still fails at thin margins, just not the thinnest:

- The first teacher-forced flip of unsloth's Q4_K_M, UD-Q4_K_XL and Q4_0 is the same digit,
  position 722 (BF16 margin 6.3 nats, 6th percentile). Our plain Q4_K_M flips at 788 and 910.
- Unsloth Q3_K_M first flips at 129 (6.2 nats, 5th percentile), the most common E1 stop.
- Over all 35 early π stops in free recall, the median stop lies at the **6th percentile** of
  BF16's margins, and 57% lie in the lowest 10%. On the 4B, by the same count, the figures are
  the 14th percentile and 35%.

Margin is still the predictor. The 9B needs about 6 nats of noise to flip a digit where the 4B
needed 0.1–1.4.

## H3: copying

**Verbatim copy (E3)**, 17,000 digits per variant:

| Variant | bpw | Errors per 100 | Kind |
|---|---|---|---|
| BF16, Q8_0 | 16, 8.5 | 0.025 | 3 deletions, all inside runs |
| Q4_K_M → UD-IQ3_XXS → Q2_K_imat | 5.0 → 3.4 | 0.03–0.07 | deletions inside runs |
| UD-IQ2_XXS | 2.9 | 1.6 | deletions of chunks |
| Q2_K, no imatrix | 3.4 | 34 | the model stops early |

Every build from 3.4 to 3.7 bpw except plain Q2_K copies at 0.04–0.07 errors per 100. Plain
Q2_K breaks on both sizes, and an imatrix rescues it on both.

**Spelling as words (E4b)**: the 4B pattern repeats. BF16 drops digits inside runs of equal
digits (78 errors in 7,000 digits, all deletions in runs; 2.5 per 100 on run-heavy strings). It
stays that way down to 3.4 bpw: Q2_K_imat makes 143 deletions, 127 of them in runs. The
looping that hit the 4B at 3.6–4.3 bpw (UD-Q2_K_XL 28, Q3_K_M 10 errors per 100) is gone. Only
UD-IQ2_XXS loops (5,216 insertions).

## H4: builds

| Type | Our plain build: KL on π / random | unsloth's file: KL on π / random |
|---|---|---|
| Q6_K | 0.00010 / 0.00026 | 0.00008 / 0.00023 |
| Q5_K_M | 0.00039 / 0.0016 | 0.00020 / 0.00050 |
| Q4_K_M | 0.0058 / **0.0096** | 0.0046 / **0.00093** |
| Q3_K_M | 0.13 / 0.0057 | 0.049 / 0.0026 |
| Q2_K | 2.39 (+ our imatrix: 0.99) | UD-Q2_K_XL: 0.77 |

The recipe gap is larger on the 9B. At Q4_K_M, generic (random-digit) noise is 10× higher in
our plain build. At Q3_K_M, π noise is 2.7× higher. Legacy Q4_0 (unsloth) has 1.7× the π noise
of unsloth's Q4_K_M, a smaller gap than on the 4B (3.3×, our two builds). In unsloth's set,
IQ4_XS has the lowest π noise of any file under 5 bpw (0.0028) and recites 576 digits, but it decodes at
38 tokens/s.

## H6: nothing to arbitrate

E5 plants one wrong digit in a copy of π or e. The 4B "corrected" π position 10 back to the
memorized digit at high bits. The 9B never does: BF16 and every build except plain Q2_K and
UD-IQ2_XXS copy all 8 planted digits faithfully. The only non-faithful outputs are copy failures (plain Q2_K 3 of 8,
UD-IQ2_XXS 1 of 8), not memorized digits. The bigger model follows the prompt over memory. H6's
premise, that memory competes with the prompt, does not hold at 9B, so there is no shift to
measure.

## Resources and speed

One Arc Pro B60, the same settings as Phase 1. VRAM is now the **steady median** during the
tests, not the peak, because a few runs showed a brief spike of about 750 MiB while the model
loaded. Phase 1's numbers barely move under the new measure (by 0–7 MiB).

| Variant | File GB | VRAM GiB | Decode tok/s | Prompt tok/s (4k) | Power W | Host RSS GiB |
|---|---|---|---|---|---|---|
| BF16 | 17.92 | 15.7 | 22 | 593 | 123 | 18.8 |
| us_Q8_0 | 9.53 | 8.7 | 42 | 1,964 | 132 | 10.1 |
| Q6_K | 7.36 | 6.9 | 50 | 1,987 | 161 | 7.9 |
| Q5_K_M | 6.47 | 6.2 | 53 | 2,067 | 157 | 7.0 |
| Q4_K_M | 5.63 | 5.5 | 60 | 1,821 | 152 | 6.2 |
| us_UD-Q4_K_XL | 5.97 | 5.8 | 52 | 1,833 | 146 | 6.6 |
| us_IQ4_XS | 5.17 | 5.1 | 38 | 1,318 | 135 | 5.8 |
| Q3_K_M | 4.62 | 4.7 | 57 | 1,461 | 154 | 5.3 |
| us_UD-IQ3_XXS | 4.02 | 4.1 | 20 | 982 | 117 | 4.6 |
| Q2_K | 3.83 | 4.1 | 61 | 1,669 | 150 | 4.6 |
| us_UD-IQ2_XXS | 3.19 | 3.4 | 31 | 996 | 137 | 3.8 |

- **VRAM is less than file + 770 MiB.** The 9B's token-embedding table is separate from its
  output layer (the 4B shares one), and llama.cpp keeps it in host memory. The overhead over the
  file size runs from −1.0 GiB at BF16 to +0.5 GiB at Q2_K, which matches the size of that table
  at each type. Q4_K_M fits an 8 GB card with an 8k context. BF16 needs 16 GB.
- **Decode flattens below ~5 bpw again**: 22 → 42 → 60 tokens/s from BF16 to Q8_0 to Q4_K_M,
  then 57–61 down to Q2_K. The 9B is 1.5–1.7× slower than the 4B at the same type.
- **IQ types**: UD-IQ3_XXS (20 tokens/s) and UD-IQ2_XXS (31) are the slowest quants, and
  IQ3_XXS is slower than BF16 itself.
- **Host RAM** still tracks file size (memory-mapped). BF16 peaks at 18.8 GiB of this
  machine's 30 GB.

## Problems found and how they were handled

| Issue | Effect | Handling |
|---|---|---|
| Peak VRAM caught load-time spikes (~750 MiB) | Q3_K_M looked 1.1 GiB over its file size | Summaries record the steady median too; analysis uses it, for both models |
| BF16 recites π past the 1,200-token cap | The 9B reference is a lower bound | Reported; it doesn't affect the cliff (half the reference) |
| e prompts read as a list or as code | Median e score of 15 is a prompt artifact | Reported per phrasing; π carries H1 |
| E6 (recall as words) | Still at the floor: π comes out as fragments ("one five nine two six five…") from the wrong start | Reported as a floor for 9B too |

## Next

Phase 3 is the 27B (Qwen 3.8 27B), with a Q8_0 reference built from the FP8 weights and a
reduced set. Questions still open from Phase 1:

- Drop E6? It has been at the floor at 4B and 9B.
- Add `--no-mmap` so host RAM measures what the model really needs?
- Should the imatrix calibration text include some numbers?
- Change the seed `20261027`? It resembles a date, and changing it means rerunning everything.
