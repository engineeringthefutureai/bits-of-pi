# Phase 3 report: Qwen 3.8 27B, and three model sizes

Kiryl Dubarenka

Phase 1 (4B): [REPORT.md](REPORT.md). Phase 2 (9B): [REPORT_9B.md](REPORT_9B.md). Web version:
[engineeringthefutureai.github.io/bits-of-pi](https://engineeringthefutureai.github.io/bits-of-pi/)

17 variants of the 27B, from Q8_0 down to 2.1 bits per weight. The setup matches Phases 1 and 2:
same llama.cpp build (`bed0a8566`, SYCL), same inputs, greedy decoding. Two things differ:

- **The reference is unsloth's Q8_0, not BF16.** BF16 is 54.7 GB and does not fit in two 24 GB
  cards. On the 4B and 9B, Q8_0 recited π character for character like BF16 in every phrasing,
  and the 9B's Q8_0 sat 0.000007 nats from BF16 on π. KL in this report is measured against
  Q8_0, which understates the true distance from BF16 by about that much.
- **The three largest files (Q8_0, Q6_K, UD-Q6_K) are split over both Arc Pro B60s.** The rest
  run on one card, as before.

The variant set:

- Our plain ladder (Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K), built from unsloth's BF16 exactly as on
  the 4B and 9B, plus Q2_K with our imatrix (computed from the Q8_0).
- 11 of unsloth's files: Q8_0 (the reference), Q4_0, and nine UD (Unsloth Dynamic) files from
  UD-Q6_K to UD-IQ2_XXS.

E0 (determinism, single-digit tokens, thinking off) passed on Q8_0 and Q6_K. The original E4 is
skipped, as on the 9B. The full table is
[`scores/qwen3.8-27b_headline.md`](scores/qwen3.8-27b_headline.md).

The 27B is a newer generation (Qwen 3.8) than the 4B and 9B (Qwen 3.5). Size is not the only
thing that changes in H5.

## Summary

| Hypothesis | 27B verdict | Deciding evidence |
|---|---|---|
| H1 recall cliff near 3 bits | **Not a plateau; cliff at 4.9 bpw** | π recall holds at the reference's 1,046 digits to 5.1 bpw, then falls to 463 at Q4_K_M ([H1](figures/qwen3.8-27b_H1_recall.png)) |
| H2 margins explain flips | **Confirmed, strongest yet** | Each 3–5 bit variant's usual free-recall stop is the first digit it gets wrong under teacher forcing; the median stop sits at the 2nd percentile of Q8_0's margins ([H2](figures/qwen3.8-27b_H2_margins.png)) |
| H3 copying is robust | **Confirmed, strongest yet** | Even plain Q2_K copies 17,000 digits without an error; only UD-IQ2_XXS drops digits ([E3](figures/qwen3.8-27b_H3_E3.png), [E4b](figures/qwen3.8-27b_H3_E4b.png)) |
| H4 recipe matters | **Confirmed at 4 bits and below** | unsloth's UD-Q4_K_M has half the π noise of our Q4_K_M and recites 990 digits against 463 ([H4](figures/qwen3.8-27b_H4_builds.png)) |
| **H5 size buys headroom** | **Only at 2–4 bits, and not per byte** | At 5–6 bits the 27B's π noise equals the 9B's; at Q3_K_M it is 3.5× lower, at Q2_K 4× lower. Per GB it is the worst of the three ([H5](figures/H5_model_size.png)) |
| H6 low bits revert to memory | **Nothing to revert** | Every variant, down to UD-IQ2_XXS, copies all 8 planted digits ([H6](figures/qwen3.8-27b_H6_planted.png)) |

Practical reading for the 27B on this hardware:

- **Q5_K_M (19.5 GB) is the smallest file that keeps all the π the model knows.** It fits one
  24 GB card and decodes at 18 tokens/s.
- **Q4_K_M already loses half.** Pick unsloth's UD-Q4_K_M at that size: 990 digits against 463.
- **Copying survives everything except UD-IQ2_XXS**, including plain Q2_K, which broke on both
  smaller models.
- **For exact recall per GB of memory, the 27B is the worst choice.** The 4B at Q8_0 recites
  1,000 digits from 4.5 GB, the 9B 1,056 from 7.4 GB. The 27B needs 19.5 GB for 1,046.
- **UD files with IQ types decode slowly on SYCL**: 8–11 tokens/s against 20–22 for the plain
  K-quants of similar size.

## H1: the 27B knows fewer digits of π than the 9B

E1 asks for π, e and √2 in three phrasings each.

| Constant | 27B Q8_0, by phrasing | 9B BF16 | 4B BF16 |
|---|---|---|---|
| π | **1046 / 1046** / 463 | 1000 (stops as asked) / 1200+ / 1180 | 1000 / 1000 / 821 |
| e | 50 / **192** / 45 | list format / **192** / 15 | 1,200 zeros / **192** / 211 |
| √2 | 65 / 105 / 98 | 219 / 202 / 203 | 176 / 105 / 102 |

**Q8_0 stops at π digit 1,046 in two phrasings.** After `…2796823030` the true digits are
`195203530185…`, and the 27B writes `897928530781…`. This is a gap in the model's memory, not
quantization damage:

- Q6_K, UD-Q6_K, Q5_K_M, UD-Q5_K_M and UD-Q4_K_XL stop at exactly the same digit. Their KL on π
  is 0.0001–0.0003 nats.
- On the 4B and 9B, Q8_0 and BF16 produced identical π recitations.
- The 9B gets past digit 1,046 and reaches the 1,200-digit cap.

So the larger, newer model has memorized fewer digits of π than the 9B. The ceiling for the
27B is a real limit at 1,046, where the 9B's ceiling was the generation cap. The third phrasing
stops at 463, one of Q8_0's thinnest margins (H2).

e repeats the pattern of the smaller models. All three know exactly 192 digits from the plain
phrasing, and the other phrasings give lists, code or early stops. Only π carries H1.

π recall moves in steps: 1,046, 990, 463, 448, 437, 271, 167 and 163. Each step is a specific
digit, and each one is also the first digit that variant gets wrong under teacher forcing (H2).

## H2: margins predict every stop

Q8_0 gets all 1,000 teacher-forced positions of π right. Its margin profile sits between the 4B
and the 9B:

| Model (reference) | Thinnest margin | Median | Positions < 3 nats | < 5 nats | Mean, first 100 → after 300 |
|---|---|---|---|---|---|
| 4B (BF16) | 0.1 nats (pos 813) | 9.2 | 14 | 73 | 5.6 → 10.0 |
| 9B (BF16) | 1.6 nats (pos 10) | 10.3 | 2 | 15 | 6.6 → 11.2 |
| 27B (Q8_0) | 1.1 nats (pos 10) | 9.3 | 6 | 20 | 6.6 → 10.0 |

The 27B's six thinnest positions are 10, 200, 50, 437, 100 and 463 (1.1–2.9 nats). The 4–5 bit
variants stop on them, or on digits almost as thin:

| Variant | First teacher-forced flip | Q8_0 margin there | Percentile | E1 π stops |
|---|---|---|---|---|
| UD-Q4_K_M | 990 | 3.1 | 1st | 990 / 271 / 990 |
| Q4_K_M | 463 | 2.9 | 0th | 463 / 463 / 463 |
| Q4_0, UD-Q3_K_XL | 437 | 2.8 | 0th | 437 |
| Q3_K_M | 271 | 4.2 | 1st | 271 / 271 / 271 |
| UD-IQ4_XS | 448 | 9.3 | 50th | 448 / 327 / 448 |
| UD-IQ3_XXS | 167 | 7.5 | 21st | 167 / 111 / 167 |
| UD-Q2_K_XL | 163 | 8.2 | 32nd | 163 |
| Q2_K | 75 | 5.5 | 3rd | 70 / 58 / 43 |

Over all 36 early π stops in free recall, the median stop lies at the **2nd percentile** of
Q8_0's margins, and 61% lie in the lowest 10%. One script computes the figures for all three
models: 12th percentile and 43% on the 4B, 6th and 61% on the 9B. (The 9B report's 6th and 57%
came from an earlier version of the count.)

The mechanism is the same on all three models. Quantization noise grows as bits drop, and the
first digits to fall are the ones the reference holds least firmly. The IQ and 2-bit files are
noisy enough to flip digits with ordinary margins (7–9 nats). The 4–5 bit files only flip the
thin ones.

One pattern deserves a follow-up. Thin margins cluster at round positions: 10, 50, 100 and 200
on the 27B; 10, 100 and 210 on the 4B; 0, 5, 10 and 20 on the 9B. π is often printed in blocks
of 10, 50 or 100 digits, so line breaks in the training data may weaken recall at block
boundaries.

## H3: copying

**Verbatim copy (E3)**, 17,000 digits per variant:

| Variant | bpw | Errors | Kind |
|---|---|---|---|
| Q8_0 and 14 others, down to UD-Q2_K_XL | 8.5 → 2.9 | 0–1 | one deletion inside a run |
| Q2_K, no imatrix | 3.2 | **0** | — |
| UD-IQ2_XXS | 2.1 | 104 (2.5 per 100) | deletions of chunks |

Plain Q2_K broke on both smaller models (59 and 34 errors per 100, by stopping early). On the
27B it copies perfectly.

**Spelling as words (E4b)**, 7,000 digits per variant: Q8_0 makes 23 errors, all deletions
inside runs of equal digits (the 9B's BF16 made 78). That stays nearly flat down to 2.9 bpw
(UD-Q2_K_XL: 22). Plain Q2_K adds 17 insertions, and UD-IQ2_XXS doubles the deletions to 71.
There is no looping at any bit width.

## H4: builds

| Type | Our plain build: KL on π / random | unsloth's file: KL on π / random |
|---|---|---|
| Q6_K | 0.00026 / 0.00044 | UD-Q6_K: 0.00014 / 0.00023 |
| Q5_K_M | **0.00014** / 0.00077 | UD-Q5_K_M: 0.00024 / 0.00042 |
| Q4_K_M | 0.0070 / 0.0022 | UD-Q4_K_M: **0.0033** / 0.0014 |
| Q3_K_M | 0.037 / 0.0063 | UD-Q3_K_XL: **0.015** / 0.0051 |
| Q2_K | 0.60 (+ our imatrix: 0.45) | UD-Q2_K_XL: **0.21** |

At 5–6 bits the recipe makes no consistent difference, and our Q5_K_M has the lower π noise. From
4 bits down, unsloth's per-layer allocation halves the π noise or better, and at Q4 that is the
difference between 990 and 463 digits. Our imatrix helps Q2_K less than on the smaller models
(0.60 → 0.45, against 2.39 → 0.99 on the 9B). Legacy Q4_0 (unsloth) has 1.25× the π noise of
UD-Q4_K_M. UD-IQ4_XS has 1.8× the π noise of Q4_0 here, though it was the best sub-5-bit file on
the 9B.

## H5: what model size buys

The plan defines the cliff as the highest-bpw ladder variant whose median π recall falls below
half the reference's.

| Plain ladder | 4B bpw | π | KL on π | 9B bpw | π | KL on π | 27B bpw | π | KL on π |
|---|---|---|---|---|---|---|---|---|---|
| reference | 16.0 | 1000 | 0 | 16.0 | 1180 | 0 | 8.5 † | 1046 | 0 |
| Q8_0 | 8.5 | 1000 | 0.00013 | 8.5 † | 1200 | 0.000007 | | | |
| Q6_K | 6.6 | 813 | 0.0011 | 6.6 | 1056 | 0.00010 | 6.6 | 1046 | 0.00026 |
| Q5_K_M | 5.8 | **428** | 0.0075 | 5.8 | 1090 | 0.00039 | 5.7 | 1046 | 0.00014 |
| Q4_K_M | 5.2 | 123 | 0.039 | 5.0 | 788 | 0.0058 | 4.9 | **463** | 0.0070 |
| Q3_K_M | 4.3 | 88 | 0.35 | 4.1 | **129** | 0.13 | 4.0 | 271 | 0.037 |
| Q2_K | 3.6 | 0 | 2.30 | 3.4 | 0 | 2.39 | 3.2 | 58 | 0.60 |

† unsloth's Q8_0. 27B KL is measured against Q8_0 (see the top of this report).

The cliff is Q5_K_M on the 4B, Q3_K_M on the 9B and **Q4_K_M on the 27B**. Read as a test of
"bigger is more robust", H5 holds only in part:

- **At 5–6 bits the 27B is no less noisy than the 9B.** KL on π is 0.0001–0.0003 for both. The
  9B's lead over the 4B (10–19× less noise) does not grow again at 27B.
- **From 4 bits down the 27B pulls ahead.** KL on π at Q3_K_M is 0.037 against 0.13 (9B) and 0.35
  (4B), and at Q2_K 0.60 against 2.3–2.4. The 4B and 9B curves met at Q2_K; the 27B's stays 4×
  below both. Fitted over the same range (Q6_K to Q2_K), its π noise grows 12× per bit, against
  26× for the 9B and 13× for the 4B. The 27B's π recall at Q3_K_M (271) and Q2_K (58) is the best
  of the three.
- **The cliff depends on where the thin digits are, not just on the noise.** The 27B's Q4_K_M
  has about the same π noise as the 9B's (0.0070 vs 0.0058). It first flips position 463, where
  Q8_0's margin is 2.9 nats. The 9B's Q4_K_M survives its own 3.6-nat digit at 267 and first
  flips at 788. The 27B has more thin digits early in π (6 under 3 nats, against 2 for the 9B), so
  similar noise costs it more digits. Margin is the main predictor but not the only one: noise
  is not spread evenly over positions. How many thin digits a model has depends on its training,
  and the 27B is a different generation.
- **Per byte, the order reverses completely.** The 4B's Q8_0 recites 1,000 digits from 4.5 GB
  and the 9B's Q6_K 1,056 from 7.4 GB. At about 11 GB the 27B's Q2_K recites 58, and the 27B
  first reaches its own ceiling at 19.5 GB. The top-right
  panel of the [H5 figure](figures/H5_model_size.png) shows the three curves moving right
  without rising.

The Phase 2 conclusion holds with a third size: **for exact recall, bits per weight matter more
than weights**. A bigger model does help at very low bit widths, where it degrades more slowly.

## H6: nothing to arbitrate

E5 plants one wrong digit in a copy of π or e. Every 27B variant, down to UD-IQ2_XXS, copies
all 8 planted digits. As on the 9B, the model follows the prompt over memory, so there is no
shift to measure.

## E6: recall as words is no longer at the floor

On the 4B and 9B, reciting π as English words produced fragments from the wrong start. The 27B
does it properly: Q8_0 says "one four one five nine two six five three five…" and is correct
for 46 digits of π, 24 of e and 27 of √2. Q5_K_M matches it exactly. Q3_K_M drops to 14, 29
and 0, and Q2_K and below reach 0–3. E6 measures something on the 27B, so it should stay in the
plan.

## Resources and speed

| Variant | GPUs | File GB | VRAM GiB | Decode tok/s | Prompt tok/s (4k) | Power W |
|---|---|---|---|---|---|---|
| us_Q8_0 | 2 | 29.05 | 27.7 | 13 | 1,093 | 150 |
| Q6_K | 2 | 22.43 | 21.9 | 17 | 1,098 | 184 |
| us_UD-Q6_K | 2 | 21.98 | 21.5 | 16 | 1,091 | 176 |
| Q5_K_M | 1 | 19.54 | 18.5 | 18 | 687 | 165 |
| us_UD-Q4_K_XL | 1 | 17.56 | 16.7 | 16 | 552 | 153 |
| Q4_K_M | 1 | 16.81 | 16.1 | 22 | 594 | 161 |
| us_UD-Q4_K_M | 1 | 16.46 | 15.7 | 15 | 502 | 149 |
| us_Q4_0 | 1 | 16.06 | 15.4 | 21 | 546 | 140 |
| us_UD-IQ4_XS | 1 | 14.25 | 13.8 | 11 | 402 | 140 |
| Q3_K_M | 1 | 13.50 | 13.2 | 20 | 459 | 164 |
| us_UD-IQ3_XXS | 1 | 10.93 | 10.8 | 8 | 327 | 130 |
| Q2_K | 1 | 10.86 | 11.0 | 22 | 539 | 161 |
| us_UD-Q2_K_XL | 1 | 9.83 | 9.8 | 8 | 324 | 132 |
| us_UD-IQ2_XXS | 1 | 7.27 | 7.8 | 10 | 316 | 135 |

- **Split variants:** VRAM and power are summed over both cards, and the speed is two-card
  speed. A split does not speed up decode, because the cards take turns, layer by layer: Q6_K
  on two cards decodes at 17 tokens/s, like Q5_K_M on one (18). Prompt processing is higher
  (1,098 against 687 tokens/s at 4k). The speed figure marks these points.
- **Decode** is 13 tokens/s at Q8_0 and 18–22 for the plain K-quants from Q5_K_M down, about 2.8×
  slower than the 9B at the same type.
- **UD files are slow on SYCL**: UD-Q4_K_XL, UD-Q4_K_M, UD-Q3_K_XL and UD-Q2_K_XL decode at
  8–16 tokens/s against 20–22 for our plain builds of similar size. Their per-layer mixes include
  IQ tensors, which this backend runs slowly.
- **One card is enough for Q5_K_M (18.5 GiB) and below**, with an 8k context.
- **The file includes an unused layer.** Qwen 3.8 ships a multi-token-prediction layer
  (`blk.64`) that llama.cpp ignores. It is 0.45 GB of the Q8_0 (1.5%) and 0.15 GB of Q2_K. File
  size and bpw include it, which shifts every 27B point right by about 0.1 bpw.
- **Host RAM is not comparable across the table.** Q8_0 and Q6_K ran with the file
  memory-mapped (host RSS 27.1 and 22.1 GiB). Every later variant ran without mmap (about
  0.9 GiB); see below.

## Problems found and how they were handled

| Issue | Effect | Handling |
|---|---|---|
| The machine ran out of RAM twice with the 27B memory-mapped (22–29 GB on a 30 GB host) | The kernel's OOM killer took down the run and the user session | The 27B loads without mmap (`--load-mode none`, a per-model `mmap: false` setting); results are unchanged and host RSS drops to ~0.9 GiB. The run also raises its own OOM score so it is killed before the session |
| A teacher-forced dump was cut short by the OOM kill and looked complete | The truncated file would have been reused | The skip check now looks for `tokens.json`, which `tf_dump` writes last |
| `llama-bench` aborts when freeing a context on a two-GPU split | The decode benchmark never ran | Each benchmark test runs in its own process; a printed row is kept whatever the exit status |
| No BF16 reference fits | KL is measured against Q8_0 | Q8_0 tracked BF16 exactly on the 4B and 9B; stated wherever KL appears |
| Unused MTP layer in the file | File size and bpw read about 1.5% high | Reported; too small to change any conclusion |

## Next

Questions still open:

- Keep E6? It is at the floor on the 4B and 9B but measures something on the 27B.
- Should the imatrix calibration text include some numbers? Our imatrix helped the 27B's Q2_K
  less than the smaller models'.
- Change the seed `20261027`? It resembles a date, and changing it means rerunning everything.
- Are thin margins really concentrated at block boundaries (positions 10, 50, 100, 200)? Checking
  this needs only the existing teacher-forced data.
