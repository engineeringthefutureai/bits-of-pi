# Phase 1 report: Qwen 3.5 4B

Kiryl Dubarenka

Web version: [bits-of-pi.engineeringthefuture.ai](https://bits-of-pi.engineeringthefuture.ai/)

A smaller model file is cheaper to run. What does it cost you? Some answers hang on one token
chosen by a hair, like the 327th digit of π or a single digit in a long account number, and this
study measures how compression damages exactly those.

I ran 28 versions of one model, Qwen 3.5 4B, from the full-size original at 16 bits per weight
down to files compressed to 2.9. All of them went through the same llama.cpp build (`bed0a8566`,
SYCL backend, one Intel Arc Pro B60). Decoding was greedy throughout. The model always took its
most likely next token. Every number below comes out of `scripts/analyze.py`, which reads the
scored data in `scores/`, the per-digit records in `tf/` and the per-version summaries in `runs/`.
The full table for all 28 versions is
[`scores/qwen3.5-4b_headline.md`](scores/qwen3.5-4b_headline.md).

## Words used in this report

[QUANTIZATION_GUIDE.md](QUANTIZATION_GUIDE.md) explains these ideas at more length, along with
how every file was built.

**Quantization** stores each of the model's numbers (its weights) in fewer bits. The file gets
smaller and faster, and every calculation picks up a little rounding error.

**Bits per weight (bpw)** is the average number of bits a weight takes in the file, measured as
file size × 8 ÷ number of weights, with the original at 16. On a model this small the measured
value sits 0.3–0.4 above the name on the tin, since the large vocabulary layers keep higher
precision. Q4_K_M measures 5.15 bpw here, for example, against a nominal 4.8.

**Version names** come from llama.cpp:

- **BF16** is the uncompressed original at 16 bits per weight. It is the **reference** that every
  other version is measured against.
- **Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K** are the standard compressed formats, and the
  number is roughly the bits per weight. **K** marks llama.cpp's newer "K-quant" family. A
  trailing **_S, _M, _L** or **_XL** picks a size within the format (small, medium, large, extra
  large).
- **Q4_0** is an older ("legacy") 4-bit format.
- **IQ4_XS, IQ3_XXS, IQ2_XS, IQ2_XXS** are "I-quants", which pack weights through lookup tables
  to save more space.
- **us_** marks a file downloaded from unsloth, a group that publishes compressed models. No
  prefix means I built it with llama.cpp's own tool.
- **UD-** stands for "Unsloth Dynamic". unsloth sets the precision layer by layer and spends more
  bits where they count.
- An **imatrix** (importance matrix) records which weights matter most on a sample text, so the
  compression can protect them. **_imat** means my own imatrix, built from Wikipedia text with
  every number taken out. **_usimat** means my build with unsloth's imatrix.
- The **plain ladder** ("DIY, no imatrix" in the figures) is my set of builds without an
  imatrix: Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M and Q2_K. Only the bit width changes along it, so
  it gives the cleanest comparison.

**The tests:**

- **E0** checks the setup. Each digit has to be its own token, the model's "thinking" mode has to
  be off, and repeated runs have to match.
- **E1** asks for π, e and √2 from memory in three different wordings and scores the digits
  before the first mistake. Figures show the **median of 3**, the middle of the three wordings.
- **E2** is **teacher forcing**. The model writes nothing. I feed it the true digits one at a
  time and record, at every position, how strongly it expected the right one.
- **E3** asks for a copy of a random digit string from the prompt.
- **E4b** asks for a digit string written out as words ("507" becomes "five zero seven"). It
  replaces E4, a first try at the prompt that failed as a test for reasons given under H3.
- **E5** plants one wrong digit in π or e and asks for an exact copy. Copy it, or "fix" it from
  memory?
- **E6** asks for π, e and √2 from memory, as words.

**Measures from teacher forcing (E2):**

- **Top-1** is the digit the model would pick at a position. A **flip** is a position where a
  compressed version's top-1 differs from the reference's.
- **Margin** is how sure the model is of a digit, given here as **odds**: how many times more
  likely the right digit is than the runner-up. 1× is a dead heat. 4× is shaky, and 10,000× is
  certain. (For technical readers, the raw margin is a log-probability gap in nats, and the odds
  are e raised to it.)
- **Percentile of margin** ranks one digit among all 1,000. At the 2nd percentile only 2% of
  digits are less certain.
- **Noise** is my plain name for the **KL divergence**, the average drift of a compressed
  version's predictions away from the reference's over every position. It has no everyday unit,
  so read it against what it did in this study:

  | Noise | What it looked like here |
  |---|---|
  | 0.0001 | recites exactly like the reference. Only a near-tie digit can flip |
  | about 0.001 | one or two of the least certain digits of π flip |
  | about 0.01 | π recall stops somewhere in the first few hundred digits |
  | about 0.1 | π recall drops to around 100 digits |
  | 1 and above | π recall is close to zero, and some builds can't copy either |

  "Noise on π" is measured on π's digits, which the model memorized. "Noise on random" is
  measured on random digits nobody could have memorized, so it shows generic damage.

**Other measures.** Decode speed is how fast the model writes, in tokens per second (tok/s), and
prompt speed how fast it reads a 4,096-token prompt (pp4096). VRAM is graphics memory used,
in GiB (1 GiB = 1.07 GB). A copy error is a **substitution** (wrong digit), an **insertion**
(extra digit) or a **deletion** (missing digit). A **run** is a stretch of one repeated digit,
like 8888.

**Log scale.** Several figures plot noise on a log scale. Each gridline there is 10 times the one
below it, so a point one gridline lower has a tenth of the noise.

**Marker shapes in the figures** (the same in every figure for every model):

| Marker | Meaning |
|---|---|
| blue circle, joined by a line | plain ladder (my builds, no imatrix) |
| orange square | my build with my own imatrix |
| green diamond | unsloth's published file |
| dark triangle, point up | my build with unsloth's imatrix |
| dark triangle, point down | legacy Q4_0 |
| black star, or dashed black line | the reference |

Only the plain ladder gets a line. The other groups mix formats, and a line through them would
draw a trend that doesn't exist.

## Summary

| Hypothesis | Verdict | What decided it |
|---|---|---|
| H1: recall survives down to about 4 bits, then falls off a cliff | **Partly supported, with a much higher threshold** | π recall holds through Q8_0 at 8.5 bpw, then declines step by step from 6.6 bpw down, with no sharp cliff. Q4_K_M, which measures 5.2 bpw, keeps 123 of 1,000 digits ([H1](figures/qwen3.5-4b_H1_recall.png)) |
| H2: errors start where the model was least sure | **Confirmed** | Recall stops on the digits the reference was least sure of, and noise on π grows 7.6× per bit removed ([H2](figures/qwen3.5-4b_H2_margins.png), [KL](figures/qwen3.5-4b_H2_kl.png)) |
| H3: copying from the prompt holds up, and when it fails it skips digits in runs | **Confirmed** | Every build at 4.3 bpw or more copies almost perfectly. Errors are missing digits inside runs, even at BF16 ([E3](figures/qwen3.5-4b_H3_E3.png), [E4b](figures/qwen3.5-4b_H3_E4b.png)) |
| H4: the compression scheme matters at equal size | **Confirmed, with a twist** | K-quants beat Q4_0 by 3.3× on π noise. A number-free imatrix cuts generic noise and leaves π noise where it was. unsloth's recipe beats all my builds ([H4](figures/qwen3.5-4b_H4_builds.png)) |
| H5: bigger models have more room to compress | **Confirmed per bit** | At Q5_K_M the 9B and 27B keep their full π recall while the 4B keeps 428 of 1,000. Details in [Phase 2](REPORT_9B.md) and [Phase 3](REPORT_27B.md) |
| H6: compressed models fall back on memory over the prompt | **Rejected, reversed** | The more compressed versions copy the planted digit *more* faithfully ([H6](figures/qwen3.5-4b_H6_planted.png)) |

If you run this model on this card:

- **Q8_0 costs nothing.** Its answers match BF16 on every test (noise 0.0001), it decodes 1.7×
  faster, and it needs 43% less graphics memory.
- **For exact recall, stay at Q6_K or above.** Every bit removed multiplies the noise on
  memorized digits by about 7.6, and π recall halves somewhere between Q6_K and Q5_K_M.
- **For copying an answer that sits in the prompt, almost anything works** down to Q3_K_M, and
  down to Q2_K with an imatrix. Plain Q2_K breaks. So do my IQ3_XXS and IQ2_XS builds.
- **Prefer unsloth's file to a standard build of the same format.** At Q4_K_M and Q3_K_M it has
  about half the noise on π for the same size and speed.
- **Avoid IQ formats on this backend.** IQ3_XXS decodes at 35–38 tokens/s, which is slower than
  uncompressed BF16.

## H1: recall holds only at high bits, then declines steadily

**Why I expected recall to hold to about 4 bits.** Ollama is for many people the first way they
run a model on their own machine, and most models in its library download as Q4_K_M by default. I
read that default as a deliberate balance between quality and resources. For me a good balance
means answers close to the original's. If that held, a Q4 file should recite π about as far as
the original. On none of the three models did it: Q4_K_M lost digits every time, and full recall
needed Q5_K_M on the 9B and 27B and Q8_0 on the 4B, where Q6_K kept 813 of 1,000. Copying from
the prompt (H3) is the exception, and there Q4_K_M stays close to the original on every model.

Asked for π, BF16 recites **1,000 correct digits** and stops, because the prompt asked for 1,000.
It knows 192 digits of e and only 105 of √2, which leaves π as the one constant with room to
watch recall shrink.

![π recall against bits per weight for the 28 versions of the 4B](figures/qwen3.5-4b_H1_recall.png)

*How to read it:* each marker is one version. Across is bits per weight, with the most compressed
files on the left, and up is the number of π digits recited before the first mistake, the middle
of three wordings. Whiskers run from the worst wording to the best, the dashed line is BF16, and
shapes follow the table above.

| Plain ladder | bpw | π digits (median of 3) | Noise on π | Noise on random digits |
|---|---|---|---|---|
| BF16 (reference) | 16.0 | 1000 | 0 | 0 |
| Q8_0 | 8.5 | 1000 | 0.00013 | 0.00008 |
| Q6_K | 6.6 | 813 | 0.0011 | 0.0012 |
| Q5_K_M | 5.9 | 428 | 0.0075 | 0.0012 |
| Q4_K_M | 5.2 | 123 | 0.039 | 0.0033 |
| Q3_K_M | 4.3 | 88 | 0.35 | 0.0059 |
| Q2_K | 3.6 | 0 | 2.30 | 0.045 |

Recall does survive compression, but only down to Q8_0 at 8.5 bits per weight, far above the 4
bits H1 expected. The name Q4_K_M is misleading here: the file measures 5.2 bits per weight, and
Q3_K_M at 4.3 is the closest thing to 4 bits on the ladder. Below Q8_0 recall falls in steps.
Q5_K_M, at 5.9 bits per weight, already recites less than half of what BF16 does. The noise
climbs smoothly as bits drop. Recall jumps because of a handful of specific digits (see H2), so
where each step lands depends on those digits.

Memorized digits are fragile. At Q4_K_M the noise on π is **12 times** the noise on random
digits. Every bit removed multiplies the noise on π by 7.6 and the noise on random digits by 3.2.

Two versions with almost the same noise can recite very different amounts: my Q6_K gets 813
digits and unsloth's Q6_K 326, at noise 0.0011 and 0.0013. Both teeter on digit 326, and only mine
gets past it. Noise is the steadier measure. Recall shows what that noise does to an answer.

## H2: errors start where the model was least sure

Teacher forcing walks the model through the true 1,000 digits of π. At each position I record how
far the right digit led the runner-up, which is the margin.

- BF16 gets all 1,000 right. For the median digit the right answer is about 10,000 times as
  likely as the runner-up. A few digits are close to tied. At position 813 the right one leads by
  only 1.1 times, at 326 by 3.8 and at 10 by 4.1.
- Free recall breaks on those same digits. Of 77 π recitations that stopped early, the commonest
  stopping points are 326 (8 times) and 88 (7), followed by 821, 10, 87 and 123 at 4 each. The
  typical stop sits at the 12th percentile of BF16's margins.
- Lightly compressed versions flip only the least certain digits. Q6_K's two teacher-forced
  errors are at 813 and 821, both in the least certain 0.1%. Q4_K_M's 14 sit around the 7.6th
  percentile. From Q3_K_M (104 errors) down, the noise outgrows most margins and errors turn up
  everywhere.

![BF16's margin along π, and where each version gets a digit wrong](figures/qwen3.5-4b_H2_margins.png)

*How to read it:* the top panel shows how sure BF16 is of each of the 1,000 digits of π, as odds
against the runner-up on a log scale. 1× is a tie and each labelled line is 100 times the one
below. Grey traces every digit. The blue line smooths it over 25 digits. Deep grey dips are digits
the model barely knows, and the whole trace rises from about 100× near the start to about 10,000×
further in. The bottom panel has one row per version, least compressed first, with a dot wherever
that version's top-1 digit is wrong under teacher forcing. Rows near the top have dots only under
the thin-margin positions (326, 813). Lower down, the rows fill in.

![Noise on the π text against bits per weight](figures/qwen3.5-4b_H2_kl.png)

*How to read it:* up is the noise on the π text, on a log scale with each gridline 10 times the
one below, and across is bits per weight. BF16 is missing because its noise is zero by definition.
The plain ladder runs in a nearly straight line. On a log scale, that means the noise multiplies
by a fixed factor with every bit removed.

## H3: copying holds, and it fails by skipping digits in runs

**Copying digits (E3).** The test has 40 random strings of 100 to 1,000 digits, 17,000 digits in
all. BF16 makes 3 errors, each a single extra digit. Every build at 4.3 bits per weight or more
copies the 1,000-digit strings perfectly. Three builds break:

| Version | bpw | Errors per 100 digits | On 1,000-digit strings | What goes wrong |
|---|---|---|---|---|
| Q2_K (no imatrix) | 3.6 | 59 | 99 | mostly missing digits: the model gives up |
| IQ3_XXS (my imatrix) | 3.6 | 9.0 | 36 | whole chunks dropped |
| IQ2_XS (my imatrix) | 3.1 | 12 | 39 | mixed |

Built another way, the same formats work. Q2_K with an imatrix makes 0.06 errors per 100 digits,
and unsloth's UD-IQ3_XXS 0.03.

**Writing digits as words (E4b).** Here BF16 drops digits too, and always inside runs. Strings
full of runs get 2.2 errors per 100 digits against 0.3 for ordinary random strings. 75 of BF16's
80 errors are one word missing from a run of three or more equal digits, so "eight eight eight
eight eight" comes out one "eight" short. That holds down to about 4.3 bits per weight. Below it
the errors turn into **extra** words as the model gets stuck repeating one, and Q2_K_imat makes
208 errors per 100 digits on run-heavy strings, 5,297 of them extra words. Wrong digits stay rare
except in builds that fall apart entirely.

![Spell-as-words errors per version, split by kind](figures/qwen3.5-4b_H3_E4b.png)

*How to read it:* one bar per version, reference at the top. Bar length is the total number of
errors over all E4b strings, split into wrong digits (blue), extra digits (orange) and missing
digits (green). Most bars are too short to see. Plain Q2_K is mostly green because it stops
writing, and the IQ2 and 2-bit builds are mostly orange because they repeat words. The E3 figure,
[`qwen3.5-4b_H3_E3.png`](figures/qwen3.5-4b_H3_E3.png), has the same layout.

E4, the first word prompt, failed as a test. Qwen 3.5 4B took "write each digit as a word" to mean
"say this number" and answered "twenty trillion two hundred…", even at BF16. Its records are
kept. E4b adds one example ("507 becomes: five zero seven") and replaces it.

## H4: format, imatrix and recipe

I built each format several ways. Noise on π tracks damage to what the model memorized, and noise
on random digits tracks generic damage.

![For each format, π recall and noise on π for every build](figures/qwen3.5-4b_H4_builds.png)

*How to read it:* one row per format, most bits at the top. Each marker is one build of that
format, shaped as in the table above, and a grey bar spans the best and worst build. Left: digits
of π, further right is better, dashed line at BF16's 1,000. Right: noise on π on a log scale,
further left is better, with unsloth's larger UD-XL files in the rows marked (XL).

| Build of Q4_K_M | π digits | Noise on π | Noise on random |
|---|---|---|---|
| plain | 123 | 0.039 | 0.0033 |
| with my number-free imatrix | 123 | 0.049 | 0.0016 |
| with unsloth's imatrix | 220 | 0.041 | 0.0024 |
| unsloth's file | 326 | 0.022 | 0.0019 |
| unsloth UD-Q4_K_XL (6% larger) | 326 | 0.014 | 0.0013 |

| Build of Q3_K_M | π digits | Noise on π | Noise on random |
|---|---|---|---|
| plain | 88 | 0.35 | 0.0059 |
| with my imatrix | 88 | 0.33 | 0.0045 |
| with unsloth's imatrix | 88 | 0.29 | 0.0028 |
| unsloth's file | 87 | 0.17 | 0.0027 |
| unsloth UD-Q3_K_XL (8% larger) | 168 | 0.10 | 0.0025 |

- **Format.** Confirmed. At the same size, legacy Q4_0 has 3.3 times the π noise of Q4_K_M (0.131
  against 0.039).
- **Imatrix.** My imatrix comes from text without a single digit string. At Q4_K_M it halves the
  generic noise and leaves π unprotected, with the noise on π rising from 0.039 to 0.049. An
  imatrix protects what its sample text uses, and the plan kept digits out on purpose to test
  exactly that. At 2–3 bits it matters for everything. It separates a Q2_K that can't copy (59
  errors per 100 digits) from one that can (0.06).
- **Recipe.** My Q4_K_M with unsloth's imatrix gains little noise-wise. Its π noise is 0.041
  against 0.022 for their file, and the rest of that drop comes from how unsloth sets the
  precision of each part of the model. Their UD-XL files go further, at a cost in size and speed (see below). The
  [follow-up report](REPORT_FOLLOWUP.md) takes their recipe apart, and finds that recall needs
  the recipe and their imatrix together.

## H6: compression weakens memory's pull

E5 asks for an exact copy of π or e with one digit changed, and of the 8 planted digits only
position 10 of π tells the versions apart. BF16 and almost every build at 4.6 bits per weight or more
"correct" it back to the real digit of π. Almost every build at 4.4 or less copies the prompt
faithfully. So compression weakens the pull of memory, which fits H1 and runs opposite to what H6
predicted.

![For each version, what it wrote at the 8 planted digits](figures/qwen3.5-4b_H6_planted.png)

*How to read it:* one bar per version, 8 planted digits each. Blue counts faithful copies of the
prompt. Orange counts the times the version wrote the true digit from memory instead, and dark
grey the times it did neither because the copy itself failed. The orange slice on BF16 and the
higher builds is π position 10.

A few builds break the pattern. My Q4_K_M_imat and unsloth's Q4_0 copy faithfully at about 5 bits
per weight, while my IQ3_XXS reverts to memory at position 10 and unsloth's UD-IQ3_XXS at
positions 10 and 50. Plain Q2_K can't copy at all.

## Resources and speed

Every run used one Arc Pro B60 (24 GB) and llama.cpp with SYCL. Context was 8,192 tokens, flash
attention off. Decode speed is the middle value the server reported over the real test
requests, within 0–5% of llama.cpp's own benchmark. Prompt speed comes from the benchmark at 4,096
tokens, because short prompts carry a fixed overhead on this backend that hides the real
throughput.

| Version | File GB | VRAM GiB | Decode tok/s | Prompt tok/s (4k) | Power W |
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

- **Graphics memory is the file size plus about 770 MiB** for every version. The extra holds the
  conversation memory (the KV cache for 8,192 tokens) and working buffers, whatever the weights,
  so a 4 GB card holds every build up to Q5_K_M. BF16 needs about 9 GB.
- **Below about 5 bits, smaller files stop getting faster.** Decode goes 38 → 65 → 89 tokens/s
  from BF16 to Q8_0 to Q4_K_M and then stays flat down to Q2_K. At that point the card has
  stopped waiting on weight reads, and a smaller file only saves memory.
- **IQ formats are slow here.** Unpacking their lookup tables costs more time than the smaller
  file saves, so IQ3_XXS (35–38 tokens/s) decodes slower than BF16. UD-XL files decode 12–28%
  slower than plain K-quants of a similar size.
- **Host memory** follows the file size, because the file is mapped into memory instead of copied.
  It doesn't come on top of the graphics memory.

![π recall and noise on π against graphics memory used](figures/qwen3.5-4b_vram_quality.png)

*How to read it:* the same quality measures as above, plotted against the graphics memory each
version needs (weights, the 8k-token cache and buffers). Left, digits of π: higher is better.
Right, noise on π on a log scale: lower is better. The black star is BF16. Use this view to pick a
file that fits a given card.

![Decode and prompt speed against bits per weight](figures/qwen3.5-4b_speed.png)

*How to read it:* decode speed during the tests on the left, prompt speed at 4,096 tokens on the
right, both against bits per weight. Higher is faster. Plain-ladder points carry their names and
the black star is BF16. The green diamonds and orange squares well under the line on the left are
the IQ and UD files.

## Problems found and how they were handled

| Problem | Effect | What I did |
|---|---|---|
| The E4 prompt was read as "say this number" | E4 tested following instructions, not copying | Kept the records. Added E4b with an explicit example (approved) and reran it on all 28 versions |
| E6 (recall as words) gives nothing usable | BF16 counts "zero one two three…" instead of reciting | Reported as something the 4B can't do |
| "e to 1000 decimal places: 2." | BF16 answers with 1,200 zeros | Real model behaviour. Results reported per constant, π carries H1 |
| E5 copies leave out the decimal point | Early runs scored a false error | Fixed the extraction, with a test, before any real run |
| The server cached prompts in host memory | Inflated host memory, and possible reuse between requests | Turned off (`--cache-ram 0`) and reran the whole study |

The setup check (E0) passed on BF16 and Q8_0. Output matched byte for byte across repeated
requests, after a server restart and on both graphics cards.

## Next

Phase 2 (Qwen 3.5 9B) is in [REPORT_9B.md](REPORT_9B.md), Phase 3 (Qwen 3.8 27B) in
[REPORT_27B.md](REPORT_27B.md) and the follow-ups in [REPORT_FOLLOWUP.md](REPORT_FOLLOWUP.md).
Questions still open from Phase 1:

- Keep E6 for larger models, or drop it?
- Would "ordinary text plus some numbers" make the H4 comparison fairer, given that a number-free
  sample text leaves recall unprotected?
