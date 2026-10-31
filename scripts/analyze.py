"""Headline table and one figure per hypothesis, from scores/ and tf/.

usage: analyze.py [--model M]   writes scores/<model>_headline.{csv,md} and figures/<model>_*.png
"""

import argparse
import csv
import json
import statistics as st
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from common import FIGURES, GEN_EXPS, RUNS, SCORES, TF, load_config, read_jsonl

# reference palette (dataviz skill): categorical slots 1-3 in fixed order, gray for the reference
GROUP_STYLE = {
    "ladder":    {"color": "#2a78d6", "marker": "o", "label": "DIY, no imatrix"},
    "imatrix":   {"color": "#eb6834", "marker": "s", "label": "DIY + own imatrix"},
    "unsloth":   {"color": "#1baf7a", "marker": "D", "label": "unsloth"},
    "usimat":    {"color": "#52514e", "marker": "^", "label": "DIY + unsloth imatrix"},
    "scheme":    {"color": "#52514e", "marker": "v", "label": "Q4_0 (legacy)"},
    "reference": {"color": "#0b0b0b", "marker": "*", "label": "reference"},
}

# H4: every build of the same quant type, side by side. Rows run from most to fewest bits.
FAMILIES = [
    ("Q8_0",          ["Q8_0", "us_Q8_0"]),
    ("Q6_K",          ["Q6_K", "us_Q6_K"]),
    ("Q5_K_M",        ["Q5_K_M", "us_Q5_K_M"]),
    ("Q4_K (XL)",     ["us_UD-Q4_K_XL"]),
    ("Q4_K_M",        ["Q4_K_M", "Q4_K_M_imat", "Q4_K_M_usimat", "us_Q4_K_M"]),
    ("Q4_0",          ["Q4_0", "us_Q4_0"]),
    ("IQ4_XS",        ["IQ4_XS", "us_IQ4_XS"]),
    ("Q3_K (XL)",     ["us_UD-Q3_K_XL"]),
    ("Q3_K_M",        ["Q3_K_M", "Q3_K_M_imat", "Q3_K_M_usimat", "us_Q3_K_M"]),
    ("IQ3_XXS",       ["IQ3_XXS", "us_UD-IQ3_XXS"]),
    ("Q2_K",          ["Q2_K", "Q2_K_imat", "us_UD-Q2_K_XL"]),
    ("IQ2",           ["IQ2_XS", "us_UD-IQ2_XXS"]),
]
BUILD_STYLE = {   # same color/marker per build kind as GROUP_STYLE
    "plain":   GROUP_STYLE["ladder"] | {"label": "DIY, no imatrix"},
    "own":     GROUP_STYLE["imatrix"] | {"label": "DIY + own imatrix"},
    "usimat":  GROUP_STYLE["usimat"] | {"label": "DIY + unsloth imatrix"},
    "unsloth": GROUP_STYLE["unsloth"] | {"label": "unsloth file"},
}
RECIPE_STYLE = {"color": "#eda100", "marker": "P", "label": "DIY, per-tensor overrides"}   # slot 4


def build_kind(variant, group):
    if variant.startswith("us_"):
        return "unsloth"
    if group == "usimat":
        return "usimat"
    if group == "imatrix":
        return "own"
    return "plain"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "figure.dpi": 130,
    "legend.frameon": False,
})


def load_meta(model):
    return {p.parent.name: json.loads(p.read_text()) for p in (RUNS / model).glob("*/meta.json")}


def load_resources(model, variant):
    p = RUNS / model / variant / "resources.json"
    return json.loads(p.read_text()) if p.exists() else {}


def tg_in_test(model, variant):
    """Median decode speed the server reported during the actual test requests (>= 100 generated tokens).
    Prompt-processing speed from these requests is not used: prompts are short, so SYCL's fixed
    per-request overhead dominates it."""
    vals = [r["timings"]["predicted_per_second"] for e in GEN_EXPS
            for r in read_jsonl(RUNS / model / variant / f"{e}.jsonl")
            if r.get("timings") and r["timings"].get("predicted_n", 0) >= 100]
    return st.median(vals) if vals else None


def load_bench(model, variant):
    """llama-bench results keyed "pp512", "pp4096", "tg128" -> tokens/s."""
    p = RUNS / model / variant / "bench.json"
    if not p.exists():
        return {}
    out = {}
    for r in json.loads(p.read_text()):
        key = f"pp{r['n_prompt']}" if r["n_prompt"] else f"tg{r['n_gen']}"
        out[key] = r["avg_ts"]
    return out


def load_tf(model, variant, text):
    p = TF / model / variant / f"{text}.parquet"
    return pq.read_table(p).to_pandas() if p.exists() else None


def by_variant(rows, exp):
    out = defaultdict(list)
    for r in rows:
        if r["exp"] == exp:
            out[r["variant"]].append(r)
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def headline(model, cfg, rows, meta):
    reference = cfg["models"][model]["reference"]
    groups = {e: by_variant(rows, e) for e in GEN_EXPS}
    table = []
    for v, m in sorted(meta.items(), key=lambda kv: -kv[1]["bpw"]):
        e1 = groups["E1"].get(v, [])
        e5 = groups["E5"].get(v, [])
        e4b = groups["E4b"].get(v, [])

        def e1_const(c):
            return [r["prefix"] for r in e1 if r["meta"]["constant"] == c]
        tf = {t: load_tf(model, v, t) for t in ("pi", "e", "sqrt2", "random")}
        is_ref = v == reference
        row = {
            "variant": v, "group": m["group"], "bpw": m["bpw"],
            # E1: median over the 3 phrasings per constant; pi has the long runway
            "E1_pi": st.median(e1_const("pi")) if e1 else None,
            "E1_pi_min": min(e1_const("pi"), default=None), "E1_pi_max": max(e1_const("pi"), default=None),
            "E1_e": st.median(e1_const("e")) if e1 else None,
            "E1_sqrt2": st.median(e1_const("sqrt2")) if e1 else None,
            "E1_median_prefix": st.median(r["prefix"] for r in e1) if e1 else None,
            "E1_min": min((r["prefix"] for r in e1), default=None),
            "E1_max": max((r["prefix"] for r in e1), default=None),
            "E1_agree_ref": st.median(r["agree_ref_prefix"] for r in e1) if e1 and "agree_ref_prefix" in e1[0] else None,
            # E2: KL on pi (memorized) and on random digits (no memory); flips on pi only, since on
            # random digits the reference is near-uniform and top-1 flips are ties, not damage
            "E2_kl_pi": None if is_ref or tf["pi"] is None else tf["pi"]["kl_vs_ref"].mean(),
            "E2_kl_random": None if is_ref or tf["random"] is None else tf["random"]["kl_vs_ref"].mean(),
            "E2_mean_kl": None if is_ref else mean([tf[t]["kl_vs_ref"].mean() for t in tf if tf[t] is not None]),
            "E2_flip_pi": None if is_ref or tf["pi"] is None else 1 - tf["pi"]["top1_matches_ref"].mean(),
            "E2_pi_wrong": None if tf["pi"] is None else int((tf["pi"]["rank"] > 0).sum()),
            "E3_err_per_100": mean(r["errors_per_100"] for r in groups["E3"].get(v, [])),
            "E3_exact_share": mean(r["exact"] for r in groups["E3"].get(v, [])),
            "E3_len1000_err": mean(r["errors_per_100"] for r in groups["E3"].get(v, []) if r["meta"]["length"] == 1000),
            "E4b_err_per_100": mean(r["errors_per_100"] for r in e4b),
            "E4b_random_err": mean(r["errors_per_100"] for r in e4b if not r["meta"]["run_heavy"]),
            "E4b_runheavy_err": mean(r["errors_per_100"] for r in e4b if r["meta"]["run_heavy"]),
            "E4b_parse_errors": sum(r["parse_errors"] for r in e4b) if e4b else None,
            "E4_v1_parse_errors": sum(r["parse_errors"] for r in groups["E4"].get(v, [])) if groups["E4"].get(v) else None,
            "E5_planted_share": mean(r["outcome"] == "planted" for r in e5),
            "E5_memory_share": mean(r["outcome"] == "memory" for r in e5),
            "E6_median_prefix": st.median(r["prefix"] for r in groups["E6"][v]) if groups["E6"].get(v) else None,
            "truncated": sum(r["truncated"] for r in rows if r["variant"] == v and r["exp"] not in ("E1", "E4", "E6")),
        }
        res = load_resources(model, v)
        b = load_bench(model, v)
        vram = res.get("vram_steady_delta_mib", res.get("vram_delta_mib"))
        row.update({
            "file_gb": round(m["file_size"] / 1e9, 2),
            "vram_mib": round(vram) if vram is not None else None,
            "vram_peak_mib": round(res["vram_delta_mib"]) if res.get("vram_delta_mib") is not None else None,
            # what is not weights: KV cache (8,192 tokens, f16), recurrent state, compute buffers, runtime,
            # minus the input token-embedding table, which llama.cpp keeps in host memory
            "vram_overhead_mib": round(vram - m["file_size"] / 2**20) if vram is not None else None,
            "host_rss_mib": round(res["host_peak_rss_mib"]) if res.get("host_peak_rss_mib") is not None else None,
            "tg_in_test_tok_s": tg_in_test(model, v),
            "tg128_tok_s": b.get("tg128"), "pp512_tok_s": b.get("pp512"), "pp4096_tok_s": b.get("pp4096"),
            "power_w": res.get("power_mean_w"),
            # the 27B's largest files are split over both GPUs: their speed is not one-card speed
            "n_gpus": len(res["device"].split(":")[1].split(",")) if res.get("device") else None,
        })
        table.append(row)

    SCORES.mkdir(exist_ok=True)
    with open(SCORES / f"{model}_headline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    def fmt(x):
        if x is None:
            return "–"
        if isinstance(x, float):
            return f"{x:.3g}" if abs(x) < 10 else f"{x:.0f}"
        return str(x)
    cols = list(table[0])
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in table]
    (SCORES / f"{model}_headline.md").write_text("\n".join(md) + "\n")
    return table


def ref_label(variant, reference):
    return f"{variant} (reference)" if variant == reference else variant


def style_ref_ticks(ax, labels, reference, axis="y"):
    """Bold the reference row's tick label and draw a rule under it."""
    ticks = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    for t in ticks:
        if t.get_text().startswith(f"{reference} (reference)"):
            t.set_fontweight("bold")
            t.set_color(INK)


def annotate_ref(ax, table, key, xkey="bpw"):
    """Label the reference point in a scatter so it can't be mistaken for a quant."""
    ref = next((r for r in table if r["group"] == "reference"), None)
    if ref and ref.get(key) is not None and ref.get(xkey) is not None:
        ax.annotate(f"{ref['variant']} (reference)", (ref[xkey], ref[key]), textcoords="offset points",
                    xytext=(-8, 8), ha="right", fontsize=8, fontweight="bold", color=INK)


def ref_type(table):
    """Quant type of the reference, for labels: BF16 for the 4B/9B, Q8_0 for the 27B."""
    ref = next((r for r in table if r["group"] == "reference"), None)
    return ref["variant"].removeprefix("us_") if ref else "reference"


def tint(color, amount=0.88):
    """Mix a hex color with white: the fill behind a label of that series."""
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return tuple(rgb + (1 - rgb) * amount)


def place_labels(ax, points, fontsize=6.5, avoid=()):
    """Boxed labels with leader lines, placed greedily where they overlap nothing.

    points: (x, y, text, color) in data coordinates. Each label is tried at growing distances in
    eight directions around its point and goes to the first spot inside the axes that clears the
    markers, the labels already placed and any `avoid` boxes (display coordinates, e.g. a legend).
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_px = ax.transData.transform
    frame = ax.get_window_extent(renderer)
    pt_px = fig.dpi / 72
    marks = [to_px((x, y)) for x, y, _, _ in points]
    taken = [matplotlib.transforms.Bbox.from_extents(mx - 6, my - 6, mx + 6, my + 6) for mx, my in marks]
    taken += list(avoid)
    dirs = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1, 1), (1, 1), (-1, -1), (1, -1)]
    # crowded points first, so they get the spots next to them
    crowd = [sum(np.hypot(*(np.subtract(m, o))) < 60 for o in marks) for m in marks]
    for i in sorted(range(len(points)), key=lambda i: -crowd[i]):
        x, y, text, color = points[i]
        mx, my = marks[i]
        t = ax.annotate(text, (x, y), textcoords="offset points", xytext=(0, 0), fontsize=fontsize,
                        color=INK, ha="center", va="center", zorder=6,
                        bbox=dict(boxstyle="square,pad=0.25", fc=tint(color), ec=color, lw=0.7),
                        arrowprops=dict(arrowstyle="-", color=color, lw=0.7, shrinkA=0, shrinkB=4))
        w, h = (lambda b: (b.width, b.height))(t.get_window_extent(renderer))
        w, h = w + 4 * pt_px, h + 4 * pt_px                       # the box padding
        best, best_cost = None, None
        for r in (16, 26, 38, 52, 68, 86, 108, 134):
            for dx, dy in dirs:
                cx = mx + dx * (r * pt_px + w / 2 * abs(dx))
                cy = my + dy * (r * pt_px * 0.6 + h / 2 * abs(dy))
                box = matplotlib.transforms.Bbox.from_extents(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                inside = (box.x0 >= frame.x0 + 2 and box.x1 <= frame.x1 - 2 and
                          box.y0 >= frame.y0 + 2 and box.y1 <= frame.y1 - 2)
                if not inside:
                    continue
                cost = sum(max(0, min(box.x1, o.x1) - max(box.x0, o.x0)) *
                           max(0, min(box.y1, o.y1) - max(box.y0, o.y0)) for j, o in enumerate(taken) if j != i)
                # a leader line that cuts through another label or marker counts as a small overlap
                seg = np.linspace((mx, my), (cx, cy), 12)[2:-1]
                cost += 40 * sum(any(o.x0 <= sx <= o.x1 and o.y0 <= sy <= o.y1 for sx, sy in seg)
                                 for j, o in enumerate(taken) if j != i)
                cost += 1.5 * r * pt_px                                  # and a long leader costs too
                if best_cost is None or cost < best_cost:
                    best, best_cost = (cx, cy, box), cost
        if best is None:
            t.remove()
            continue
        cx, cy, box = best
        t.xyann = ((cx - mx) / pt_px, (cy - my) / pt_px)
        taken.append(box)


def short_name(variant):
    """Label text: unsloth files carry their marker and colour, so the us_ prefix can go."""
    return variant.removeprefix("us_")


def scatter_by_group(ax, table, key, err=None, xkey="bpw"):
    """Markers per group; only the DIY ladder is joined by a line (one recipe, varying bits).
    Other groups mix types (K, IQ, UD) so a line between them would imply a trend that isn't there."""
    for g, style in GROUP_STYLE.items():
        pts = sorted([r for r in table if r["group"] == g and r[key] is not None and r.get(xkey) is not None],
                     key=lambda r: r[xkey])
        if not pts:
            continue
        x = [r[xkey] for r in pts]
        y = [r[key] for r in pts]
        kw = dict(color=style["color"], marker=style["marker"], markersize=13 if g == "reference" else 7,
                  label=f"reference ({ref_type(table)})" if g == "reference" else style["label"],
                  markeredgecolor="white", markeredgewidth=1)
        if err:
            lo = [r[key] - r[err[0]] for r in pts]
            hi = [r[err[1]] - r[key] for r in pts]
            ax.errorbar(x, y, yerr=[lo, hi], linestyle="-" if g == "ladder" else "none",
                        capsize=3, elinewidth=1, alpha=0.9, **kw)
        else:
            ax.plot(x, y, linestyle="-" if g == "ladder" else "none", **kw)


def label_groups(ax, table, key, groups, xkey="bpw"):
    """Boxed labels for the points of the given groups. Call after tight_layout, so they stay put."""
    pts = [(r[xkey], r[key], short_name(r["variant"]), GROUP_STYLE[r["group"]]["color"])
           for r in table if r["group"] in groups and r[key] is not None and r.get(xkey) is not None]
    legend = ax.get_legend()
    place_labels(ax, pts, avoid=[legend.get_window_extent()] if legend else ())


def fig_h1(model, table):
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ref = next((r for r in table if r["group"] == "reference"), None)
    scatter_by_group(ax, [r for r in table if r["group"] != "reference"], "E1_pi",
                     err=("E1_pi_min", "E1_pi_max"))
    if ref and ref["E1_pi"] is not None:
        ax.axhline(ref["E1_pi"], color=INK, linestyle="--", linewidth=1.2)
        ax.annotate(f"{ref_type(table)} reference ({ref['bpw']:.1f} bpw): {ref['E1_pi']:.0f} digits",
                    (ax.get_xlim()[0], ref["E1_pi"]), textcoords="offset points", xytext=(4, -13),
                    fontsize=9, fontweight="bold", color=INK)
    ax.set_xlabel("bits per weight (file size × 8 / parameters)")
    ax.set_ylabel("E1 correct digits of π\n(median of 3 phrasings, whiskers min–max)")
    ax.set_title(f"{model}: free recall of π vs bits per weight", loc="left", color=INK)
    ax.legend(loc="lower right")
    top = ax.get_ylim()[1]
    ax.set_ylim(-0.1 * top, top)          # room under zero for the labels of the most compressed files
    fig.tight_layout()
    label_groups(ax, table, "E1_pi", [g for g in GROUP_STYLE if g != "reference"])
    fig.savefig(FIGURES / f"{model}_H1_recall.png")
    plt.close(fig)


def fig_h2(model, cfg, table):
    reference = cfg["models"][model]["reference"]
    ref = load_tf(model, reference, "pi")
    if ref is None:
        return
    variants = [reference] + [r["variant"] for r in table if r["variant"] != reference]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 3 + 0.22 * len(variants)), sharex=True,
                                 gridspec_kw={"height_ratios": [2, max(2, 0.22 * len(variants))]})
    # shown as odds: a margin of m nats means the right digit is e^m times as likely as the runner-up
    m = ref["margin"].to_numpy()
    a1.plot(ref["pos"], np.exp(m), color=GRID, linewidth=0.8)
    k = 25
    # divide by the window actually covered, so the ends are not dragged toward zero by padding
    smooth = np.convolve(m, np.ones(k), mode="same") / np.convolve(np.ones_like(m), np.ones(k), mode="same")
    a1.plot(ref["pos"], np.exp(smooth), color=GROUP_STYLE["ladder"]["color"],
            label=f"{reference} (reference), {k}-digit moving average")
    a1.axhline(1, color=MUTED, linewidth=0.8)
    a1.set_yscale("log")
    a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}×"))
    a1.set_ylabel("right digit vs runner-up,\ntimes more likely (log)")
    a1.set_title(f"{model}: how sure the reference is of each π digit, and who gets it wrong",
                 loc="left", color=INK)
    a1.legend(loc="upper right")
    for i, v in enumerate(variants):
        df = load_tf(model, v, "pi")
        if df is None:
            continue
        wrong = df["pos"][df["rank"] > 0]
        a2.scatter(wrong, [i] * len(wrong), s=6, color=GROUP_STYLE["ladder"]["color"])
    a2.set_yticks(range(len(variants)), [ref_label(v, reference) for v in variants], fontsize=7)
    style_ref_ticks(a2, variants, reference)
    a2.axhline(0.5, color=MUTED, linewidth=0.8)
    a2.invert_yaxis()
    a2.set_xlabel("digit position")
    a2.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{model}_H2_margins.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    scatter_by_group(ax, table, "E2_kl_pi")
    ax.set_yscale("log")
    ax.annotate(f"{reference} (reference) is the baseline: noise = 0 by definition", xy=(0.01, 0.02),
                xycoords="axes fraction", ha="left", va="bottom", fontsize=8, fontweight="bold", color=INK)
    ax.set_xlabel("bits per weight")
    ax.set_ylabel(f"noise on the π text\n(KL divergence from {reference}, log)")
    ax.set_title(f"{model}: noise vs bits per weight", loc="left", color=INK)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / f"{model}_H2_kl.png")
    plt.close(fig)


def fig_tradeoffs(model, table):
    """Quality against the VRAM a variant actually needs: the "which card runs it" view."""
    for r in table:
        r["vram_gib"] = r["vram_mib"] / 1024 if r.get("vram_mib") is not None else None
    panels = [("E1_pi", "E1 correct digits of π (median of 3)", False),
              ("E2_kl_pi", f"noise on π (KL from {ref_type(table)}, log)", True)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (key, label, log) in zip(axes, panels):
        scatter_by_group(ax, table, key, xkey="vram_gib")
        annotate_ref(ax, table, key, xkey="vram_gib")
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("VRAM used, GiB (weights + 8k-token f16 KV cache + buffers)")
        ax.set_ylabel(label)
    ref = next((r for r in table if r["group"] == "reference"), None)
    if ref and ref.get("vram_gib"):
        axes[1].annotate(f"{ref['variant']} (reference): noise = 0 by definition, {ref['vram_gib']:.1f} GiB",
                         xy=(0.99, 0.98), xycoords="axes fraction", ha="right", va="top", fontsize=8,
                         fontweight="bold", color=INK)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"{model}: quality vs VRAM needed", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{model}_vram_quality.png")
    plt.close(fig)


def fig_speed(model, table):
    """llama-bench: decode is memory-bound (smaller is faster), prompt processing is compute-bound."""
    panels = [("tg_in_test_tok_s", "decode during the tests, tokens/s\n(median over requests, server timings)"),
              ("pp4096_tok_s", "prompt processing, tokens/s\n(llama-bench pp4096)")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, (key, label) in zip(axes, panels):
        scatter_by_group(ax, table, key)
        annotate_ref(ax, table, key)
        for r in table:
            if (r.get("n_gpus") or 1) > 1 and r.get(key) is not None:
                ax.annotate("2 GPUs", (r["bpw"], r[key]), textcoords="offset points", xytext=(4, -11),
                            fontsize=7, color=MUTED)
        ax.set_xlabel("bits per weight")
        ax.set_ylabel(label)
        vals = [r[key] for r in table if r.get(key) is not None]
        if vals:
            ax.set_ylim(0, 1.15 * max(vals))
    axes[0].legend(loc="lower right", fontsize=8)
    split = any((r.get("n_gpus") or 1) > 1 for r in table)
    fig.suptitle(f"{model}: speed on one Arc Pro B60 (SYCL)"
                 + ('; points marked "2 GPUs" are split over both cards' if split else ""),
                 x=0.01, ha="left", color=INK)
    fig.tight_layout()
    for ax, (key, _) in zip(axes, panels):
        label_groups(ax, table, key, ("ladder", "scheme"))
    fig.savefig(FIGURES / f"{model}_speed.png")
    plt.close(fig)


def fig_h4(model, table):
    """One row per quant type, one marker per build: plain, own imatrix, unsloth imatrix, unsloth file."""
    by_name = {r["variant"]: r for r in table}
    fams = [(f, [v for v in vs if v in by_name]) for f, vs in FAMILIES]
    fams = [(f, vs) for f, vs in fams if vs]
    panels = [("E1_pi", "E1 correct digits of π (median of 3)", False),
              ("E2_kl_pi", f"noise on π (KL from {ref_type(table)}, log)", True)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.9 + 0.42 * len(fams)), sharey=True)
    seen = set()
    for ax, (key, label, log) in zip(axes, panels):
        for i, (fam, vs) in enumerate(fams):
            xs = [by_name[v][key] for v in vs if by_name[v][key] is not None]
            if len(xs) > 1:
                ax.plot([min(xs), max(xs)], [i, i], color=GRID, linewidth=3, zorder=1)
            for v in vs:
                r = by_name[v]
                if r[key] is None:
                    continue
                kind = build_kind(v, r["group"])
                st_ = BUILD_STYLE[kind]
                ax.scatter(r[key], i, color=st_["color"], marker=st_["marker"], s=60, zorder=3,
                           edgecolor="white", linewidth=1, label=st_["label"] if (kind not in seen and ax is axes[0]) else None)
                if ax is axes[0]:
                    seen.add(kind)
        ax.set_xlabel(label)
        if log:
            ax.set_xscale("log")
        ax.grid(axis="y", visible=False)
    ref = next((r for r in table if r["group"] == "reference"), None)
    if ref and ref["E1_pi"] is not None:
        axes[0].axvline(ref["E1_pi"], color=INK, linestyle="--", linewidth=1.2)
        axes[0].annotate(f"{ref_type(table)} reference: {ref['E1_pi']:.0f}", (ref["E1_pi"], 0), textcoords="offset points",
                         xytext=(-4, 0), ha="right", va="center", fontsize=8, fontweight="bold", color=INK)
    axes[1].annotate(f"{ref_type(table)} reference: noise = 0", xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8,
                     fontweight="bold", color=INK)
    axes[0].set_yticks(range(len(fams)), [f for f, _ in fams])
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"{model}: same quant type, different builds", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{model}_H4_builds.png")
    plt.close(fig)


def fig_errors(model, rows, table, exp, title):
    reference = next(r["variant"] for r in table if r["group"] == "reference")
    variants = [r["variant"] for r in table]
    data = by_variant(rows, exp)
    kinds = [("subs", "substitution", "#2a78d6"), ("ins", "insertion", "#eb6834"), ("dels", "deletion", "#1baf7a")]
    fig, ax = plt.subplots(figsize=(9, 0.8 + 0.26 * len(variants)))
    left = np.zeros(len(variants))
    for key, label, color in kinds:
        vals = np.array([sum(r[key] for r in data.get(v, [])) for v in variants], dtype=float)
        ax.barh(variants, vals, left=left, color=color, label=label, edgecolor="white", linewidth=1)
        left += vals
    ax.invert_yaxis()
    ax.set_yticks(range(len(variants)), [ref_label(v, reference) for v in variants])
    style_ref_ticks(ax, variants, reference)
    ax.axhline(0.5, color=MUTED, linewidth=0.8)
    ax.set_xlabel(f"total errors over all {exp} inputs")
    ax.set_title(f"{model}: {title}", loc="left", color=INK)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=7)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(FIGURES / f"{model}_H3_{exp}.png")
    plt.close(fig)


def fig_h6(model, rows, table):
    reference = next(r["variant"] for r in table if r["group"] == "reference")
    variants = [r["variant"] for r in table]
    data = by_variant(rows, "E5")
    kinds = [("planted", "copied prompt", "#2a78d6"), ("memory", "wrote memorized digit", "#eb6834"),
             ("other", "other", "#52514e")]
    fig, ax = plt.subplots(figsize=(8, 0.8 + 0.26 * len(variants)))
    left = np.zeros(len(variants))
    for key, label, color in kinds:
        vals = np.array([sum(r["outcome"] == key for r in data.get(v, [])) for v in variants], dtype=float)
        ax.barh(variants, vals, left=left, color=color, label=label, edgecolor="white", linewidth=1)
        left += vals
    ax.invert_yaxis()
    ax.set_yticks(range(len(variants)), [ref_label(v, reference) for v in variants])
    style_ref_ticks(ax, variants, reference)
    ax.axhline(0.5, color=MUTED, linewidth=0.8)
    ax.set_xlabel("E5 sequences (of 8)")
    ax.set_title(f"{model}: planted digit, did it copy the prompt or recite memory?", loc="left", color=INK)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=7)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(FIGURES / f"{model}_H6_planted.png")
    plt.close(fig)


MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # categorical slots 1-3, one per model size
SEQ_COLORS = ["#a9c8f2", "#7aa9e8", "#4d8ade", "#2a6fc4", "#1b5aa8", "#0f3d78"]   # one hue, light to dark


def seq_colors(k):
    """k steps of SEQ_COLORS, darkest last; interpolated when there are more than six."""
    if k <= len(SEQ_COLORS):
        return SEQ_COLORS[-k:] if k else []
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ_COLORS)
    return [matplotlib.colors.to_hex(cmap(t)) for t in np.linspace(0, 1, k)]


# Follow-up R: unsloth's recipe taken apart, one change per row, top to bottom
RECIPE_ROWS = [
    ("Q4_K_M", [
        ("Q4_K_M",                 "stock recipe, no imatrix"),
        ("Q4_K_M_imat",            "stock recipe, own imatrix"),
        ("Q4_K_M_usimat",          "stock recipe, unsloth imatrix"),
        ("Q4_K_M_usimat_ssmout",   "  + ssm_out at 5 bits"),
        ("Q4_K_M_usimat_ssmab",    "  + ssm_alpha, ssm_beta at 8 bits"),
        ("Q4_K_M_usimat_qkv",      "  + attn_qkv at 5 bits"),
        ("Q4_K_M_usrecipe",        "unsloth recipe, no imatrix"),
        ("Q4_K_M_usrecipe_imat",   "unsloth recipe, own imatrix"),
        ("Q4_K_M_usrecipe_usimat", "unsloth recipe, unsloth imatrix (our rebuild)"),
        ("us_Q4_K_M",              "unsloth's file"),
    ]),
    ("Q2_K", [
        ("Q2_K",                   "stock recipe, no imatrix"),
        ("Q2_K_ssm",               "  + ssm_out 5 bits, ssm_alpha/beta 8 bits"),
        ("Q2_K_imat",              "stock recipe, own imatrix"),
        ("Q2_K_imat_ssm",          "  + ssm_out 5 bits, ssm_alpha/beta 8 bits"),
        ("Q2_K_usrecipe_imat",     "UD-Q2_K_XL recipe, own imatrix"),
        ("Q2_K_usrecipe_usimat",   "UD-Q2_K_XL recipe, unsloth imatrix (our rebuild)"),
        ("us_UD-Q2_K_XL",          "unsloth's UD-Q2_K_XL file"),
    ]),
]


def fig_recipe(model, table):
    need = ("E1_pi", "E1_pi_min", "E1_pi_max", "E2_kl_pi", "file_gb")
    rows = {r["variant"]: r for r in table if all(r.get(k) is not None for k in need)}
    blocks = [(t, [(v, lab) for v, lab in items if v in rows]) for t, items in RECIPE_ROWS]
    blocks = [(t, items) for t, items in blocks if items]
    if not any(rows[v]["group"] == "recipe" for _, items in blocks for v, _ in items):
        return
    n = sum(len(items) for _, items in blocks)
    fig, axes = plt.subplots(len(blocks), 3, figsize=(13, 1.2 + 0.36 * n), sharex="col", squeeze=False,
                             gridspec_kw={"height_ratios": [len(items) for _, items in blocks],
                                          "width_ratios": [2.2, 1.6, 1.2]})
    for (title, items), (a1, a2, a3) in zip(blocks, axes):
        y = np.arange(len(items))[::-1]
        for yi, (v, _) in zip(y, items):
            r = rows[v]
            style = BUILD_STYLE[build_kind(v, r["group"])] if r["group"] != "recipe" else RECIPE_STYLE
            a1.errorbar([r["E1_pi"]], [yi], xerr=[[r["E1_pi"] - r["E1_pi_min"]], [r["E1_pi_max"] - r["E1_pi"]]],
                        color=style["color"], marker=style["marker"], markersize=7, capsize=3,
                        markeredgecolor="white", markeredgewidth=1, elinewidth=1)
            a1.annotate(f"{r['E1_pi']:.0f}", (r["E1_pi"], yi), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=7, color=MUTED)
            a2.plot([r["E2_kl_pi"]], [yi], color=style["color"], marker=style["marker"], markersize=7,
                    markeredgecolor="white", markeredgewidth=1, linestyle="none")
            a3.barh([yi], [r["file_gb"]], color=style["color"], height=0.6)
            a3.annotate(f"{r['file_gb']:.2f}", (r["file_gb"], yi),
                        textcoords="offset points", xytext=(3, -3), fontsize=7, color=MUTED)
        a1.set_yticks(y, [lab for _, lab in items], fontsize=8)
        for a in (a2, a3):
            a.set_yticks(y, [])
        a1.set_ylim(-0.6, len(items) - 0.4)
        a1.set_title(f"{title} family", loc="left", color=INK, fontsize=10)
        a1.set_xscale("symlog", linthresh=10)
        a2.set_xscale("log")
    axes[-1][0].set_xlabel("E1 correct digits of π (median of 3, whiskers worst to best wording)")
    axes[-1][1].set_xlabel("noise on π (KL from BF16, log)")
    axes[-1][2].set_xlabel("file size (GB)")
    handles = [plt.Line2D([], [], color=s["color"], marker=s["marker"], linestyle="none", label=s["label"])
               for s in (BUILD_STYLE["plain"], BUILD_STYLE["own"], BUILD_STYLE["usimat"], RECIPE_STYLE,
                         BUILD_STYLE["unsloth"])]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(f"{model}: unsloth's recipe taken apart, one change at a time", x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / f"{model}_recipe.png")
    plt.close(fig)


def load_headlines(models):
    tables = {}
    for m in models:
        p = SCORES / f"{m}_headline.csv"
        if p.exists():
            with open(p) as f:
                tables[m] = [{k: (v if k in ("variant", "group") else None if v in ("", "None") else float(v))
                              for k, v in r.items()} for r in csv.DictReader(f) if r["group"] != "recipe"]
    return tables


def fig_h5(models):
    """H5: the same quant types across model sizes, from each model's headline table."""
    tables = load_headlines(models)
    if len(tables) < 2:
        return
    panels = [("E1_pi", "E1 correct digits of π (median of 3)", False),
              ("E2_kl_pi", "noise on π (KL from the reference, log)", True),
              ("E2_kl_random", "noise on random digits (KL from the reference, log)", True)]
    fig = plt.figure(figsize=(12, 9.5))
    grid = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    xkey = "bpw"
    for ax, (key, label, log) in zip(axes, panels):
        for (m, table), color in zip(tables.items(), MODEL_COLORS):
            for group, ls, marker in (("ladder", "-", "o"), ("unsloth", "none", "D")):
                pts = sorted([r for r in table if r["group"] == group and isinstance(r.get(key), float)],
                             key=lambda r: r[xkey])
                ax.plot([r[xkey] for r in pts], [r[key] for r in pts], linestyle=ls, marker=marker,
                        color=color, markersize=6, markeredgecolor="white", markeredgewidth=1,
                        alpha=1 if group == "ladder" else 0.7,
                        label=f"{m} {'DIY ladder' if group == 'ladder' else 'unsloth files'}")
            ref = next((r for r in table if r["group"] == "reference"), None)
            if not log and ref and isinstance(ref.get(key), float):
                ax.axhline(ref[key], color=color, linestyle="--", linewidth=1,
                           label=f"{m} {ref['variant'].removeprefix('us_')} (reference)")
        if log:
            ax.set_yscale("log")
        else:
            ax.axhline(1200, color=MUTED, linestyle=":", linewidth=1, label="generation cap (1,200 digits)")
        ax.set_xlim(2.5, 9)
        ax.set_xlabel("bits per weight")
        ax.set_ylabel(label)
    axes[0].set_title(f"same quant types, {len(tables)} model sizes", loc="left", color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGURES / "H5_model_size.png")
    plt.close(fig)


def pick_rate(tables, key, max_gap=0.5):
    """Of two versions of one model within max_gap bpw that recall different lengths, how often
    the one with the smaller `key` (larger for bpw) is the one that recalls more."""
    hits = n = 0
    for table in tables.values():
        rows = [r for r in table if isinstance(r.get("E2_kl_pi"), float) and isinstance(r.get("E1_pi"), float)]
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                if abs(a["bpw"] - b["bpw"]) > max_gap or a["E1_pi"] == b["E1_pi"]:
                    continue
                sign = 1 if key == "bpw" else -1
                n += 1
                hits += sign * (a[key] - b[key]) * (a["E1_pi"] - b["E1_pi"]) > 0
    return hits, n


def fig_noise(models):
    """Can the noise score stand in for the slow recall test? Noise on π vs on unrelated digits."""
    tables = load_headlines(models)
    ceiling = {m: next((r["E1_pi"] for r in t if r["group"] == "reference"), None) for m, t in tables.items()}
    tables = {m: [r for r in t if isinstance(r.get("E1_pi"), float)] for m, t in tables.items()
              if ceiling[m] is not None}
    # versions that recall as far as their reference sit at the ceiling and say nothing about the slope
    pts = [(m, r) for m, t in tables.items() for r in t
           if isinstance(r.get("E2_kl_pi"), float) and r["E1_pi"] < ceiling[m]]
    if len(pts) < len(tables) + 2:
        return
    # one slope for all sizes, one offset per size: log10(digits + 1) = a * log10(noise) + b_model
    x = np.log10([r["E2_kl_pi"] for _, r in pts])
    y = np.log10([r["E1_pi"] + 1 for _, r in pts])
    onehot = np.array([[m == k for k in tables] for m, _ in pts], dtype=float)
    coef, *_ = np.linalg.lstsq(np.column_stack([x, onehot]), y, rcond=None)
    resid = y - np.column_stack([x, onehot]) @ coef
    stats = {"slope": coef[0], "digits_per_10x": 10 ** -coef[0], "typical_miss": 10 ** resid.std(),
             "r2": 1 - resid.var() / y.var()}
    for key in ("E2_kl_pi", "E2_kl_random", "bpw"):
        stats[f"pick_{key}"] = pick_rate(tables, key)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    panels = [("E2_kl_pi", "noise on π (KL from the reference, log)"),
              ("E2_kl_random", "noise on random digits (KL from the reference, log)")]
    for ax, (key, label) in zip(axes, panels):
        for (m, table), color in zip(tables.items(), MODEL_COLORS):
            for group in GROUP_STYLE:
                sel = [r for r in table if r["group"] == group and isinstance(r.get(key), float)]
                if sel:
                    ax.scatter([r[key] for r in sel], [max(r["E1_pi"], 1) for r in sel], s=40,
                               marker=GROUP_STYLE[group]["marker"], color=color, edgecolor="white",
                               linewidth=1, zorder=3)
            if key == "E2_kl_pi":
                xs = np.logspace(np.log10(min(r[key] for r in table if isinstance(r.get(key), float))),
                                 np.log10(max(r[key] for r in table if isinstance(r.get(key), float))), 50)
                b = coef[1 + list(tables).index(m)]
                ax.plot(xs, np.minimum(10 ** (coef[0] * np.log10(xs) + b) - 1, ceiling[m]),
                        color=color, linewidth=1.5, alpha=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(label)
    axes[0].set_ylabel("E1 correct digits of π (median of 3, log; 0 drawn at 1)")
    def share(key):
        h, n = stats[key]
        return f"{h / n:.0%}" if n else "n/a"
    axes[0].set_title(f"noise on π: picks the better of two similar files {share('pick_E2_kl_pi')} of the time",
                      loc="left", color=INK, fontsize=10)
    axes[1].set_title(f"noise on random digits: {share('pick_E2_kl_random')} (file size alone: {share('pick_bpw')})",
                      loc="left", color=INK, fontsize=10)
    handles = [plt.Line2D([], [], color=c, marker="o", linestyle="-", label=m)
               for m, c in zip(tables, MODEL_COLORS)]
    handles += [plt.Line2D([], [], color=MUTED, marker=GROUP_STYLE[g]["marker"], linestyle="none",
                           label=GROUP_STYLE[g]["label"]) for g in ("ladder", "imatrix", "unsloth", "usimat", "scheme")]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=8, frameon=False)
    fig.suptitle("does a quick noise test predict how far a version recites π?", x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGURES / "noise_predicts_recall.png")
    plt.close(fig)
    with open(SCORES / "noise_predicts_recall.json", "w") as f:
        json.dump(stats, f, indent=1)


def digit_odds_under(setting, top_lp, top_ids, correct, lp_correct):
    """Chance of sampling the right digit at one position, applying llama.cpp's order:
    top-k, then top-p on the renormalized survivors, then temperature."""
    if setting["top_k"] <= 0 and setting["top_p"] >= 1 and setting["temperature"] == 1:
        return float(np.exp(lp_correct))
    lp = top_lp[:setting["top_k"]] if setting["top_k"] > 0 else top_lp
    ids = top_ids[:len(lp)]
    p = np.exp(lp - lp.max())
    p /= p.sum()
    keep = int(np.searchsorted(np.cumsum(p), setting["top_p"]) + 1) if setting["top_p"] < 1 else len(p)
    lp, ids = lp[:keep], ids[:keep]
    q = np.exp((lp - lp.max()) / setting["temperature"])
    q /= q.sum()
    hit = np.where(ids == correct)[0]
    return float(q[hit[0]]) if len(hit) else 0.0


def sampling_rows(cfg, model):
    """Forecast vs observed for every sampled variant of a model: survival curves and summaries."""
    out = []
    s = cfg.get("sampling", {})
    for variant in s.get("variants", {}).get(model, []):
        top_p = TF / model / variant / "sampling" / "pi_top.parquet"
        samples = read_jsonl(RUNS / model / variant / "S.jsonl")
        tf = load_tf(model, variant, "pi")
        if not top_p.exists() or not samples or tf is None:
            continue
        top = pq.read_table(top_p).to_pandas()
        lp_correct = tf["logprob"].to_numpy()
        n_digits = len(top)
        wrong = np.where(tf["rank"].to_numpy() > 0)[0]
        greedy = int(wrong[0]) if len(wrong) else n_digits          # top choice first differs from π
        for name, setting in s["settings"].items():
            p = np.array([digit_odds_under(setting, np.asarray(r.top_logprobs), np.asarray(r.top_ids),
                                           r.correct_token, lp_correct[k]) for k, r in enumerate(top.itertuples())])
            survive = np.concatenate([[1.0], np.cumprod(p)])          # P(at least n digits right), falling
            got = np.array([r["prefix"] for r in samples if r["setting"] == name])
            observed = np.array([(got >= n).mean() for n in range(n_digits + 1)])
            out.append({"model": model, "variant": variant, "setting": name, "n": len(got),
                        "greedy": greedy, "forecast_mean": float(survive[1:].sum()),
                        "observed_mean": float(got.mean()),
                        "forecast_median": int((survive >= 0.5).sum()) - 1,   # last n with P(>= n) >= 1/2
                        "observed_median": float(np.median(got)),
                        "survive": survive, "observed": observed, "p": p})
    return out


def fig_sampling(cfg, models):
    rows = [r for m in models for r in sampling_rows(cfg, m)]
    if not rows:
        return
    models = [m for m in models if any(r["model"] == m for r in rows)]
    settings = list(cfg["sampling"]["settings"])
    fig, axes = plt.subplots(len(models), len(settings), figsize=(12, 4.2 * len(models)), squeeze=False)
    titles = {"t1": "plain sampling (temperature 1)", "chat": "chat settings (temp 0.7, top-k 20, top-p 0.8)"}
    for i, m in enumerate(models):
        for j, name in enumerate(settings):
            ax = axes[i][j]
            sel = [r for r in rows if r["model"] == m and r["setting"] == name]
            for r, color in zip(sel, seq_colors(len(sel))[::-1]):
                n = np.arange(len(r["survive"]))
                ax.plot(n, r["survive"], color=color, linewidth=1.5,
                        label=f"{r['variant']}: forecast {r['forecast_mean']:.0f}, measured {r['observed_mean']:.0f}")
                ax.step(n, r["observed"], where="post", color=color, linewidth=1, linestyle=":")
            ax.set_xscale("symlog", linthresh=10)
            ax.set_xlim(0, len(sel[0]["survive"]))
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{m}, {titles.get(name, name)}", loc="left", color=INK, fontsize=10)
            ax.set_xlabel("digits of π recited correctly (log past 10)")
            ax.set_ylabel("share of recitations that get this far")
            ax.legend(fontsize=7, loc="lower left", frameon=False, title="mean digits", title_fontsize=7)
    fig.suptitle("per-digit certainty forecasts sampled recitations (solid: forecast from one teacher-forced pass, "
                 f"dotted: measured over {rows[0]['n']} real samples)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / "sampling_forecast.png")
    plt.close(fig)
    keys = ["model", "variant", "setting", "n", "greedy", "forecast_mean", "observed_mean",
            "forecast_median", "observed_median"]
    with open(SCORES / "sampling.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 1) if isinstance(r[k], float) else r[k]) for k in keys})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    args = ap.parse_args()
    cfg = load_config()
    FIGURES.mkdir(exist_ok=True)
    for model in [args.model] if args.model else cfg["models"]:
        rows = read_jsonl(SCORES / f"{model}.jsonl")
        if not rows:
            continue
        meta = load_meta(model)
        full = headline(model, cfg, rows, meta)
        table = [r for r in full if r["group"] != "recipe"]   # follow-up R has its own figure
        fig_h1(model, table)
        fig_tradeoffs(model, table)
        fig_speed(model, table)
        fig_h4(model, table)
        fig_h2(model, cfg, table)
        fig_errors(model, rows, table, "E3", "verbatim copy errors by type")
        fig_errors(model, rows, table, "E4b", "spell-as-words errors by type (clarified prompt)")
        fig_h6(model, rows, table)
        fig_recipe(model, full)
        print(f"{model}: {len(table)} variants -> scores/{model}_headline.md, figures/{model}_*.png")
    fig_h5(list(cfg["models"]))
    fig_noise(list(cfg["models"]))
    fig_sampling(cfg, list(cfg["models"]))


if __name__ == "__main__":
    main()
