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


def scatter_by_group(ax, table, key, label_points=False, err=None, xkey="bpw"):
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
        if label_points and g in ("ladder", "scheme"):
            for r in pts:
                ax.annotate(r["variant"], (r[xkey], r[key]), textcoords="offset points", xytext=(4, 4),
                            fontsize=7, color=MUTED)


def fig_h1(model, table):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ref = next((r for r in table if r["group"] == "reference"), None)
    scatter_by_group(ax, [r for r in table if r["group"] != "reference"], "E1_pi",
                     label_points=True, err=("E1_pi_min", "E1_pi_max"))
    if ref and ref["E1_pi"] is not None:
        ax.axhline(ref["E1_pi"], color=INK, linestyle="--", linewidth=1.2)
        ax.annotate(f"{ref_type(table)} reference ({ref['bpw']:.1f} bpw): {ref['E1_pi']:.0f} digits",
                    (ax.get_xlim()[0], ref["E1_pi"]), textcoords="offset points", xytext=(4, -13),
                    fontsize=9, fontweight="bold", color=INK)
    ax.set_xlabel("bits per weight (file size × 8 / parameters)")
    ax.set_ylabel("E1 correct digits of π\n(median of 3 phrasings, whiskers min–max)")
    ax.set_title(f"{model}: free recall of π vs bits per weight", loc="left", color=INK)
    ax.legend(loc="lower right")
    fig.tight_layout()
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
    m = ref["margin"].to_numpy()
    a1.plot(ref["pos"], m, color=GRID, linewidth=0.8)
    k = 25
    # divide by the window actually covered, so the ends are not dragged toward zero by padding
    smooth = np.convolve(m, np.ones(k), mode="same") / np.convolve(np.ones_like(m), np.ones(k), mode="same")
    a1.plot(ref["pos"], smooth, color=GROUP_STYLE["ladder"]["color"],
            label=f"{reference} (reference) margin, {k}-digit moving average")
    a1.axhline(0, color=MUTED, linewidth=0.8)
    a1.set_ylabel("margin (nats)")
    a1.set_title(f"{model}: π margin under teacher forcing, and each variant's wrong digits",
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
    ax.annotate(f"{reference} (reference) is the baseline: KL = 0 by definition", xy=(0.01, 0.02),
                xycoords="axes fraction", ha="left", va="bottom", fontsize=8, fontweight="bold", color=INK)
    ax.set_xlabel("bits per weight")
    ax.set_ylabel(f"E2 mean KL divergence vs {reference}\non the π text (nats)")
    ax.set_title(f"{model}: logit noise vs bits per weight", loc="left", color=INK)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / f"{model}_H2_kl.png")
    plt.close(fig)


def fig_tradeoffs(model, table):
    """Quality against the VRAM a variant actually needs: the "which card runs it" view."""
    for r in table:
        r["vram_gib"] = r["vram_mib"] / 1024 if r.get("vram_mib") is not None else None
    panels = [("E1_pi", "E1 correct digits of π (median of 3)", False),
              ("E2_kl_pi", f"E2 mean KL vs {ref_type(table)} on π (nats, log)", True)]
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
        axes[1].annotate(f"{ref['variant']} (reference): KL = 0 by definition, {ref['vram_gib']:.1f} GiB",
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
        scatter_by_group(ax, table, key, label_points=True)
        annotate_ref(ax, table, key)
        for r in table:
            if (r.get("n_gpus") or 1) > 1 and r.get(key) is not None:
                ax.annotate("2 GPUs", (r["bpw"], r[key]), textcoords="offset points", xytext=(4, -11),
                            fontsize=7, color=MUTED)
        ax.set_xlabel("bits per weight")
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
    axes[0].legend(loc="upper right", fontsize=8)
    split = any((r.get("n_gpus") or 1) > 1 for r in table)
    fig.suptitle(f"{model}: speed on one Arc Pro B60 (SYCL)"
                 + ("; points marked “2 GPUs” are split over both cards" if split else ""),
                 x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{model}_speed.png")
    plt.close(fig)


def fig_h4(model, table):
    """One row per quant type, one marker per build: plain, own imatrix, unsloth imatrix, unsloth file."""
    by_name = {r["variant"]: r for r in table}
    fams = [(f, [v for v in vs if v in by_name]) for f, vs in FAMILIES]
    fams = [(f, vs) for f, vs in fams if vs]
    panels = [("E1_pi", "E1 correct digits of π (median of 3)", False),
              ("E2_kl_pi", f"E2 mean KL vs {ref_type(table)} on π (nats, log)", True)]
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
    axes[1].annotate(f"{ref_type(table)} reference: KL = 0", xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8,
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
    ax.set_title(f"{model}: planted error — prompt vs memory", loc="left", color=INK)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=7)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(FIGURES / f"{model}_H6_planted.png")
    plt.close(fig)


MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # categorical slots 1-3, one per model size


def fig_h5(models):
    """H5: the same quant types across model sizes, from each model's headline table."""
    tables = {}
    for m in models:
        p = SCORES / f"{m}_headline.csv"
        if p.exists():
            with open(p) as f:
                tables[m] = [{k: (float(v) if v not in ("", "None") and k not in ("variant", "group") else v)
                              for k, v in r.items()} for r in csv.DictReader(f)]
    if len(tables) < 2:
        return
    panels = [("E1_pi", "E1 correct digits of π (median of 3)", False, "bpw"),
              ("E1_pi", "E1 correct digits of π (median of 3)", False, "file_gb"),
              ("E2_kl_pi", "E2 mean KL vs reference on π (nats)", True, "bpw"),
              ("E2_kl_random", "E2 mean KL vs reference on random digits (nats)", True, "bpw")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    axes = axes.ravel()
    for ax, (key, label, log, xkey) in zip(axes, panels):
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
        if xkey == "bpw":
            ax.set_xlim(2.5, 9)
            ax.set_xlabel("bits per weight")
        else:
            ax.set_xlim(0, max(r["file_gb"] for tb in tables.values() for r in tb) * 1.05)
            ax.set_xlabel("file size (GB): equal memory, different model sizes")
        ax.set_ylabel(label)
    axes[0].set_title(f"same quant types, {len(tables)} model sizes", loc="left", color=INK)
    axes[1].set_title("π recall by file size", loc="left", color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGURES / "H5_model_size.png")
    plt.close(fig)


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
        table = headline(model, cfg, rows, meta)
        fig_h1(model, table)
        fig_tradeoffs(model, table)
        fig_speed(model, table)
        fig_h4(model, table)
        fig_h2(model, cfg, table)
        fig_errors(model, rows, table, "E3", "verbatim copy errors by type")
        fig_errors(model, rows, table, "E4b", "spell-as-words errors by type (clarified prompt)")
        fig_h6(model, rows, table)
        print(f"{model}: {len(table)} variants -> scores/{model}_headline.md, figures/{model}_*.png")
    fig_h5(list(cfg["models"]))


if __name__ == "__main__":
    main()
