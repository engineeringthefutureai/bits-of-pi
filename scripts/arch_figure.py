"""Diagram of the Qwen 3.5 4B layer stack and the tensors inside each kind of layer.

usage: arch_figure.py   writes figures/qwen3.5-4b_architecture.png

Numbers are read from Qwen3.5-4B-BF16.gguf (32 layers, full attention where (i + 1) % 4 == 0, as in
llama.cpp's src/models/qwen35.cpp). Shares are of all 4.21 billion weights, summed over the layers
that hold the tensor.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from analyze import INK, MUTED, tint
from common import FIGURES

LINEAR, FULL, FFN, OTHER = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
N_LAYERS, INTERVAL = 32, 4


def box(ax, x, y, w, h, text, color, fs=8, bold=False, sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.6",
                                fc=tint(color), ec=color, lw=1.0))
    ty = y + h / 2 + (0.55 if sub else 0)
    ax.text(x + w / 2, ty, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", family="monospace" if not bold else None)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.75, sub, ha="center", va="center", fontsize=6.5, color=MUTED)


def arrow(ax, x0, y0, x1, y1, color=MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
                                 color=color, lw=0.9, shrinkA=0, shrinkB=0))


def plus(ax, x, y):
    ax.add_patch(matplotlib.patches.Circle((x, y), 0.8, fc="white", ec=MUTED, lw=0.9))
    ax.text(x, y - 0.05, "+", ha="center", va="center", fontsize=9, color=INK)


def ffn_block(ax, x, y, w):
    """Shared feed-forward half, drawn from y upward. Returns the top y."""
    box(ax, x + w * 0.2, y, w * 0.6, 2.2, "post_attention_norm", OTHER, fs=7)
    arrow(ax, x + w / 2, y + 2.2, x + w / 2, y + 3.4)
    half = w * 0.46
    box(ax, x, y + 3.4, half, 3.2, "ffn_gate", FFN, sub="18%")
    box(ax, x + w - half, y + 3.4, half, 3.2, "ffn_up", FFN, sub="18%")
    ax.text(x + w / 2, y + 7.6, "SwiGLU: silu(gate) × up", ha="center", va="center", fontsize=6.5,
            color=MUTED)
    arrow(ax, x + w / 2, y + 8.3, x + w / 2, y + 9.2)
    box(ax, x + w * 0.2, y + 9.2, w * 0.6, 3.2, "ffn_down", FFN, sub="18%")
    return y + 12.4


def linear_layer(ax, x, y, w):
    c = x + w / 2
    box(ax, x + w * 0.25, y, w * 0.5, 2.2, "attn_norm", OTHER, fs=7)
    arrow(ax, c, y + 2.2, c, y + 3.2)
    box(ax, x, y + 3.2, w * 0.62, 3.2, "attn_qkv", LINEAR, sub="12%  query, key, value")
    box(ax, x + w * 0.66, y + 3.2, w * 0.34, 3.2, "attn_gate", LINEAR, sub="6%")
    arrow(ax, x + w * 0.31, y + 6.4, x + w * 0.31, y + 7.4)
    box(ax, x, y + 7.2, w * 0.62, 2.9, "ssm_conv1d", LINEAR, fs=7, sub="short convolution")
    arrow(ax, x + w * 0.31, y + 10.1, x + w * 0.31, y + 10.8)
    ax.add_patch(Rectangle((x, y + 10.8), w * 0.62, 6.2, fc="white", ec=LINEAR, lw=1.0, ls="--"))
    ax.text(x + w * 0.31, y + 16.2, "fixed-size state, updated every token", ha="center", va="center",
            fontsize=6.5, color=MUTED)
    box(ax, x + 0.6, y + 11.4, w * 0.62 / 2 - 0.9, 3.4, "ssm_alpha, _beta", LINEAR, fs=6.5,
        sub="keep / write, 0.1%")
    box(ax, x + w * 0.31 + 0.3, y + 11.4, w * 0.31 - 0.9, 3.4, "ssm_a, ssm_dt", LINEAR, fs=6.5,
        sub="decay constants")
    arrow(ax, x + w * 0.31, y + 17.0, x + w * 0.31, y + 18.0)
    box(ax, x, y + 18.0, w * 0.62, 2.2, "ssm_norm", OTHER, fs=7)
    gx = x + w * 0.83
    ax.plot([gx, gx], [y + 6.4, y + 19.1], color=LINEAR, lw=0.9)
    arrow(ax, gx, y + 19.1, x + w * 0.62, y + 19.1, color=LINEAR)
    ax.text(gx + 0.6, y + 13, "× gate", rotation=90, ha="left", va="center", fontsize=6.5, color=MUTED)
    arrow(ax, x + w * 0.31, y + 20.2, x + w * 0.31, y + 21.0)
    box(ax, x + w * 0.2, y + 21.0, w * 0.6, 3.2, "ssm_out", LINEAR, sub="6%")
    plus(ax, c, y + 25.4)
    arrow(ax, c, y + 24.2, c, y + 24.6)
    arrow(ax, c, y + 26.2, c, y + 27.2)
    top = ffn_block(ax, x, y + 27.2, w)
    plus(ax, c, top + 1.2)
    arrow(ax, c, top, c, top + 0.4)
    return top + 2.0


def full_layer(ax, x, y, w):
    c = x + w / 2
    box(ax, x + w * 0.25, y, w * 0.5, 2.2, "attn_norm", OTHER, fs=7)
    arrow(ax, c, y + 2.2, c, y + 3.2)
    third = (w - 1.2) / 3
    for i, (name, share) in enumerate([("attn_q", "4%"), ("attn_k", "0.5%"), ("attn_v", "0.5%")]):
        box(ax, x + i * (third + 0.6), y + 3.2, third, 3.2, name, FULL, sub=share)
    ax.text(c, y + 7.3, "q_norm, k_norm on query and key", ha="center", va="center", fontsize=6.5,
            color=MUTED)
    arrow(ax, c, y + 7.9, c, y + 9.0)
    ax.add_patch(Rectangle((x, y + 9.0), w, 8.0, fc="white", ec=FULL, lw=1.0, ls="--"))
    ax.text(c, y + 14.2, "attention over every earlier token", ha="center", va="center", fontsize=7,
            color=INK)
    ax.text(c, y + 12.0, "keys and values kept in the KV cache,\n16 query heads share 4 key/value heads",
            ha="center", va="center", fontsize=6.5, color=MUTED)
    arrow(ax, c, y + 17.0, c, y + 21.0)
    box(ax, x + w * 0.2, y + 21.0, w * 0.6, 3.2, "attn_output", FULL, sub="2%")
    plus(ax, c, y + 25.4)
    arrow(ax, c, y + 24.2, c, y + 24.6)
    arrow(ax, c, y + 26.2, c, y + 27.2)
    top = ffn_block(ax, x, y + 27.2, w)
    plus(ax, c, top + 1.2)
    arrow(ax, c, top, c, top + 0.4)
    return top + 2.0


def main():
    fig, ax = plt.subplots(figsize=(13, 6.6))
    ax.set_xlim(0, 130)
    ax.set_ylim(-5.5, 58)
    ax.axis("off")

    # left: the stack
    sx, sw, y0, lh = 6, 15, 9, 1.1
    box(ax, sx, 2, sw, 4.6, "token_embd", OTHER, sub="15%  token → vector")
    arrow(ax, sx + sw / 2, 6.6, sx + sw / 2, y0)
    for i in range(N_LAYERS):
        full = (i + 1) % INTERVAL == 0
        color = FULL if full else LINEAR
        ax.add_patch(Rectangle((sx, y0 + i * lh), sw, lh * 0.82, fc=tint(color, 0.55 if full else 0.75),
                               ec=color, lw=0.6))
        if i % 4 == 0 or i == N_LAYERS - 1:
            ax.text(sx - 0.8, y0 + i * lh + lh * 0.41, str(i), ha="right", va="center", fontsize=6.5,
                    color=MUTED)
    top = y0 + N_LAYERS * lh
    arrow(ax, sx + sw / 2, top, sx + sw / 2, top + 1.6)
    box(ax, sx, top + 1.6, sw, 4.6, "output", OTHER, sub="vector → next-token odds")
    ax.text(sx + sw / 2, top + 7.0, "tied: reuses token_embd", ha="center", va="bottom", fontsize=6.5,
            color=MUTED)
    ax.text(sx + sw / 2, 56, "Qwen 3.5 4B", ha="center", va="center", fontsize=10, fontweight="bold",
            color=INK)
    ax.text(sx + sw / 2, 54, "32 layers, 2,560 wide", ha="center", va="center", fontsize=7.5,
            color=MUTED)
    # bracket for one repeating group
    gy0, gy1 = y0, y0 + 4 * lh - lh * 0.18
    ax.plot([sx + sw + 0.8, sx + sw + 1.6, sx + sw + 1.6, sx + sw + 0.8], [gy0, gy0, gy1, gy1],
            color=INK, lw=0.8)
    ax.text(sx + sw + 2.3, (gy0 + gy1) / 2, "3 linear + 1 full,\nrepeated 8 times", ha="left",
            va="center", fontsize=6.5, color=INK)

    # right: one layer of each kind
    lx, lw_ = 36, 38
    fx, fw = 86, 38
    ax.text(lx + lw_ / 2, 56, "Linear-attention layer (Gated DeltaNet)", ha="center", va="center",
            fontsize=10, fontweight="bold", color=LINEAR)
    ax.text(lx + lw_ / 2, 54, "24 of 32 layers: 0–2, 4–6, …, 28–30", ha="center", va="center",
            fontsize=7.5, color=MUTED)
    ax.text(fx + fw / 2, 56, "Full-attention layer", ha="center", va="center", fontsize=10,
            fontweight="bold", color=FULL)
    ax.text(fx + fw / 2, 54, "8 of 32 layers: 3, 7, 11, …, 31", ha="center", va="center",
            fontsize=7.5, color=MUTED)
    base = 4.5
    for x, w in ((lx, lw_), (fx, fw)):
        ax.text(x + w / 2, base - 2.6, "from the layer below", ha="center", va="center", fontsize=6.5,
                color=MUTED)
        arrow(ax, x + w / 2, base - 1.8, x + w / 2, base)
    t1 = linear_layer(ax, lx, base, lw_)
    t2 = full_layer(ax, fx, base, fw)
    for x, w, t in ((lx, lw_, t1), (fx, fw, t2)):
        arrow(ax, x + w / 2, t - 0.6, x + w / 2, t + 0.6)
        ax.text(x + w / 2, t + 1.4, "to the layer above", ha="center", va="center", fontsize=6.5,
                color=MUTED)
        # residual stream: skip connections around both halves
        rx = x - 1.6
        ax.plot([x + w / 2, rx, rx, x + w / 2 - 0.8], [base - 1.0, base - 1.0, base + 25.4, base + 25.4],
                color=MUTED, lw=0.7, ls=":")
        ax.plot([x + w / 2, rx, rx, x + w / 2 - 0.8], [base + 26.6, base + 26.6, t - 0.8, t - 0.8],
                color=MUTED, lw=0.7, ls=":")
    ax.text(lx - 2.4, base + 13, "skip connection", rotation=90, ha="right", va="center", fontsize=6.5,
            color=MUTED)

    # legend
    lgx, lgy = 36, -2.4
    for k, (name, color) in enumerate([("linear attention", LINEAR), ("full attention", FULL),
                                       ("feed-forward", FFN), ("embedding, norms", OTHER)]):
        ax.add_patch(Rectangle((lgx + k * 13, lgy), 1.8, 1.2, fc=tint(color), ec=color, lw=0.9))
        ax.text(lgx + k * 13 + 2.6, lgy + 0.6, name, ha="left", va="center", fontsize=7.5, color=INK)
    ax.text(lgx, lgy - 1.4, "% = share of all 4.2 billion weights, summed over the layers that hold the tensor",
            ha="left", va="center", fontsize=6.5, color=MUTED)

    fig.tight_layout()
    out = FIGURES / "qwen3.5-4b_architecture.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
