"""Regenerate the paper's figures from the committed result files.

Run from the repository root:
    PYTHONPATH=src python paper/make_figures.py
Requires matplotlib. Every value plotted is read from results/*.json.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from moocr.metrics import cer
from moocr.normalization import SCORING_V1, normalize

RESULTS = Path("results")
OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)
DARK, MID, LIGHT = "#2b2b2b", "#8a8a8a", "#d6d6d6"


def load(name):
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def outcome_counts(run):
    exact = partial = catastrophic = 0
    for s in run["per_sample"]:
        c = cer(normalize(str(s["pred"]), SCORING_V1), normalize(str(s["truth"]), SCORING_V1))
        if c == 0:
            exact += 1
        elif c <= 1:
            partial += 1
        else:
            catastrophic += 1
    return exact, partial, catastrophic


def fig_outcomes():
    systems = [
        ("EasyOCR", "easyocr_heldout"),
        ("Qwen2-VL-2B", "qwen_vl_heldout"),
        ("Arbitration", "sim_arb_tau40_heldout"),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 1.9))
    for row, (label, name) in enumerate(systems):
        e, p, c = outcome_counts(load(name))
        n = e + p + c
        left = 0.0
        for value, color, text_color in ((e, DARK, "white"), (p, LIGHT, "black"), (c, MID, "white")):
            share = 100 * value / n
            ax.barh(row, share, left=left, color=color, edgecolor="white", height=0.62)
            if share >= 6:
                ax.text(left + share / 2, row, f"{share:.1f}%", ha="center", va="center",
                        color=text_color, fontsize=8)
            left += share
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels([s[0] for s in systems])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of 1,750 held-out words (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (DARK, LIGHT, MID)]
    ax.legend(handles, ["exact", "wrong, CER at most 1", "degenerate, CER above 1"],
              ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "outcomes.pdf")
    plt.close(fig)


def fig_sweep():
    sweep = load("policy_sweep_dev")
    base = load("easyocr_dev")["scores"]["normalized"]["corpus_cer"]
    flag_only = next(r for r in sweep if r["tau"] is None)
    rows = [r for r in sweep if r["tau"] is not None]
    taus = [r["tau"] for r in rows]
    cers = [r["cer_norm"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    ax.axhline(base, color=MID, linestyle="--", linewidth=1)
    ax.text(0.705, base + 0.002, "EasyOCR alone", ha="right", va="bottom", color=MID, fontsize=8)
    ax.axhline(flag_only["cer_norm"], color=MID, linestyle=":", linewidth=1)
    ax.text(0.295, flag_only["cer_norm"] + 0.002, "degeneration flag only", ha="left",
            va="bottom", color=MID, fontsize=8)
    ax.plot(taus, cers, color=DARK, marker="o", markersize=3.5, linewidth=1.2)
    chosen = next(r for r in rows if abs(r["tau"] - 0.40) < 1e-9)
    ax.plot([chosen["tau"]], [chosen["cer_norm"]], marker="o", markersize=8,
            markerfacecolor="none", markeredgecolor=DARK)
    ax.annotate("chosen", (chosen["tau"], chosen["cer_norm"]), xytext=(0.40, 0.150),
                ha="center", fontsize=8, arrowprops={"arrowstyle": "-", "color": DARK, "linewidth": 0.6})
    ax.set_xlabel("Confidence threshold on Qwen2-VL output")
    ax.set_ylabel("Dev corpus CER (normalized)")
    ax.set_ylim(0.14, 0.25)
    fig.tight_layout()
    fig.savefig(OUT / "sweep.pdf")
    plt.close(fig)


def fig_corrector():
    rows = load("corrector_calibration_dev")
    margins = [str(r["margin"]) for r in rows]
    x = range(len(rows))
    width = 0.38
    fig, ax = plt.subplots(figsize=(5.6, 2.3))
    ax.bar([i - width / 2 for i in x], [r["fixed"] for r in rows], width, color=DARK, label="words fixed")
    ax.bar([i + width / 2 for i in x], [r["broke"] for r in rows], width, color=LIGHT,
           edgecolor=MID, linewidth=0.5, label="words broken")
    for i, r in enumerate(rows):
        ax.text(i - width / 2, r["fixed"] + 1, str(r["fixed"]), ha="center", fontsize=8)
        ax.text(i + width / 2, r["broke"] + 1, str(r["broke"]), ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(margins)
    ax.set_xlabel("Required pseudo log-likelihood margin")
    ax.set_ylabel("Words out of 200")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "corrector.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_outcomes()
    fig_sweep()
    fig_corrector()
    print("figures written to", OUT)
