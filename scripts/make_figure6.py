"""make_figure6.py — Generate Figure 6 (NLI contradiction rate comparison).

Reads source data from reinference_results/factscore_paper_per_sample_*.csv.
Outputs figure/figure6_revised.png/.pdf (300 dpi) and copies them to paper/.
"""

from __future__ import annotations

import pathlib
import shutil

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent.resolve()   # repo root (scripts/../)
DATA_DIR = ROOT / "reinference_results"
PAPER_DIR = ROOT / "paper"

# ── System definitions ────────────────────────────────────────────────────────
# Mapping from display label to CSV filename stem
SYSTEMS: dict[str, str] = {
    "Naive RAG":        "naive_rag",
    "TF-IDF Router":    "tfidf_router",
    "Simplified CRAG":  "crag",
    "HyDE":             "hyde",
    "NexRoute (Ours)":  "nexroute",
}

# X-axis order as specified
ORDER = [
    "Naive RAG",
    "TF-IDF Router",
    "Simplified CRAG",
    "HyDE",
    "NexRoute (Ours)",
]

# ── Load and compute metrics ──────────────────────────────────────────────────
def load_metrics() -> dict[str, dict]:
    records: dict[str, dict] = {}
    for label, stem in SYSTEMS.items():
        csv_path = DATA_DIR / f"factscore_paper_per_sample_{stem}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Required data file not found: {csv_path}")
        df = pd.read_csv(csv_path)

        cross_mask = df["is_cross"] == True
        law_mask = df["expected_domain"] == "LAW"
        clean_mask = df["is_clean"] == True

        def rate(mask=None):
            sub = df if mask is None else df[mask]
            n = sub["n_sents"].sum()
            c = sub["n_contra"].sum()
            return 100.0 * c / n, int(n), int(c)

        overall_r, overall_n, _ = rate()
        cross_r, cross_n, _     = rate(cross_mask)
        single_r, single_n, _   = rate(~cross_mask)
        excl_law_r, excl_law_n, _ = rate(~law_mask)

        records[label] = {
            "overall":   overall_r,
            "cross":     cross_r,
            "single":    single_r,
            "n_sents":   overall_n,
            "n_sents_cross":  cross_n,
            "n_sents_single": single_n,
            "excl_law":  excl_law_r,
            "n_sents_excl_law": excl_law_n,
        }
    return records


# ── Verification table ────────────────────────────────────────────────────────
def print_verification(records: dict) -> None:
    table7_paper = {
        "Naive RAG":       (10.8, 5.7, 12.3),
        "TF-IDF Router":   (11.7, 4.7, 13.6),
        "Simplified CRAG": (9.8,  3.8, 11.6),
        "HyDE":            (11.0, 6.0, 12.5),
        "NexRoute (Ours)": (10.1, 3.6, 13.6),
    }
    table10_paper = {
        "Naive RAG":       10.3,
        "TF-IDF Router":   11.2,
        "Simplified CRAG":  8.9,
        "HyDE":            10.7,
        "NexRoute (Ours)":  7.6,
    }

    print("\n=== Verification: Table 7 (Full-500) ===")
    print(f"{'System':<20} {'Overall':>8} {'Paper':>8} {'CROSS':>7} {'Paper':>7} "
          f"{'Single':>8} {'Paper':>8} {'n_sents':>9}")
    print("-" * 75)
    for name in ORDER:
        r = records[name]
        p7 = table7_paper[name]
        print(f"{name:<20} {r['overall']:>8.3f} {p7[0]:>8.1f} "
              f"{r['cross']:>7.3f} {p7[1]:>7.1f} "
              f"{r['single']:>8.3f} {p7[2]:>8.1f} "
              f"{r['n_sents']:>9,}")

    print("\n=== Verification: Table 10 (Full-500 excl-LAW) ===")
    print(f"{'System':<20} {'excl-LAW':>9} {'Paper':>8} {'n_sents_exclLAW':>18}")
    print("-" * 58)
    for name in ORDER:
        r = records[name]
        p10 = table10_paper[name]
        match = "✓" if abs(round(r["excl_law"], 1) - p10) < 0.05 else "✗"
        print(f"{name:<20} {r['excl_law']:>9.3f} {p10:>8.1f} {r['n_sents_excl_law']:>18,}  {match}")


# ── Colour scheme ─────────────────────────────────────────────────────────────
C_BASELINE = "#bdc3c7"   # light grey  – other baselines
C_LOWEST   = "#2980b9"   # steel blue  – lowest overall (Simplified CRAG)
C_NEXROUTE = "#e67e22"   # orange      – NexRoute

def bar_colors(highlight_lowest_name: str) -> list[str]:
    cols = []
    for name in ORDER:
        if name == "NexRoute (Ours)":
            cols.append(C_NEXROUTE)
        elif name == highlight_lowest_name:
            cols.append(C_LOWEST)
        else:
            cols.append(C_BASELINE)
    return cols


# ── Figure ────────────────────────────────────────────────────────────────────
def make_figure(records: dict) -> plt.Figure:
    overall_vals = [records[n]["overall"] for n in ORDER]
    cross_vals   = [records[n]["cross"]   for n in ORDER]

    nexroute_nsents = records["NexRoute (Ours)"]["n_sents"]
    baseline_names  = [n for n in ORDER if n != "NexRoute (Ours)"]
    baseline_avg_n  = np.mean([records[n]["n_sents"] for n in baseline_names])

    nexroute_excl  = records["NexRoute (Ours)"]["excl_law"]
    crag_excl      = records["Simplified CRAG"]["excl_law"]
    naive_excl     = records["Naive RAG"]["excl_law"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    plt.rcParams.update({"font.size": 10})

    x = np.arange(len(ORDER))
    width = 0.55

    # ── Left: Overall ────────────────────────────────────────────────────────
    ax = axes[0]
    c_left = bar_colors("Simplified CRAG")
    bars = ax.bar(x, overall_vals, width=width, color=c_left,
                  edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars, overall_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.12,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Overall NLI Contradiction Rate (Full-500)", fontsize=11, pad=8)
    ax.set_ylabel("NLI contradiction rate (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, rotation=18, ha="right", fontsize=9)
    ax.set_ylim(0, max(overall_vals) * 1.22)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    legend_left = [
        mpatches.Patch(color=C_LOWEST,   label="Lowest overall — Simplified CRAG"),
        mpatches.Patch(color=C_NEXROUTE, label="NexRoute (Ours)"),
        mpatches.Patch(color=C_BASELINE, label="Other baselines"),
    ]
    ax.legend(handles=legend_left, fontsize=8, loc="upper left")

    # ── Right: CROSS ─────────────────────────────────────────────────────────
    ax = axes[1]
    c_right = bar_colors("NexRoute (Ours)")   # NexRoute is lowest on CROSS
    # Override: Simplified CRAG also highlighted (second lowest); make it a
    # distinct intermediate colour so the lowest is still unambiguous.
    c_right_adj: list[str] = []
    for name, col in zip(ORDER, c_right):
        if name == "Simplified CRAG":
            c_right_adj.append("#7fb3d3")    # lighter blue, distinct but not primary
        else:
            c_right_adj.append(col)

    bars_r = ax.bar(x, cross_vals, width=width, color=c_right_adj,
                    edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars_r, cross_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("CROSS-Domain NLI Contradiction Rate (Full-500)", fontsize=11, pad=8)
    ax.set_ylabel("NLI contradiction rate (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, rotation=18, ha="right", fontsize=9)
    ax.set_ylim(0, max(cross_vals) * 1.28)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    legend_right = [
        mpatches.Patch(color=C_NEXROUTE, label="Lowest CROSS — NexRoute (Ours)"),
        mpatches.Patch(color="#7fb3d3",  label="Simplified CRAG"),
        mpatches.Patch(color=C_BASELINE, label="Other baselines"),
    ]
    ax.legend(handles=legend_right, fontsize=8, loc="upper right")

    # ── Footnote ──────────────────────────────────────────────────────────────
    footnote = (
        f"Full-500. NexRoute n_sents = {nexroute_nsents:,} vs. baseline avg = "
        f"{baseline_avg_n:.0f} (LAW adapter degeneration; see Section 4.5). "
        f"Excl. LAW (Table 10): NexRoute {nexroute_excl:.1f}%, "
        f"Simplified CRAG {crag_excl:.1f}%, Naive RAG {naive_excl:.1f}%."
    )
    fig.text(0.5, -0.01, footnote,
             ha="center", va="top", fontsize=8, style="italic",
             color="#444444")

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    return fig


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    records = load_metrics()
    print_verification(records)

    fig = make_figure(records)

    out_png = ROOT / "figure" / "figure6_revised.png"
    out_pdf = ROOT / "figure" / "figure6_revised.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")

    # Copy to paper/ so LaTeX can find them
    shutil.copy(out_png, PAPER_DIR / "figure6_revised.png")
    shutil.copy(out_pdf, PAPER_DIR / "figure6_revised.pdf")
    print(f"Copied to: {PAPER_DIR / 'figure6_revised.png'}")
    print(f"Copied to: {PAPER_DIR / 'figure6_revised.pdf'}")

    plt.close(fig)
