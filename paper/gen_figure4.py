"""
Figure 4 — NexRoute confusion matrices (TF-IDF vs NexRoute)
Data source: results/fig4_counts_tfidf_lr.csv, results/fig4_counts_nexroute.csv
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ─── Load from CSV (source of truth) ───────────────────────────────────────────
root = Path(__file__).parent.parent          # /home/sclab/moe_model
labels = ["LAW", "MED", "SOC", "STEM", "CROSS"]

def load_cm(path):
    df = pd.read_csv(path, index_col=0)
    return df.loc[labels, labels].values.astype(float)

tfidf   = load_cm(root / "results/fig4_counts_tfidf_lr.csv")
nexroute = load_cm(root / "results/fig4_counts_nexroute.csv")

# Sanity check
assert (tfidf.sum(axis=1)    == 100).all(), "TF-IDF row sums ≠ 100"
assert (nexroute.sum(axis=1) == 100).all(), "NexRoute row sums ≠ 100"
assert tfidf.sum()    == 500
assert nexroute.sum() == 500

n_total       = 500
tfidf_correct = int(np.trace(tfidf))
nex_correct   = int(np.trace(nexroute))
tfidf_acc     = tfidf_correct   / n_total * 100
nex_acc       = nex_correct     / n_total * 100

# ─── Plotting ──────────────────────────────────────────────────────────────────
def plot_confmat(ax, mat, title):
    """Plot one 5×5 confusion matrix with count and row-% annotations."""
    im = ax.imshow(mat, vmin=0, vmax=100, cmap="Blues", aspect="equal")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(range(5));  ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks(range(5));  ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted Domain", fontsize=11)
    ax.set_ylabel("Gold Domain",      fontsize=11)

    # Minor grid lines between cells
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate: count (row %)
    # Since each row = 100 samples, count == row-percentage
    for i in range(5):
        for j in range(5):
            v = int(mat[i, j])
            color = "white" if v > 50 else "black"
            weight = "bold" if i == j else "normal"
            # Diagonal: show count + (%)
            if i == j:
                txt = f"{v}\n({v}%)"
            else:
                txt = f"{v}" if v > 0 else ""
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=9.5, color=color, fontweight=weight)

    return im

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4),
                          constrained_layout=True)

im1 = plot_confmat(
    axes[0], tfidf,
    f"(a) TF-IDF Domain Router\nAcc = {tfidf_acc:.1f}%  "
    f"({tfidf_correct} / {n_total} correct)"
)
im2 = plot_confmat(
    axes[1], nexroute,
    f"(b) NexRoute\nAcc = {nex_acc:.1f}%  "
    f"({nex_correct} / {n_total} correct)"
)

# Shared colorbar
cbar = fig.colorbar(im2, ax=axes, shrink=0.88, pad=0.02)
cbar.set_label("Count  (= row-%, since n = 100 per gold domain)",
               fontsize=10)

# Footer note
fig.text(
    0.5, -0.01,
    r"N = 500  (100 samples per domain).  "
    r"NexRoute uses DeBERTa-large router with $\tau$ = 0.30, $\delta$ = 0.40.",
    ha="center", fontsize=9.5, style="italic"
)

# ─── Save ──────────────────────────────────────────────────────────────────────
out_dir = Path(__file__).parent          # paper/
out_png = out_dir / "figure4_revised_final.png"
out_pdf = out_dir / "figure4_revised_final.pdf"

# Remove broken symlink if present
if out_png.is_symlink() or out_png.exists():
    out_png.unlink(missing_ok=True)
if out_pdf.is_symlink() or out_pdf.exists():
    out_pdf.unlink(missing_ok=True)

plt.savefig(out_png, dpi=500, bbox_inches="tight")
plt.savefig(out_pdf,           bbox_inches="tight")
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")

# Also copy to figure/ directory for backup
fig_dir = root / "figure"
fig_dir.mkdir(exist_ok=True)
import shutil
shutil.copy(out_png, fig_dir / "figure4_revised_final.png")
shutil.copy(out_pdf, fig_dir / "figure4_revised_final.pdf")
print(f"Backup: {fig_dir}/figure4_revised_final.png")
