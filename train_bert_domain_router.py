"""
img4.py — BERT domain router 기반 hyperparameter sensitivity analysis
"""
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROUTER_DIR = "/home/sclab/moe_model/routers/bert_domain_router"
DATA_DIR   = "/home/sclab/moe_model/processed_v2"

LABEL2ID = {"LAW": 0, "MED": 1, "STEM": 2, "SOC": 3, "CROSS": 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

MAX_LEN    = 128
BATCH_SIZE = 64

# ── 모델 로드 ────────────────────────────────────────
print("BERT router 로드 중...")
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(ROUTER_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(ROUTER_DIR)
model.to(device)
model.eval()
print(f"  Device: {device}")

# ── Val 데이터 로드 ──────────────────────────────────
print("검증 데이터 로드 중...")
with open(os.path.join(DATA_DIR, "router_val.json")) as f:
    val_data = json.load(f)

val_X, val_y = [], []
for item in val_data:
    domain = item.get("domain","").upper()
    if domain not in LABEL2ID:
        continue
    val_X.append(item["instruction"].strip())
    val_y.append(domain)
print(f"  검증 샘플: {len(val_X)}개")

# ── 확률 일괄 계산 ───────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.enc = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt"
        )
    def __len__(self): return self.enc["input_ids"].shape[0]
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.enc.items()}

print("확률 계산 중...")
dl    = DataLoader(TextDataset(val_X, tokenizer), batch_size=BATCH_SIZE)
probs = []
with torch.no_grad():
    for batch in tqdm(dl, desc="inference"):
        batch   = {k: v.to(device) for k, v in batch.items()}
        logits  = model(**batch).logits
        prob    = torch.softmax(logits, dim=-1).cpu().numpy()
        probs.append(prob)
probs = np.vstack(probs)
print(f"  확률 shape: {probs.shape}")

# ── Grid search ──────────────────────────────────────
def evaluate(probs, labels, tau, delta):
    correct = 0
    for prob, true_label in zip(probs, labels):
        sorted_idx = np.argsort(prob)[::-1]
        s1 = prob[sorted_idx[0]]
        s2 = prob[sorted_idx[1]]
        top1 = ID2LABEL[sorted_idx[0]]
        pred = "CROSS" if (s2 >= tau and (s1 - s2) <= delta) else top1
        if pred == true_label:
            correct += 1
    return correct / len(labels) * 100

tau_range   = np.round(np.arange(0.10, 0.55, 0.05), 2)
delta_range = np.round(np.arange(0.10, 0.65, 0.05), 2)

print("\nGrid search 시작...")
grid = np.zeros((len(tau_range), len(delta_range)))
for i, tau in enumerate(tqdm(tau_range, desc="tau")):
    for j, delta in enumerate(delta_range):
        grid[i, j] = evaluate(probs, val_y, tau, delta)

opt_idx   = np.unravel_index(np.argmax(grid), grid.shape)
opt_tau   = tau_range[opt_idx[0]]
opt_delta = delta_range[opt_idx[1]]
opt_acc   = grid[opt_idx]
print(f"\n최적: τ={opt_tau:.2f}, δ={opt_delta:.2f} → {opt_acc:.2f}%")

# paper 값 기준 성능도 출력
paper_tau_idx   = np.argmin(np.abs(tau_range   - 0.30))
paper_delta_idx = np.argmin(np.abs(delta_range - 0.40))
paper_acc = grid[paper_tau_idx, paper_delta_idx]
print(f"논문 값 τ=0.30, δ=0.40 → {paper_acc:.2f}%")

# sensitivity curves
delta_fixed_idx = np.argmin(np.abs(delta_range - opt_delta))
tau_acc         = grid[:, delta_fixed_idx]
tau_fixed_idx   = np.argmin(np.abs(tau_range - opt_tau))
delta_acc       = grid[tau_fixed_idx, :]

# ── Plot ─────────────────────────────────────────────
BLUE    = "#185FA5"
BLUE_LT = "#B5D4F4"
RED     = "#E24B4A"

fig = plt.figure(figsize=(16, 12))
fig.suptitle("Hyperparameter Sensitivity Analysis: CROSS Detection Thresholds",
             fontsize=14, fontweight='bold', y=0.98)
gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

# τ sensitivity
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(tau_range, tau_acc, 'o-', color=BLUE, linewidth=2, markersize=5,
         markerfacecolor=BLUE_LT, markeredgecolor=BLUE, markeredgewidth=1.5)
ax1.axvline(x=opt_tau, color=RED, linestyle='--', linewidth=1.5, alpha=0.8,
            label=f'τ = {opt_tau:.2f} (optimal)')
ax1.scatter([opt_tau], [grid[opt_idx]], color=RED, s=120, zorder=5, marker='*')
ax1.fill_between(tau_range, grid.max()-2, tau_acc,
                 where=(tau_acc >= grid.max()-2), alpha=0.12, color=BLUE)
ax1.set_xlabel('τ  (min activation threshold)', fontsize=11)
ax1.set_ylabel('Routing accuracy (%)', fontsize=11)
ax1.set_title(f'τ sensitivity  (δ = {opt_delta:.2f} fixed)', fontsize=11)
ax1.set_ylim(max(tau_acc.min()-1, 85), min(tau_acc.max()+1, 100))
ax1.set_xticks(tau_range)
ax1.tick_params(axis='x', labelsize=9, rotation=45)
ax1.tick_params(axis='y', labelsize=9)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f}%'))

# δ sensitivity
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(delta_range, delta_acc, 's-', color=BLUE, linewidth=2, markersize=5,
         markerfacecolor=BLUE_LT, markeredgecolor=BLUE, markeredgewidth=1.5)
ax2.axvline(x=opt_delta, color=RED, linestyle='--', linewidth=1.5, alpha=0.8,
            label=f'δ = {opt_delta:.2f} (optimal)')
ax2.scatter([opt_delta], [grid[opt_idx]], color=RED, s=120, zorder=5, marker='*')
ax2.fill_between(delta_range, grid.max()-2, delta_acc,
                 where=(delta_acc >= grid.max()-2), alpha=0.12, color=BLUE)
ax2.set_xlabel('δ  (max score gap)', fontsize=11)
ax2.set_ylabel('Routing accuracy (%)', fontsize=11)
ax2.set_title(f'δ sensitivity  (τ = {opt_tau:.2f} fixed)', fontsize=11)
ax2.set_ylim(max(delta_acc.min()-1, 85), min(delta_acc.max()+1, 100))
ax2.set_xticks(delta_range)
ax2.tick_params(axis='x', labelsize=9, rotation=45)
ax2.tick_params(axis='y', labelsize=9)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f}%'))

# Joint heatmap
ax3 = fig.add_subplot(gs[1, :])
sns.heatmap(
    grid,
    ax=ax3,
    xticklabels=[f'{d:.2f}' for d in delta_range],
    yticklabels=[f'{t:.2f}' for t in tau_range],
    cmap='Blues',
    annot=True,
    fmt='.1f',
    annot_kws={'size': 8},
    linewidths=0.3,
    linecolor='white',
    cbar_kws={'label': 'Routing accuracy (%)', 'shrink': 0.6},
    vmin=grid.min(), vmax=100
)
# optimal point
ax3.add_patch(mpatches.Rectangle(
    (opt_idx[1], opt_idx[0]), 1, 1,
    fill=False, edgecolor=RED, linewidth=3
))
ax3.text(opt_idx[1]+0.5, opt_idx[0]-0.2, '★',
         ha='center', va='bottom', fontsize=14, color=RED)

ax3.set_xlabel('δ  (max score gap)', fontsize=11)
ax3.set_ylabel('τ  (min activation threshold)', fontsize=11)
ax3.set_title(
    f'Joint grid search — optimal: τ={opt_tau:.2f}, δ={opt_delta:.2f} → {opt_acc:.2f}%   '
    f'(논문 τ=0.30, δ=0.40 → {paper_acc:.2f}%)',
    fontsize=11
)
ax3.tick_params(axis='x', labelsize=9, rotation=45)
ax3.tick_params(axis='y', labelsize=9, rotation=0)

plt.savefig('/home/sclab/moe_model/img4.png', dpi=300,
            bbox_inches='tight', facecolor='white')
print("\n저장 완료: /home/sclab/moe_model/img4.png")
plt.close()