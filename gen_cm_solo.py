import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

GT_PATH   = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
EVAL_PATH = "/home/sclab/moe_model/eval_results_7b_v3_500.json"
LABELS    = ['LAW','MED','SOC','STEM','CROSS']

with open(GT_PATH) as f:   gt = json.load(f)
with open(EVAL_PATH) as f: nx = json.load(f)
text_map = {d['instruction'].strip(): d for d in gt}
idx = {l:i for i,l in enumerate(LABELS)}

cm = np.zeros((5,5), dtype=int)
for e in nx:
    q = e.get('query','').strip()
    gt_item = text_map.get(q)
    if not gt_item: continue
    t = gt_item.get('expected_domain','')
    p = 'CROSS' if e.get('is_cross',False) else (e.get('domain_pred',[''])[0] if e.get('domain_pred') else '')
    if t in idx and p in idx: cm[idx[t],idx[p]] += 1

rs = cm.sum(axis=1,keepdims=True); rs[rs==0]=1
pct = cm/rs*100
acc = cm.trace()/cm.sum()*100
print(f"NexRoute 7B: {acc:.1f}%")
for i,l in enumerate(LABELS):
    print(f"  {l}: {cm[i,i]}/{cm[i].sum()} = {pct[i,i]:.0f}%")

cmap = LinearSegmentedColormap.from_list('nx',
    ['#F5F4FE','#CECBF6','#AFA9EC','#8B85DC','#6D64CC','#3C3489'],N=256)

fig, ax = plt.subplots(figsize=(7, 6.2), facecolor='white')
im = ax.imshow(pct, cmap=cmap, vmin=0, vmax=100,
               aspect='equal', interpolation='nearest')

for i in range(5):
    for j in range(5):
        v = pct[i,j]
        tc = 'white' if v > 45 else '#222'
        fw = 'bold' if i==j else 'normal'
        fs = 13 if i==j else 10
        ax.text(j, i-0.1, f'{v:.0f}%',
                ha='center', va='center',
                fontsize=fs, color=tc, fontweight=fw)
        if i==j:
            ax.text(j, i+0.32, f'n={cm[i,j]}',
                    ha='center', va='center',
                    fontsize=7.5, color=tc, alpha=0.85)

for i in range(6):
    ax.axhline(i-.5, color='white', lw=1.8)
    ax.axvline(i-.5, color='white', lw=1.8)
for i in range(5):
    ax.add_patch(plt.Rectangle((i-.5,i-.5),1,1,
                 lw=2.5, edgecolor='white',
                 facecolor='none', zorder=3))

ax.set_xticks(range(5)); ax.set_xticklabels(LABELS, fontsize=11)
ax.set_yticks(range(5)); ax.set_yticklabels(LABELS, fontsize=11)
ax.set_xlabel('Predicted Domain', fontsize=12, labelpad=8)
ax.set_ylabel('True Domain', fontsize=12, labelpad=8)
ax.tick_params(length=0); ax.spines[:].set_visible(False)
ax.set_title('NexRoute 7B — Domain Routing Confusion Matrix',
             fontsize=13, fontweight='bold', pad=10)
ax.text(0.5, 1.025, f'Overall accuracy: {acc:.1f}%  (N=500)',
        transform=ax.transAxes,
        ha='center', fontsize=11, color='#444', style='italic')

cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
cb.set_label('Routing Accuracy (%)', fontsize=10)

plt.tight_layout()
plt.savefig('/home/sclab/moe_model/fig3_nexroute_cm.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('/home/sclab/moe_model/fig3_nexroute_cm.pdf',
            bbox_inches='tight', facecolor='white')
print('저장 완료')
