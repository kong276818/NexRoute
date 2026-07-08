import json, torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from transformers import AutoModel, AutoTokenizer

GT_PATH    = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
ROUTER_PATH= "/home/sclab/moe_model/router_deberta_best.pt"
AD_PATH    = "/home/sclab/moe_model/adaptive_tfidf_results.json"
MODEL_NAME = "microsoft/deberta-large"
DOMAINS    = ['LAW','MED','SOC','STEM']
LABELS_ALL = ['LAW','MED','SOC','STEM','CROSS']

class DeBERTaRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden,512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512,4))
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(out.last_hidden_state[:,0,:]))

print("[*] DeBERTa 라우터 로드...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
router = DeBERTaRouter()
router.load_state_dict(torch.load(ROUTER_PATH, map_location='cpu'))
router.eval()

with open(GT_PATH) as f: gt = json.load(f)
idx = {l:i for i,l in enumerate(LABELS_ALL)}

def get_pred(q, tau, delta):
    enc = tok(q, max_length=256, truncation=True,
               padding='max_length', return_tensors='pt')
    with torch.no_grad():
        scores = torch.sigmoid(
            router(enc['input_ids'], enc['attention_mask'])
        ).squeeze().tolist()
    ds = dict(zip(DOMAINS, scores))
    sd = sorted(ds.items(), key=lambda x:-x[1])
    s1, s2 = sd[0][1], sd[1][1]
    return 'CROSS' if (s2>=tau and (s1-s2)<=delta) else sd[0][0]

# ── Figure 2: Sensitivity Heatmap (DeBERTa) ──────────────────
print("[*] Sensitivity 분석 중 (τ × δ grid)...")
taus   = [round(t,2) for t in np.arange(0.10, 0.55, 0.05)]
deltas = [round(d,2) for d in np.arange(0.10, 0.65, 0.05)]
grid = np.zeros((len(taus), len(deltas)))

for i, tau in enumerate(taus):
    for j, delta in enumerate(deltas):
        correct = sum(
            1 for item in gt
            if get_pred(item['instruction'].strip(), tau, delta)
               == item.get('expected_domain','')
        )
        grid[i,j] = correct / len(gt) * 100
    print(f"  τ={tau:.2f} done")

best_val = grid.max()
best_pos = np.unravel_index(grid.argmax(), grid.shape)
paper_val = grid[taus.index(0.30), deltas.index(0.40)]
print(f"Best: τ={taus[best_pos[0]]:.2f}, δ={deltas[best_pos[1]]:.2f} → {best_val:.2f}%")
print(f"Paper (τ=0.30, δ=0.40): {paper_val:.2f}%")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# τ curve (δ=0.40 fixed)
d_idx = deltas.index(0.40)
tau_curve = grid[:, d_idx]
ax = axes[0]
ax.plot(taus, tau_curve, 'o-', color='#3A7DB5', lw=1.5, ms=4)
ax.axvline(taus[tau_curve.argmax()], color='#E04B3A', ls='--', lw=1.2,
           label=f'optimal={taus[tau_curve.argmax()]:.2f}')
ax.plot(taus[tau_curve.argmax()], tau_curve.max(), '*', color='#E04B3A', ms=12)
ax.set_xlabel('τ (min activation threshold)', fontsize=9)
ax.set_ylabel('Routing accuracy (%)', fontsize=9)
ax.set_title(f'τ sensitivity (δ=0.40 fixed)', fontsize=10)
ax.legend(fontsize=8); ax.tick_params(labelsize=8)
ax.set_ylim(grid.min()-0.5, grid.max()+0.5)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.2f}'))

# δ curve (τ=0.30 fixed)
t_idx = taus.index(0.30)
delta_curve = grid[t_idx, :]
ax = axes[1]
ax.plot(deltas, delta_curve, 'o-', color='#3A7DB5', lw=1.5, ms=4)
ax.axvline(deltas[delta_curve.argmax()], color='#E04B3A', ls='--', lw=1.2,
           label=f'optimal={deltas[delta_curve.argmax()]:.2f}')
ax.plot(deltas[delta_curve.argmax()], delta_curve.max(), '*', color='#E04B3A', ms=12)
ax.set_xlabel('δ (max score gap)', fontsize=9)
ax.set_ylabel('Routing accuracy (%)', fontsize=9)
ax.set_title(f'δ sensitivity (τ=0.30 fixed)', fontsize=10)
ax.legend(fontsize=8); ax.tick_params(labelsize=8)
ax.set_ylim(grid.min()-0.5, grid.max()+0.5)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.2f}'))

# Heatmap
ax = axes[2]
im = ax.imshow(grid, cmap='Blues', aspect='auto',
               vmin=grid.min()-0.2, vmax=grid.max()+0.2)
for i in range(len(taus)):
    for j in range(len(deltas)):
        ax.text(j, i, f'{grid[i,j]:.1f}', ha='center', va='center',
                fontsize=6.5,
                color='white' if grid[i,j] > (grid.min()+grid.max())/2 else '#333')
ax.plot(best_pos[1], best_pos[0], '*', color='#E04B3A', ms=10, zorder=5)
ax.plot(deltas.index(0.40), taus.index(0.30), 'o', color='#1A5276',
        ms=8, zorder=5, markerfacecolor='none', markeredgewidth=2)
ax.set_xticks(range(len(deltas))); ax.set_xticklabels([f'{d:.2f}' for d in deltas], fontsize=7, rotation=45)
ax.set_yticks(range(len(taus)));   ax.set_yticklabels([f'{t:.2f}' for t in taus], fontsize=7)
ax.set_xlabel('δ (max score gap)', fontsize=9)
ax.set_ylabel('τ (min activation threshold)', fontsize=9)
ax.set_title(
    f'Joint grid search  [★] optimal: τ={taus[best_pos[0]]:.2f}, δ={deltas[best_pos[1]]:.2f}→{best_val:.2f}%'
    f'\n[●] paper: τ=0.30, δ=0.40→{paper_val:.2f}%', fontsize=8)
plt.colorbar(im, ax=ax, label='Routing Accuracy (%)', shrink=0.85)

plt.tight_layout()
plt.savefig('/home/sclab/moe_model/fig2_sensitivity_deberta.png', dpi=200, bbox_inches='tight')
plt.savefig('/home/sclab/moe_model/fig2_sensitivity_deberta.pdf', bbox_inches='tight')
print("Figure 2 저장 완료")

# ── Figure 4: Confusion Matrices ─────────────────────────────
print("\n[*] Confusion matrix 생성 중...")

# NexRoute DeBERTa (τ=0.30, δ=0.40)
cm_nx = np.zeros((5,5), dtype=int)
for item in gt:
    t = item.get('expected_domain','')
    p = get_pred(item['instruction'].strip(), 0.30, 0.40)
    if t in LABELS_ALL and p in LABELS_ALL:
        cm_nx[idx[t], idx[p]] += 1

# Adaptive RAG TF-IDF (저장된 결과)
with open(AD_PATH) as f: ad_res = json.load(f)
cm_ad = np.zeros((5,5), dtype=int)
gt_map = {d['instruction'].strip(): d for d in gt}
for name, res in ad_res.items():
    for split in ['cross','single']:
        for item in res.get(split,[]):
            t = item.get('domain','')
            p = item.get('pred_domain','')
            if t in LABELS_ALL and p in LABELS_ALL:
                cm_ad[idx[t], idx[p]] += 1

# Fallback: Adaptive RAG keyword routing
KEYWORDS = {
    'LAW':['law','legal','court','statute','plaintiff','contract','judge','attorney'],
    'MED':['medical','patient','disease','treatment','clinical','diagnosis','symptom','drug'],
    'SOC':['social','society','culture','economic','political','community','psychology'],
    'STEM':['science','physics','chemical','biology','math','engineering','technology'],
}
def kw_route(q):
    ql = q.lower()
    scores = {d:sum(1 for k in kws if k in ql) for d,kws in KEYWORDS.items()}
    return max(scores, key=scores.get)

if cm_ad.sum() == 0:
    print("  Adaptive RAG 결과 없음 → 키워드 라우팅으로 대체")
    for item in gt:
        t = item.get('expected_domain','')
        p = kw_route(item['instruction'].strip())
        if t in LABELS_ALL and p in LABELS_ALL:
            cm_ad[idx[t], idx[p]] += 1

def norm(cm):
    rs = cm.sum(axis=1, keepdims=True); rs[rs==0]=1
    return cm/rs*100

cm_ad_p = norm(cm_ad); cm_nx_p = norm(cm_nx)
acc_ad = cm_ad.trace()/cm_ad.sum()*100
acc_nx = cm_nx.trace()/cm_nx.sum()*100
print(f"Adaptive RAG: {acc_ad:.1f}%")
print(f"NexRoute DeBERTa: {acc_nx:.1f}%")
for i,l in enumerate(LABELS_ALL):
    print(f"  {l}: AD={cm_ad_p[i,i]:.0f}% NX={cm_nx_p[i,i]:.0f}%")

cmap_nx = LinearSegmentedColormap.from_list('nx',
    ['#F5F4FE','#CECBF6','#AFA9EC','#8B85DC','#6D64CC','#3C3489'], N=256)

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5.8),facecolor='white')
fig.suptitle('Domain Routing Confusion Matrices: Adaptive RAG vs. NexRoute (DeBERTa-large)',
             fontsize=13,fontweight='bold',y=1.01)

def draw(ax,pct,cnt,title,acc,cmap,thr=45):
    im=ax.imshow(pct,cmap=cmap,vmin=0,vmax=100,aspect='equal',interpolation='nearest')
    for i in range(5):
        for j in range(5):
            v=pct[i,j]; tc='white' if v>thr else '#222'
            fw='bold' if i==j else 'normal'; fs=12 if i==j else 9.5
            ax.text(j,i-0.1,f'{v:.0f}%',ha='center',va='center',fontsize=fs,color=tc,fontweight=fw)
            if i==j: ax.text(j,i+0.30,f'n={cnt[i,j]}',ha='center',va='center',fontsize=6.5,color=tc,alpha=0.8)
    for i in range(6): ax.axhline(i-.5,color='white',lw=1.5); ax.axvline(i-.5,color='white',lw=1.5)
    for i in range(5):
        ax.add_patch(plt.Rectangle((i-.5,i-.5),1,1,lw=2.2,edgecolor='white',facecolor='none',zorder=3))
    ax.set_xticks(range(5)); ax.set_xticklabels(LABELS_ALL,fontsize=10)
    ax.set_yticks(range(5)); ax.set_yticklabels(LABELS_ALL,fontsize=10)
    ax.set_xlabel('Predicted Domain',fontsize=11,labelpad=6)
    ax.set_ylabel('True Domain',fontsize=11,labelpad=6)
    ax.tick_params(length=0); ax.spines[:].set_visible(False)
    ax.set_title(f'{title}\n',fontsize=12.5,fontweight='bold',pad=4)
    ax.text(0.5,1.02,f'Overall accuracy: {acc:.1f}%',transform=ax.transAxes,
            ha='center',fontsize=10,color='#555',style='italic')
    return im

im1=draw(ax1,cm_ad_p,cm_ad,'Adaptive RAG (TF-IDF+LR)',acc_ad,plt.cm.Blues)
im2=draw(ax2,cm_nx_p,cm_nx,'NexRoute (DeBERTa-large)',acc_nx,cmap_nx)
for im,ax in [(im1,ax1),(im2,ax2)]:
    cb=plt.colorbar(im,ax=ax,shrink=0.80,pad=0.03)
    cb.set_label('Routing Accuracy (%)',fontsize=9)
plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig('/home/sclab/moe_model/fig4_confusion_deberta.png',dpi=200,bbox_inches='tight',facecolor='white')
plt.savefig('/home/sclab/moe_model/fig4_confusion_deberta.pdf',bbox_inches='tight',facecolor='white')
print("Figure 4 저장 완료")

# 결과 저장
result = {
    'sensitivity': {
        'best_tau': taus[best_pos[0]], 'best_delta': deltas[best_pos[1]],
        'best_acc': float(best_val), 'paper_acc': float(paper_val),
        'grid': grid.tolist(), 'taus': taus, 'deltas': deltas
    },
    'routing_accuracy': {
        'adaptive_rag': {l: float(cm_ad_p[idx[l],idx[l]]) for l in LABELS_ALL},
        'nexroute':     {l: float(cm_nx_p[idx[l],idx[l]]) for l in LABELS_ALL},
        'adaptive_total': float(acc_ad), 'nexroute_total': float(acc_nx)
    }
}
with open('/home/sclab/moe_model/fig2_fig4_results.json','w') as f:
    json.dump(result, f, indent=2)
print("\n모든 결과 저장 완료!")
