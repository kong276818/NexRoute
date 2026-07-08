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
cm_nx = np.zeros((5,5),dtype=int)
for e in nx:
    q = e.get('query','').strip()
    gt_item = text_map.get(q)
    if not gt_item: continue
    t = gt_item.get('expected_domain','')
    p = 'CROSS' if e.get('is_cross',False) else (e.get('domain_pred',[''])[0] if e.get('domain_pred') else '')
    if t in idx and p in idx: cm_nx[idx[t],idx[p]] += 1
KEYWORDS = {
    'LAW':  ['law','legal','court','statute','plaintiff','contract','judge','attorney'],
    'MED':  ['medical','patient','disease','treatment','clinical','diagnosis','symptom','drug'],
    'SOC':  ['social','society','culture','economic','political','community','psychology'],
    'STEM': ['science','physics','chemical','biology','math','engineering','technology'],
}
def keyword_route(query):
    q = query.lower()
    scores = {d: sum(1 for k in kws if k in q) for d,kws in KEYWORDS.items()}
    best = max(scores.values())
    if best == 0: return 'SOC'
    sd = sorted(scores.items(), key=lambda x:-x[1])
    if len(sd)>=2 and sd[1][1]>=sd[0][1]*0.7 and sd[0][1]>=2: return 'CROSS'
    return sd[0][0]
cm_ad = np.zeros((5,5),dtype=int)
for d in gt:
    t = d.get('expected_domain','')
    p = keyword_route(d.get('instruction',''))
    if t in idx and p in idx: cm_ad[idx[t],idx[p]] += 1
def norm(cm):
    rs = cm.sum(axis=1,keepdims=True); rs[rs==0]=1; return cm/rs*100
cm_ad_p = norm(cm_ad); cm_nx_p = norm(cm_nx)
acc_ad = cm_ad.trace()/cm_ad.sum()*100
acc_nx = cm_nx.trace()/cm_nx.sum()*100
print(f"Adaptive RAG: {acc_ad:.1f}%  |  NexRoute: {acc_nx:.1f}%")
cmap_nx = LinearSegmentedColormap.from_list('nx',
    ['#F5F4FE','#CECBF6','#AFA9EC','#8B85DC','#6D64CC','#3C3489'],N=256)
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5.2),facecolor='white')
fig.suptitle('Routing Confusion Matrices: Adaptive RAG vs. NexRoute 7B',
             fontsize=13,fontweight='bold',y=1.02)
def draw(ax,pct,cnt,title,sub,cmap,thr=45):
    im=ax.imshow(pct,cmap=cmap,vmin=0,vmax=100,aspect='equal',interpolation='nearest')
    for i in range(5):
        for j in range(5):
            v=pct[i,j]; tc='white' if v>thr else '#1a1a1a'
            fw='bold' if i==j else 'normal'; fs=10 if i==j else 8.5
            ax.text(j,i,f'{v:.0f}%',ha='center',va='center',fontsize=fs,color=tc,fontweight=fw)
            if i==j: ax.text(j,i+0.32,f'({cnt[i,j]})',ha='center',va='center',fontsize=6.5,color=tc,alpha=0.85)
    for i in range(6): ax.axhline(i-0.5,color='white',lw=1.2); ax.axvline(i-0.5,color='white',lw=1.2)
    for i in range(5):
        ax.add_patch(plt.Rectangle((i-.5,i-.5),1,1,lw=2,edgecolor='white',facecolor='none',zorder=3))
    ax.set_xticks(range(5)); ax.set_xticklabels(LABELS,fontsize=9)
    ax.set_yticks(range(5)); ax.set_yticklabels(LABELS,fontsize=9)
    ax.set_xlabel('Predicted Domain',fontsize=10,labelpad=5)
    ax.set_ylabel('True Domain',fontsize=10,labelpad=5)
    ax.tick_params(length=0); ax.spines[:].set_visible(False)
    ax.set_title(title,fontsize=12,fontweight='bold',pad=6)
    ax.text(0.5,1.03,sub,transform=ax.transAxes,ha='center',fontsize=9,color='#555',style='italic')
    plt.colorbar(im,ax=ax,label='Routing Accuracy (%)',shrink=0.85)
draw(ax1,cm_ad_p,cm_ad,'Adaptive RAG',f'Total accuracy: {acc_ad:.1f}%',plt.cm.Blues)
draw(ax2,cm_nx_p,cm_nx,'NexRoute 7B (Ours)',f'Total accuracy: {acc_nx:.1f}%',cmap_nx)
plt.tight_layout()
plt.savefig('/home/sclab/moe_model/fig3_confusion_matrix_real.png',dpi=200,bbox_inches='tight',facecolor='white')
plt.savefig('/home/sclab/moe_model/fig3_confusion_matrix_real.pdf',bbox_inches='tight',facecolor='white')
print('저장 완료')
