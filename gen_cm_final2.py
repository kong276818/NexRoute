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

# NexRoute CM (실측)
cm_nx = np.zeros((5,5), dtype=int)
for e in nx:
    q = e.get('query','').strip()
    gt_item = text_map.get(q)
    if not gt_item: continue
    t = gt_item.get('expected_domain','')
    p = 'CROSS' if e.get('is_cross',False) else (e.get('domain_pred',[''])[0] if e.get('domain_pred') else '')
    if t in idx and p in idx: cm_nx[idx[t],idx[p]] += 1

# Adaptive RAG - 확장 키워드 버전
KEYWORDS = {
    'LAW': [
        'law','legal','court','statute','plaintiff','defendant','contract',
        'judge','attorney','liability','crime','rights','judicial','verdict',
        'legislation','constitutional','tort','jurisdiction','appeal','holding',
        'doctrine','precedent','lawsuit','trial','criminal','civil','regulation',
        'penalty','prosecution','counsel','testimony','conviction','settlement',
        'negligence','fraud','copyright','patent','trademark','clause','breach',
        'damages','injunction','custody','probate','felony','misdemeanor',
    ],
    'MED': [
        'medical','patient','disease','treatment','clinical','diagnosis',
        'symptom','drug','doctor','health','therapy','surgery','medicine',
        'hospital','physician','prognosis','pathology','anatomy','infection',
        'cancer','tumor','vaccine','antibiotic','virus','bacteria','chronic',
        'acute','syndrome','disorder','prescription','dosage','pharmaceutical',
        'cardiovascular','neurological','psychiatric','genetic','immune','blood',
        'tissue','organ','nerve','bone','muscle','pain','fever','inflammation',
    ],
    'SOC': [
        'social','society','culture','economic','political','community',
        'psychology','sociology','behavior','cultural','anthropology',
        'demographic','inequality','gender','race','ideology','governance',
        'policy','welfare','education','poverty','democracy','capitalism',
        'religion','ethnic','migration','urbanization','globalization',
        'historical','philosophy','moral','ethics','class','institution',
        'population','labor','market','trade','power','identity','norm',
    ],
    'STEM': [
        'science','physics','chemical','biology','math','engineering',
        'technology','equation','energy','cell','quantum','algorithm',
        'scientific','molecule','experiment','hypothesis','evolution',
        'genetics','thermodynamics','calculus','programming','computer',
        'reaction','atomic','electromagnetic','protein','dna','ecosystem',
        'chemistry','mathematics','statistics','probability','algebra',
        'element','compound','organism','species','force','velocity',
        'acceleration','mass','charge','field','wave','radiation','neural',
    ],
}

def keyword_route(query):
    q = query.lower()
    scores = {d: sum(1 for k in kws if k in q) for d,kws in KEYWORDS.items()}
    best = max(scores.values())
    if best == 0:
        return 'SOC'  # 매칭 없으면 가장 일반적인 SOC
    return max(scores, key=scores.get)

cm_ad = np.zeros((5,5), dtype=int)
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
print("\nAdaptive RAG 도메인별:")
for i,l in enumerate(LABELS):
    print(f"  {l}: {cm_ad[i,i]}/{cm_ad[i].sum()} = {cm_ad_p[i,i]:.0f}%  {list(cm_ad[i])}")

# 시각화
cmap_nx = LinearSegmentedColormap.from_list('nx',
    ['#F5F4FE','#CECBF6','#AFA9EC','#8B85DC','#6D64CC','#3C3489'],N=256)
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,5.8),facecolor='white')
fig.suptitle('Domain Routing Confusion Matrices: Adaptive RAG vs. NexRoute 7B',
             fontsize=14,fontweight='bold',y=1.01)

def draw(ax,pct,cnt,title,acc,cmap,thr=45):
    im = ax.imshow(pct,cmap=cmap,vmin=0,vmax=100,aspect='equal',interpolation='nearest')
    for i in range(5):
        for j in range(5):
            v=pct[i,j]; tc='white' if v>thr else '#222'
            fw='bold' if i==j else 'normal'; fs=12 if i==j else 9.5
            ax.text(j,i-0.1,f'{v:.0f}%',ha='center',va='center',fontsize=fs,color=tc,fontweight=fw)
            if i==j: ax.text(j,i+0.30,f'n={cnt[i,j]}',ha='center',va='center',fontsize=6.5,color=tc,alpha=0.8)
    for i in range(6): ax.axhline(i-.5,color='white',lw=1.5); ax.axvline(i-.5,color='white',lw=1.5)
    for i in range(5):
        ax.add_patch(plt.Rectangle((i-.5,i-.5),1,1,lw=2.2,edgecolor='white',facecolor='none',zorder=3))
    ax.set_xticks(range(5)); ax.set_xticklabels(LABELS,fontsize=10)
    ax.set_yticks(range(5)); ax.set_yticklabels(LABELS,fontsize=10)
    ax.set_xlabel('Predicted Domain',fontsize=11,labelpad=6)
    ax.set_ylabel('True Domain',fontsize=11,labelpad=6)
    ax.tick_params(length=0); ax.spines[:].set_visible(False)
    ax.set_title(f'{title}\n',fontsize=12.5,fontweight='bold',pad=4)
    ax.text(0.5,1.02,f'Overall accuracy: {acc:.1f}%',transform=ax.transAxes,
            ha='center',fontsize=10,color='#555',style='italic')
    return im

im1 = draw(ax1,cm_ad_p,cm_ad,'Adaptive RAG',acc_ad,plt.cm.Blues)
im2 = draw(ax2,cm_nx_p,cm_nx,'NexRoute 7B (Ours)',acc_nx,cmap_nx)
for im,ax in [(im1,ax1),(im2,ax2)]:
    cb = plt.colorbar(im,ax=ax,shrink=0.80,pad=0.03)
    cb.set_label('Routing Accuracy (%)',fontsize=9)

plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig('/home/sclab/moe_model/fig3_cm_final2.png',dpi=200,bbox_inches='tight',facecolor='white')
plt.savefig('/home/sclab/moe_model/fig3_cm_final2.pdf',bbox_inches='tight',facecolor='white')
print('저장 완료')
