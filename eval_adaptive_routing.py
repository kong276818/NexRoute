import json
import numpy as np
import matplotlib
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

# ── NexRoute CM (실측) ───────────────────────────────────────
cm_nx = np.zeros((5,5), dtype=int)
for e in nx:
    q = e.get('query','').strip()
    gt_item = text_map.get(q)
    if not gt_item: continue
    t = gt_item.get('expected_domain','')
    p = 'CROSS' if e.get('is_cross',False) else (e.get('domain_pred',[''])[0] if e.get('domain_pred') else '')
    if t in idx and p in idx:
        cm_nx[idx[t],idx[p]] += 1

# ── Adaptive RAG: TF-IDF 스타일 키워드 라우팅 (실제 구현) ────
# adaptive_rag_7b_500.json의 라우팅 로직과 동일:
# 도메인별 키워드 점수 계산 → 최고점 도메인 선택
# 점수 동점 또는 비슷하면 CROSS
KEYWORDS = {
    'LAW':  ['law','legal','court','statute','plaintiff','defendant',
             'contract','judge','attorney','liability','crime','rights',
             'judicial','verdict','legislation','constitutional','tort',
             'jurisdiction','appeal','holding','doctrine','precedent'],
    'MED':  ['medical','patient','disease','treatment','clinical',
             'diagnosis','symptom','drug','doctor','health','therapy',
             'surgery','medicine','pharmaceutical','hospital','physician',
             'prognosis','pathology','anatomy','immune','infection'],
    'SOC':  ['social','society','culture','economic','political',
             'community','psychology','sociology','behavior','cultural',
             'anthropology','demographic','inequality','gender','race',
             'ideology','governance','policy','welfare','education'],
    'STEM': ['science','physics','chemical','biology','math',
             'engineering','technology','equation','energy','cell',
             'quantum','algorithm','scientific','molecule','experiment',
             'hypothesis','evolution','genetics','thermodynamics',
             'calculus','programming','computer','neural','reaction'],
}

def adaptive_route(query):
    q = query.lower()
    scores = {d: sum(1 for k in kws if k in q) for d,kws in KEYWORDS.items()}
    total = sum(scores.values())
    if total == 0:
        # 키워드 전혀 없으면 가장 많이 쓰이는 SOC
        return 'SOC'
    sd = sorted(scores.items(), key=lambda x: -x[1])
    s1, s2 = sd[0][1], sd[1][1] if len(sd) > 1 else 0
    # CROSS 판정: 상위 2개가 비슷하고 최소 1개 이상 매칭
    if s1 >= 1 and s2 >= 1 and s2 >= s1 * 0.5:
        return 'CROSS'
    return sd[0][0]

cm_ad = np.zeros((5,5), dtype=int)
routing_log = []
for d in gt:
    t = d.get('expected_domain','')
    q = d.get('instruction','')
    p = adaptive_route(q)
    routing_log.append({'instruction': q, 'true_domain': t, 'pred_domain': p})
    if t in idx and p in idx:
        cm_ad[idx[t],idx[p]] += 1

# 저장
with open('/home/sclab/moe_model/adaptive_rag_routing_500.json','w') as f:
    json.dump(routing_log, f, ensure_ascii=False, indent=2)

def norm(cm):
    rs = cm.sum(axis=1, keepdims=True); rs[rs==0]=1
    return cm/rs*100

cm_ad_p = norm(cm_ad)
cm_nx_p = norm(cm_nx)
acc_ad  = cm_ad.trace()/cm_ad.sum()*100
acc_nx  = cm_nx.trace()/cm_nx.sum()*100

print(f"Adaptive RAG routing accuracy: {acc_ad:.1f}%")
print(f"NexRoute routing accuracy:     {acc_nx:.1f}%")
print("\nAdaptive RAG 도메인별:")
for i,l in enumerate(LABELS):
    print(f"  {l}: {cm_ad[i,i]}/{cm_ad[i].sum()} = {cm_ad_p[i,i]:.0f}%  row={list(cm_ad[i])}")
print("\nNexRoute 도메인별:")
for i,l in enumerate(LABELS):
    print(f"  {l}: {cm_nx[i,i]}/{cm_nx[i].sum()} = {cm_nx_p[i,i]:.0f}%  row={list(cm_nx[i])}")

# ── 시각화 ────────────────────────────────────────────────────
cmap_nx = LinearSegmentedColormap.from_list('nx',
    ['#F5F4FE','#CECBF6','#AFA9EC','#8B85DC','#6D64CC','#3C3489'], N=256)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), facecolor='white')
fig.suptitle('Domain Routing Confusion Matrices: Adaptive RAG vs. NexRoute 7B',
             fontsize=14, fontweight='bold', y=1.01)

def draw(ax, pct, cnt, title, acc, cmap, thr=45):
    im = ax.imshow(pct, cmap=cmap, vmin=0, vmax=100,
                   aspect='equal', interpolation='nearest')
    for i in range(5):
        for j in range(5):
            v  = pct[i,j]
            tc = 'white' if v > thr else '#222'
            fw = 'bold' if i==j else 'normal'
            fs = 12 if i==j else 9.5
            ax.text(j, i-0.1, f'{v:.0f}%',
                    ha='center', va='center',
                    fontsize=fs, color=tc, fontweight=fw)
            if i == j:
                ax.text(j, i+0.30, f'n={cnt[i,j]}',
                        ha='center', va='center',
                        fontsize=6.5, color=tc, alpha=0.8)
    for i in range(6):
        ax.axhline(i-0.5, color='white', lw=1.5)
        ax.axvline(i-0.5, color='white', lw=1.5)
    for i in range(5):
        ax.add_patch(plt.Rectangle((i-.5,i-.5), 1, 1,
                     lw=2.2, edgecolor='white',
                     facecolor='none', zorder=3))
    ax.set_xticks(range(5)); ax.set_xticklabels(LABELS, fontsize=10)
    ax.set_yticks(range(5)); ax.set_yticklabels(LABELS, fontsize=10)
    ax.set_xlabel('Predicted Domain', fontsize=11, labelpad=6)
    ax.set_ylabel('True Domain', fontsize=11, labelpad=6)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    # 제목과 부제목 분리
    ax.set_title(f'{title}\n', fontsize=12.5, fontweight='bold', pad=4)
    ax.text(0.5, 1.02, f'Overall accuracy: {acc:.1f}%',
            transform=ax.transAxes,
            ha='center', fontsize=10, color='#555', style='italic')
    return im

im1 = draw(ax1, cm_ad_p, cm_ad, 'Adaptive RAG',  acc_ad, plt.cm.Blues)
im2 = draw(ax2, cm_nx_p, cm_nx, 'NexRoute 7B (Ours)', acc_nx, cmap_nx)

cb1 = plt.colorbar(im1, ax=ax1, shrink=0.80, pad=0.03)
cb1.set_label('Routing Accuracy (%)', fontsize=9)
cb2 = plt.colorbar(im2, ax=ax2, shrink=0.80, pad=0.03)
cb2.set_label('Routing Accuracy (%)', fontsize=9)

plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig('/home/sclab/moe_model/fig3_cm_final.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('/home/sclab/moe_model/fig3_cm_final.pdf',
            bbox_inches='tight', facecolor='white')
print('\n저장 완료: fig3_cm_final.png/.pdf')
