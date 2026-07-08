import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5)
ax.axis('off')

BLUE_D = '#185FA5'
TEAL   = '#2E86AB'
AMBER  = '#BA7517'
RED_D  = '#A32D2D'
GRAY   = '#888780'

# 쿼리 텍스트
ax.text(7, 4.5,
    'User Query ($q$): "A patient suffered from severe hepatotoxicity after being prescribed Drug X for weight loss.\n'
    'Why might the physician be held legally liable despite the signed consent form?"',
    ha='center', va='center', fontsize=9, style='italic', color=BLUE_D,
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#EEF4FC',
              edgecolor=BLUE_D, linewidth=0.8, alpha=0.9))

# 연결선
ax.plot([1.2, 12.8], [2.5, 2.5], color=GRAY, linewidth=1.5, zorder=1)

phases = [
    {
        'x': 1.5,
        'title': 'Phase 1',
        'subtitle': 'Dual-Axis Routing',
        'lines': [
            r'$d^* =$ CROSS (MED+LAW)',
            r'$f^* =$ Why',
            r'isCross$(q)$ = True',
        ],
        'color': '#D6E8F7',
        'border': BLUE_D,
    },
    {
        'x': 5.0,
        'title': 'Phase 2',
        'subtitle': 'Partitioned Retrieval',
        'lines': [
            r'$\mathcal{C}_\mathrm{cross} = \mathcal{C}_\mathrm{MED} \cup \mathcal{C}_\mathrm{LAW}$',
            'Budget proportional',
            r'to $s_1, s_2$',
        ],
        'color': '#D6EEE8',
        'border': TEAL,
    },
    {
        'x': 8.5,
        'title': 'Phase 3',
        'subtitle': 'Prompt Scaffolding',
        'lines': [
            r'Scaffold $\mathcal{S}_\mathrm{Why}$',
            'Causal chain directive',
            'IKA resolved',
        ],
        'color': '#FFF3DC',
        'border': AMBER,
    },
    {
        'x': 12.0,
        'title': 'Phase 4',
        'subtitle': 'Expert Generation',
        'lines': [
            r'Activate $e_\mathrm{CROSS}$',
            'LoRA adapter loaded',
            r'Response $\mathbf{H}$',
        ],
        'color': '#F7E8E8',
        'border': RED_D,
    },
]

box_w = 2.8
box_h = 2.2

for i, p in enumerate(phases):
    cx = p['x']

    # 타임라인 점
    circle = plt.Circle((cx, 2.5), 0.12, color=p['border'], zorder=3)
    ax.add_patch(circle)

    # 박스
    box = FancyBboxPatch(
        (cx - box_w/2, 0.2), box_w, box_h,
        boxstyle='round,pad=0.1',
        facecolor=p['color'], edgecolor=p['border'],
        linewidth=1.2, zorder=2
    )
    ax.add_patch(box)

    # 제목
    ax.text(cx, 0.2 + box_h - 0.22, p['title'],
            ha='center', va='top', fontsize=10,
            fontweight='700', color=p['border'])
    ax.text(cx, 0.2 + box_h - 0.50, p['subtitle'],
            ha='center', va='top', fontsize=8.5,
            fontweight='500', color=p['border'])

    # 구분선
    ax.plot([cx - box_w/2 + 0.15, cx + box_w/2 - 0.15],
            [0.2 + box_h - 0.65, 0.2 + box_h - 0.65],
            color=p['border'], linewidth=0.6, alpha=0.5)

    # 내용
    for j, line in enumerate(p['lines']):
        ax.text(cx, 0.2 + box_h - 0.85 - j*0.38,
                line, ha='center', va='top',
                fontsize=8.5, color='#333333')

    # 화살표
    if i < len(phases) - 1:
        ax.annotate('', xy=(cx + box_w/2 + 0.22, 1.3),
                    xytext=(cx + box_w/2 + 0.02, 1.3),
                    arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.2))

plt.savefig('/home/sclab/moe_model/img8.png', dpi=300,
            bbox_inches='tight', facecolor='white')
print("저장 완료: img8.png")
plt.close()