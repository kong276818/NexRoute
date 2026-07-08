import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# 캔버스 설정 (가로 18, 세로 11 비율의 여유로운 공간)
fig, ax = plt.subplots(figsize=(18, 11.5))
ax.set_xlim(-0.5, 18.5)
ax.set_ylim(-3.5, 9.5)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── LaTeX 코드와 100% 일치하는 색상 팔레트 정의 ──────────────────
clrPF = "#EEEDFE"; clrPE = "#9B95E0"; clrPT = "#3C3489"
clrPH = "#E0DDFB"; clrPB = "#DDD9FD"
clrTF = "#E1F5EE"; clrTE = "#5DCAA5"; clrTT = "#085041"
clrTH = "#C8EEE1"; clrTB = "#CDEEE2"
clrGF = "#D8F2EA"; clrGE = "#1D9E75"; clrGT = "#074434"
clrGH = "#BAE9D5"; clrGB = "#C4E8D8"
clrCF = "#FAECE7"; clrCE = "#E88060"; clrCT = "#712B13"
clrCH = "#F5CEC0"; clrCB = "#F5D5C8"
clrRF = "#FBEAF0"; clrRE = "#D4537E"; clrRT = "#72243E"
clrIF = "#F2F0EB"; clrIE = "#B0ADA4"; clrIT = "#3A3935"

# ── 헬퍼 함수 ───────────────────────────────────────────────
def box(x, y, w, h, fc, ec, r=0.15, ls="solid", lw=1.5):
    ax.add_patch(FancyBboxPatch((x,y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec,
        linewidth=lw, linestyle=ls, zorder=2))

def txt(x, y, s, sz=10, col="#333", bold=False, ha="center", va="center", it=False, **kwargs):
    weight = "bold" if bold else "normal"
    style = "italic" if it else "normal"
    ax.text(x, y, s, fontsize=sz, color=col, ha=ha, va=va,
            fontweight=weight, fontstyle=style, zorder=5, **kwargs)

def arrow(x1, y1, x2, y2, col="#888", lw=1.5, dash=False, hw=0.15, hl=0.2):
    ls = (0, (5, 5)) if dash else "solid"
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
        arrowprops=dict(arrowstyle=f"->,head_width={hw},head_length={hl}",
                        color=col, lw=lw, linestyle=ls), zorder=6)

def line(x1, y1, x2, y2, col="#888", lw=1.0, dash=False):
    ls = (0, (4, 4)) if dash else "solid"
    ax.plot([x1,x2], [y1,y2], color=col, lw=lw, linestyle=ls, zorder=3)

# ════════════════════════════════════════════════════════════
# Title
txt(8.90, 8.70, "NexRoute: Dual-Axis Nexus Routing Framework", 15, "#000", bold=True)

# ════════════════════════════════════════════════════════════
# (0) INPUT
box(0, 0, 1.60, 8.0, clrIF, clrIE, r=0.15, lw=1.0)
txt(0.80, 4.20, "Input", 11, clrIT, bold=True)
txt(0.80, 3.80, r"$q$", 14, clrIT, it=True)

# ════════════════════════════════════════════════════════════
# (1) ROUTER
box(2.00, 0, 4.80, 8.0, clrPF, clrPE, r=0.15, lw=1.0)
box(2.00, 7.30, 4.80, 0.70, clrPH, "none", r=0.15)
line(2.00, 7.30, 6.80, 7.30, clrPE, lw=1.0)
txt(4.40, 7.65, "(1) Dual-Axis Router", 11, clrPT, bold=True)

line(4.40, 0.15, 4.40, 7.20, clrPE, lw=1.0, dash=True) # Center dashed line

# Domain Axis R_D
txt(3.20, 6.15, "Domain Axis $R_D$", 10, clrPT, bold=True)
txt(3.20, 5.50, r"DeBERTa-large" + "\n" + r"MLP$(1024 \rightarrow 256 \rightarrow 4)$" + "\n" + r"BCE loss, multi-label" + "\n" + r"$\tau=0.30,\ \delta=0.40$", 9, clrPT)

box(2.12, 3.55, 2.16, 1.30, clrPB, clrPE, r=0.08, lw=0.8)
txt(3.20, 4.20, r"$d^* \in \mathcal{D}$" + "\n" + r"$= \{\mathrm{LAW}, \mathrm{MED},$" + "\n" + r"$\mathrm{SOC}, \mathrm{STEM}, \mathrm{CROSS}\}$", 9, clrPT)

box(2.12, 2.70, 2.16, 0.75, "#D5D0FC", clrPE, r=0.08, lw=0.8)
txt(3.20, 3.075, r"$s_2 \geq \tau \ \wedge\ s_1-s_2 \leq \delta$", 9, clrPT, it=True)

# Function Axis R_F
txt(5.60, 6.15, "Function Axis $R_F$", 10, clrPT, bold=True)
txt(5.60, 5.50, "DeBERTa-large (shared)\n4-way softmax head\ncross-entropy loss", 9, clrPT)

box(4.52, 3.55, 2.16, 1.30, clrPB, clrPE, r=0.08, lw=0.8)
txt(5.60, 4.20, r"$f^* \in \mathcal{F}$" + "\n" + r"$= \{\mathrm{What}, \mathrm{How},$" + "\n" + r"$\mathrm{Why}, \mathrm{Compare}\}$", 9, clrPT)

# Function tiles
box(4.54, 0.82, 0.97, 0.48, "#F2F0FE", clrPE, r=0.05, lw=0.5); txt(5.025, 1.06, "What", 8, clrPT, bold=True)
box(5.61, 0.82, 0.97, 0.48, "#F2F0FE", clrPE, r=0.05, lw=0.5); txt(6.095, 1.06, "How", 8, clrPT, bold=True)
box(4.54, 0.28, 0.97, 0.48, "#F2F0FE", clrPE, r=0.05, lw=0.5); txt(5.025, 0.52, "Why", 8, clrPT, bold=True)
box(5.61, 0.28, 0.97, 0.48, "#F2F0FE", clrPE, r=0.05, lw=0.5); txt(6.095, 0.52, "Comp.", 8, clrPT, bold=True)

txt(4.40, 1.72, r"Nexus point $(d^*, f^*)$", 10, clrPT, it=True)

# ════════════════════════════════════════════════════════════
# (2) RETRIEVAL
box(7.20, 0, 3.10, 8.0, clrTF, clrTE, r=0.15, lw=1.0)
box(7.20, 7.30, 3.10, 0.70, clrTH, "none", r=0.15)
line(7.20, 7.30, 10.30, 7.30, clrTE, lw=1.0)
txt(8.75, 7.65, "(2) Partitioned Retrieval", 11, clrTT, bold=True)

txt(8.75, 6.20, "multilingual-E5-large\n\nFAISS IndexFlatIP\n\n" + r"$K=10,\ \ell_2$-norm", 10, clrTT)

box(7.36, 0.20, 2.78, 4.60, clrTB, clrTE, r=0.08, lw=0.8)
txt(8.75, 4.20, r"$\mathrm{IDI}(q) = 0$", 14, clrTT, bold=True)
txt(8.75, 3.70, "guaranteed by", 10, clrTT, it=True)
txt(8.75, 3.10, r"$\mathcal{K}_i \cap \mathcal{K}_j = \emptyset$" + "\n" + r"$\forall i \neq j \in \mathcal{D}$", 11, clrTT)
txt(8.75, 2.00, "CROSS budget:", 10, clrTT, bold=True)
txt(8.75, 1.10, r"$K_j = \left\lfloor \frac{K \cdot s_j}{s_1+s_2} \right\rfloor$", 13, clrTT)

# ════════════════════════════════════════════════════════════
# (3) SCAFFOLDING
box(10.70, 0, 3.50, 8.0, clrGF, clrGE, r=0.15, lw=1.0)
box(10.70, 7.30, 3.50, 0.70, clrGH, "none", r=0.15)
line(10.70, 7.30, 14.20, 7.30, clrGE, lw=1.0)
txt(12.45, 7.65, "(3) Prompt Scaffolding", 11, clrGT, bold=True)

box(10.86, 5.50, 3.18, 1.60, clrGB, clrGE, r=0.08, lw=0.8)
txt(12.45, 6.30, r"$\mathcal{P}(q, d^*, f^*) = $" + "\n\n" + r"$[\mathrm{sys}_{d^*},\ \mathcal{S}_{f^*}(\mathcal{C}_{d^*}),\ q]$", 10, clrGT)

box(10.86, 0.20, 3.18, 5.10, "#CDECDE", clrGE, r=0.08, lw=0.8)
txt(11.10, 4.80, r"Scaffolds $\mathcal{S}_{f^*}$:", 11, clrGT, bold=True, ha="left")
txt(11.10, 4.00, r"$\mathbf{What} \rightarrow$ identify key facts", 10, clrGT, ha="left")
txt(11.10, 3.30, r"$\mathbf{How} \rightarrow$ step-by-step flow", 10, clrGT, ha="left")
txt(11.10, 2.60, r"$\mathbf{Why} \rightarrow$ causal chain", 10, clrGT, ha="left")
txt(11.10, 1.90, r"$\mathbf{Compare} \rightarrow$ contrastive", 10, clrGT, ha="left")
txt(12.45, 0.80, r"IKA resolved via $\mathcal{S}_{f^*}$", 10, clrGT, it=True)

# ════════════════════════════════════════════════════════════
# (4) EXPERT
box(14.60, 0, 3.20, 8.0, clrCF, clrCE, r=0.15, lw=1.0)
box(14.60, 7.30, 3.20, 0.70, clrCH, "none", r=0.15)
line(14.60, 7.30, 17.80, 7.30, clrCE, lw=1.0)
txt(16.20, 7.65, "(4) Expert Generation", 11, clrCT, bold=True)

txt(16.20, 6.15, r"Qwen2.5-7B (NF4)" + "\n\n" + r"$r=16,\ \alpha=32$, 5 modules" + "\n\n" + "PEFT adapter switch", 10, clrCT)

box(14.76, 3.30, 2.88, 1.60, clrCB, clrCE, r=0.08, lw=0.8)
txt(16.20, 4.10, r"$\mathbf{H} = \mathrm{LLM}_{\theta+\Delta\theta_{d^*}}$" + "\n\n" + r"$(\mathcal{P}(q,d^*,f^*))$", 10, clrCT)

box(14.80, 0.20, 2.80, 1.80, "#F7E1D8", clrCE, r=0.08, lw=0.8)
txt(16.20, 1.35, r"Response $\mathbf{H}$", 11, clrCT, bold=True)
txt(16.20, 0.60, r"14.8 GB VRAM, $3{\times} <$ 72B", 9, clrCT)

# ════════════════════════════════════════════════════════════
# ARROWS
arrow(1.65, 4.0, 1.95, 4.0, col="#999")
arrow(6.85, 4.0, 7.15, 4.0, col="#999")
arrow(10.35, 4.0, 10.65, 4.0, col="#999")
arrow(14.25, 4.0, 14.55, 4.0, col="#999")

# ════════════════════════════════════════════════════════════
# CROSS BOX (Bottom)
arrow(4.40, 0.0, 4.40, -0.48, col=clrRE, dash=True)
arrow(16.20, -1.62, 16.20, 0.0, col=clrRE, dash=True)

box(2.00, -1.62, 15.80, 1.10, clrRF, clrRE, r=0.10, ls=(0,(5,4)), lw=1.2)

txt(3.00, -1.07, "CROSS Detection & Activation", 11, clrRT, bold=True, ha="left")
txt(7.30, -1.07, r"$[\mathrm{isCross}(q)=\mathrm{True}]$", 11, clrRT, ha="left")
txt(12.00, -1.07, r"Activate: $e_{\mathrm{CROSS}}$", 11, clrRT, bold=True, ha="left")

txt(3.00, -1.35, r"Cond: $s_2 \geq \tau \ \wedge\ s_1-s_2 \leq \delta$", 10, clrRT, ha="left")
txt(7.30, -1.35, r"Context: $\mathcal{C}_{\mathrm{cross}} = \mathcal{C}_{d_1} \cup \mathcal{C}_{d_2}$", 10, clrRT, ha="left")


# ════════════════════════════════════════════════════════════
# FOOTER STATS
line(0, -2.00, 17.80, -2.00, "#E0E0E0", lw=1.5)

txt(0.80, -2.50, "$15,744$ samples\nMMLU subsets", 10, clrIT)
txt(4.40, -2.50, r"$87.4\%$ routing acc." + "\n" + r"$+2.2\%$p vs baseline", 10, clrPT)
txt(8.75, -2.50, "VRAM $14.8$ GB\n$3{\\times} <$ 72B", 10, clrTT)
txt(12.45, -2.50, "BERTScore $89.53$\nCROSS domain #1", 10, clrGT)
txt(16.20, -2.50, r"Halluc. $12.4\%$" + "\n" + r"Win rate $70.0\%$", 10, clrCT)

# 파일 저장
plt.tight_layout(pad=0.1)
plt.savefig("nexroute_full.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig("nexroute_full.pdf", bbox_inches="tight", facecolor="white")
print("✅ 고해상도 전체 프레임워크 저장 완료 (nexroute_full.png / nexroute_full.pdf)")