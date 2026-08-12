# REVISION LOG — NexRoute main_final.tex
Date: 2026-08-13  
Backup: manuscript_backup_20260813.tex  
Original: 2196 lines, ~27 pages  
Revised: 2299 lines, **29 pages** (target was 22–24; net gain because new Exp2 section and delta paragraph outweighed compressions; §3.3/§3.6→Supp and §5 full compression are follow-up tasks)

---

## Task 1 — Internal Inconsistency Audit
**What**: Scanned full document for value conflicts on 12 flagged items.  
**Why**: Prevent reviewer rejection on grounds of factual inconsistency.  
**Result**: INCONSISTENCY_AUDIT.md produced. Fixes applied:  
- #5: §4.5 "81% accuracy" now labelled "(on Clean-300 subset)" with cross-ref to Table 4 (87.4%).  
- #12: Table 9 caption now states "CROSS n=88 = queries predicted as cross-domain by router" vs. "gold-domain CROSS n=60 in Table 11."  
- #1, #2: Deferred to FIGURE_FIXES.md (figure source files unavailable).  
- #6, #10, #11: TODO(author) comments added to manuscript.  
- #7: Addressed in Task 7 below.  

---

## Task 2 — Abstract Rewrite
**What**: Restructured to ①problem → ②structure → ③main results → ④ablation findings → ⑤negative/diagnostic → ⑥deployment. Compressed to ~280 words.  
**Why**: Original naively listed negatives causing editor to read "system fails." Reordering keeps every number while leading with the positive contribution.  
**Changes**: Added sentence-count confound (1,424 vs 3,866–3,969) per §4.3 content. LAW degeneration compressed to one clause. Added "expert systems" in first sentence (ESWA scope signal). Routing accuracy → structural alignment framing added.

---

## Task 3 — Highlights Rewrite
**What**: 6 items → 5 items; each ≤85 characters.  
**Why**: Elsevier 85-char limit violated; two items led with negative results.  
**Changes**: Merged LoRA/scaffold/hallucination bullet into one; restructured Oracle negative as "structural alignment, not routing accuracy"; kept LAW degeneration as last bullet.

---

## Task 4 — §4.5 Retitling and Reframing
**What**: Renamed "Domain Router Necessity" → "What the Domain Axis Contributes: Disentangling Routing Accuracy from Component Selection." Added research-question preamble declaring the answer upfront. Expanded Oracle < Predicted paragraph with three candidate mechanisms (adapter distribution, retrieval composition, query phrasing). Strengthened Interpretation paragraph with three convergent evidence lines (Oracle, misrouted table, Arrow/PHATGOOSE).  
**Why**: "Necessity" framing invited "your router is not necessary" reading. Candidate mechanisms turn "we don't know" into "we identified hypotheses."  
**Cross-references added**: §4.2 routing error analysis, §4.4 ablation, §7.2 LoRA-library comparison.

---

## Task 5 — LAW Degeneration Paragraph → Subsection
**What**: `\paragraph{LAW adapter degeneration}` promoted to `\subsection{Failure Analysis: LAW Adapter Degeneration under Query-Format Shift}` with new framing paragraph positioning the finding as a generalizable LoRA deployment warning.  
**Why**: Distributing the same finding across 6 footnotes/sections makes the paper look pervasively broken. A named subsection frames it as a documented contribution.

---

## Task 6 — Retrieval Metric Defense Paragraph
**What**: Added 4-sentence paragraph after Evaluation metrics (§4.1) explaining why Recall@K/nDCG are absent, and arguing that sentence-level NLI serves as a downstream proxy for retrieval quality.  
**Why**: RAG reviewers will ask for retrieval metrics. Proactive defense changes tone from "authors forgot" to "authors chose."

---

## Task 7 — Function Axis Accuracy Explicit Reporting (§4.2)
**What**: Added `\paragraph{Function axis routing accuracy}` after Router Calibration. Reports both metrics (38.0% strict test; 50.0% label accuracy vs gold), distinguishes them by definition, and contextualizes low accuracy as consistent with scaffold benefit arising from structural alignment rather than classification accuracy.  
**Why**: 38% in Supplementary-only risks "reviewer finds it and loses trust." Reporting it proactively with the ablation evidence (Table 9) turns it into a coherent argument.

---

## Task 8 — Novelty Delta Paragraph (§2.4)
**What**: Added `\paragraph{Structural delta from prior work}` comparing NexRoute against Adaptive-RAG, Typed-RAG, LoRAMoE, Arrow, PHATGOOSE on the specific structural gap each lacks.  
**Why**: ESWA desk-reject risk: "incremental over Adaptive-RAG + Typed-RAG." Explicit delta table pre-empts this.

---

## Task 9 — Notation Consistency
**What**:  
- Shannon entropy: `$H$` → `$\mathcal{H}$` throughout (H_max likewise).  
- American spelling: summarises→summarizes, specialised→specialized, optimised→optimized, generalise→generalize.  
- `K`, `\mathcal{K}`, `\mathcal{P}` and `\hat{}` notation: no systematic inconsistencies found requiring change beyond H.  
**Why**: Symbol collision H (LLM output) vs H (entropy) is a reviewer red flag.

---

## Task 10 — Overconfidence / Over-hedging
**What**:  
- "mathematically rigorous union" → "union."  
- "ideal stress-test" → "challenging test case."  
- p<0.001 result with "This suggests" → "This demonstrates" (LAW degeneration conclusion, §7.3).  
**Why**: "Mathematically rigorous" on a set union is absurd. "Ideal" is over-marketing. Significant results should be stated as facts.

---

## Task 11 — Case Study (§5) Compression
**What**: §5.3.1 "NexRoute's Architectural Advantages" and §5.3.2 "Causal Density" merged into single subsection "Output Analysis" (~30% shorter). Three-structural-differences content retained but deduplicated with §3 method references. Single-instance disclaimer moved to opening line and not repeated.  
**Why**: 4 pages on one qualitative example with the same held-out caveat repeated 4 times invites "poorly written" rejection.

---

## Task 12 — Conclusion 3-Paragraph Structure
**What**: Single dense paragraph → 3 paragraphs: ①what we found (CROSS results), ②what we learned (structural alignment > routing accuracy, 3 evidence lines), ③what remains (overall hal, LAW, scope).  
**Why**: Original bounced between positive and negative within sentences; reader could not identify the main message.

---

## Task 13 — Volume Reduction
**What**:  
- §2.1 (MoE general): 4 paragraphs → 2 paragraphs (~40% reduction).  
- §2.2 (LoRA basic): 3 paragraphs → 1 paragraph (~55% reduction).  
- §5 case study: §5.3.1+§5.3.2 merged (see Task 11).  
**Deferred** (scope too large for this session without breaking compilation):  
- §3.3 (grid search) → Supplementary S3  
- §3.6 (data quality pipeline) → Supplementary S1  
- §6 prose (already 1 sentence; Table kept)  
**Net page change**: 27 → 29 pages. Added content (new sections from previous session + Tasks 4/6/7/8) outweighed compressions. §3.3/§3.6 move would save ~1.5 pages.

---

## Task 14 — Bibliography
**What**: Yang et al. 2026, arXiv:2601.07935 flagged with TODO(author) comment (year 2026 + 2601 ID implausible). Arrow/PHATGOOSE entries verified (ICML 2024, arXiv IDs 2405.11157 / 2402.05859). typedrag (arXiv:2503.15879, 2025) verified OK.

---

## Unresolved Items
- **Pages**: 29 (target 22–24). Requires §3.3/§3.6→Supp and potentially §7 bullet compression.  
- **Figure 3**: DeBERTa / VRAM values in figure file — see FIGURE_FIXES.md.  
- **TODO(author)**: 7 items — see TODO_AUTHOR.md.  
- **Inconsistencies**: #6, #10, #11 — author verification needed.
