# STEP 7B — Cover Letter

## 사용 지침
- 신규 투고로 서술 (KBS에 처음 제출)
- KBS scope: "knowledge representation, expert systems, intelligent systems" 강조
- Novelty 3줄 요약 포함 (첫 문단에)
- LoRA-based expert knowledge + knowledge-based system design 각도 강조

---

## Cover Letter — KBS 버전 (완성본)

Dear Editor-in-Chief,

We submit for consideration the manuscript titled **"Why Better Routing Does Not Guarantee Better Generation: NexRoute for Structural Resource Binding in Cross-Domain Knowledge-Based LLMs"** for publication in *Knowledge-Based Systems*.

**The central contribution** of this work is a reframing of the routing design objective for cross-domain knowledge-based LLM systems. We show that optimizing routing accuracy toward its oracle upper bound *decreases* generation quality by 1.72 ROUGE-L points (p < 0.001), while LoRA-library routing methods with routing entropy comparable to NexRoute lag by ≥18 ROUGE-L points on cross-domain queries. These two results establish that **structural resource binding**---coupling each routing decision to a domain-exclusive knowledge source, a LoRA expert, and a reasoning-format scaffold---matters more than routing probability optimization, a principle absent from all existing MoE and RAG-MoE frameworks.

**Why this work fits Knowledge-Based Systems.** The paper directly addresses the design of knowledge-based LLM systems for professional domains: NexRoute treats query routing not as a label-prediction problem, but as a *knowledge access and expert activation policy*. Each routing decision jointly selects (i) which domain knowledge partition to retrieve from, (ii) which LoRA-specialized expert to activate, and (iii) which function-conditioned reasoning scaffold to apply---all in one synchronized forward pass. This three-way resource binding is the core knowledge-based system contribution. The system operates within 14.8 GB of GPU memory, enabling single-GPU on-premise deployment in data-restricted legal, medical, or regulatory environments. Controlled ablation, oracle routing experiments, and out-of-distribution validation on MedQuAD (n=200; 90.5% routing accuracy vs. 40.0% for TF-IDF) confirm that structural resource binding---not raw parameter scale---drives expert system quality.

**Technical highlights:**

1. **(Problem)** We formally define two failure modes in single-axis routing systems---functional mismatch and cross-domain evidence contamination---and provide the first controlled evidence that routing accuracy is not the productive design objective.

2. **(Method)** A jointly trained DeBERTa-large encoder produces a domain decision and a function decision in one forward pass (the nexus point), simultaneously binding each query to a domain-exclusive retrieval partition, a LoRA-specialized adapter, and a reasoning-aligned prompt scaffold. Cross-domain queries are detected without a fifth output class via soft threshold over the top-two domain activations.

3. **(Evidence)** On a 300-sample benchmark fully independent of LoRA adapter training, NexRoute 7B exceeds all 7B baselines by ≥6.06 pp CROSS-domain ROUGE-L (Holm-corrected p < 0.001) and attains the lowest CROSS-domain NLI contradiction rate (3.6%), while operating within a single-GPU deployment envelope.

This manuscript has not been previously published and is not under consideration elsewhere. All authors have approved the submission.

We look forward to your consideration.

Sincerely,

[Author names withheld for double-blind review]

---

## Novelty 3줄 요약 (커버레터 첨부 또는 에디터 메모용)

1. **Counter-intuitive finding**: Oracle-domain routing (accuracy upper bound) degrades generation quality, demonstrating that routing accuracy optimization is the wrong design objective for cross-domain expert systems.

2. **Structural wiring principle**: Binding routing decisions simultaneously to retrieval scope and prompt scaffold format---not optimizing routing probability---is the productive design choice, confirmed by three converging lines of evidence (oracle routing, LoRA-library comparison, ablation).

3. **Practical deployment**: A 7B model with domain-partitioned retrieval and function-aware scaffolding exceeds all 7B baselines on cross-domain quality and operates within a single-GPU memory envelope (14.8 GB), enabling on-premise deployment in data-restricted professional environments.
