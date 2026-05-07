# NexRoute
NexRoute routes each query through two axes — domain (MED/LAW/STEM/SOC) and function type (WHAT/HOW/WHY) — to adaptively select LoRA-based experts in cross-domain large language models.

# NexRoute 🧭
### A Dual-Axis Nexus Routing Framework for Adaptive Expert Selection in Cross-Domain Large Language Models

> **Submitted to Expert Systems with Applications (ESWA)**

---

## Overview

**NexRoute** routes each query through two independent axes to activate the most suitable LoRA-based expert:

- **Domain Axis** — MED / LAW / STEM / SOC
- **Function Axis** — WHAT / HOW / WHY / COMPARE

Unlike conventional single-axis MoE routing, NexRoute identifies *both* the domain and the functional intent of a query, enabling precise expert selection in cross-domain settings.

---

## Architecture

Query Input
↓
┌─────────────────────────┐
│   Dual-Axis Router      │
│  Domain Axis  │  Func Axis  │
└─────────────────────────┘
↓
Expert Selection (LoRA Modules)
↓
Response

---

## Datasets

| Domain | Dataset | Samples |
|--------|---------|---------|
| MED | PubMedQA | ~10,000 |
| LAW | LegalBench | ~10,000 |
| STEM | SciQ | ~10,000 |
| SOC | CommonsenseQA | ~10,000 |
| CROSS | Custom benchmark | ~1,200 |

---

## Base Model

- **Qwen2.5** (via HuggingFace)
- Fine-tuned with **LoRA** (rank=16, alpha=32)

---

## Installation

```bash
git clone https://github.com/kong276818/NexRoute.git
cd NexRoute
pip install -r requirements.txt
```

---

## Usage

```bash
# Data preprocessing
python data/preprocess.py

# Training
python train.py --domain MED --func_type WHY

# Evaluation
python evaluate.py --benchmark medqa
```

---

## Results

> 🔄 Experiments in progress. Results will be updated upon completion.

---

## Citation

```bibtex
@article{kong2025nexroute,
  title={NexRoute: A Dual-Axis Nexus Routing Framework for Adaptive Expert Selection in Cross-Domain Large Language Models},
  author={Kong, Junyoung and Ha, Youngguk},
  journal={Expert Systems with Applications},
  year={2025}
}
```

---

## License

MIT License © 2025 Junyoung Kong, Konkuk University
