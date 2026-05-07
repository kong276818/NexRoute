# NexRoute
NexRoute routes each query through two axes — domain (MED/LAW/STEM/SOC) and function type (WHAT/HOW/WHY) — to adaptively select LoRA-based experts in cross-domain large language models.

# NexRoute 🧭

> **NexRoute: A Dual-Axis Nexus Routing Framework for Adaptive Expert Selection in Cross-Domain Large Language Models**

[![Paper](https://img.shields.io/badge/Paper-ESWA%202025-blue)](https://www.sciencedirect.com/journal/expert-systems-with-applications)
[![Model](https://img.shields.io/badge/Base%20Model-Qwen2.5-orange)](https://huggingface.co/Qwen)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Under%20Review-yellow)]()

---

## 📌 Overview

**NexRoute** is a dual-axis routing framework for Mixture-of-Experts (MoE) based Large Language Models.  
Unlike conventional single-axis routing methods that only consider domain categories, NexRoute routes each query through **two independent axes simultaneously**:

| Axis | Categories | Role |
|------|-----------|------|
| **Domain Axis** | MED / LAW / STEM / SOC | *What field does this query belong to?* |
| **Function Axis** | WHAT / HOW / WHY / COMPARE | *What does the user intend to ask?* |

By decomposing queries along both axes, NexRoute identifies the **nexus point** — the intersection of domain knowledge and functional intent — to activate the most appropriate LoRA-based expert module.

---

## 🏗️ Architecture

Query Input
                         │
          ┌──────────────▼──────────────┐
          │       Dual-Axis Router       │
          │                             │
          │  ┌──────────┐ ┌──────────┐  │
          │  │  Domain  │ │Function  │  │
          │  │  Router  │ │  Router  │  │
          │  └────┬─────┘ └────┬─────┘  │
          │       │  Nexus     │        │
          │       └─────┬──────┘        │
          └─────────────▼───────────────┘
                        │
      ┌─────────────────▼─────────────────┐
      │         Expert Selection           │
      │                                   │
      │  [MED-LoRA] [LAW-LoRA] [STEM-LoRA]│
      │  [SOC-LoRA] [CROSS-LoRA]          │
      └─────────────────┬─────────────────┘
                        │
                   Final Response


### Key Components

- **Domain Router**: Classifies queries into domain categories using a lightweight classifier head on top of the frozen base model
- **Function Router**: Identifies the functional intent (WHAT/HOW/WHY/COMPARE) of the query independently
- **Nexus Gate**: Combines both routing signals to determine the final expert activation weights
- **LoRA Experts**: Domain-specialized low-rank adaptation modules (rank=16, alpha=32) trained independently per domain

---

## 📂 Repository Structure
NexRoute/
├── data/
│   ├── preprocess.py          # Dataset preprocessing & func_type labeling
│   ├── dataset_loader.py      # HuggingFace dataset loaders
│   └── processed/             # Preprocessed jsonl files (per domain)
│
├── model/
│   ├── router.py              # Dual-axis router implementation
│   ├── nexus_gate.py          # Nexus gating mechanism
│   └── lora_expert.py         # LoRA expert module
│
├── train.py                   # Training script
├── evaluate.py                # Evaluation script
├── inference.py               # Single query inference
├── requirements.txt
└── README.md

---

## 📊 Datasets

### Training Data

| Domain | Dataset | HuggingFace ID | Samples |
|--------|---------|----------------|---------|
| MED | PubMedQA | `qiaojin/PubMedQA` | ~10,000 |
| LAW | LegalBench | `nguyen-brat/LegalBench` | ~10,000 |
| STEM | SciQ | `allenai/sciq` | ~10,000 |
| SOC | CommonsenseQA | `commonsense_qa` | ~10,000 |
| CROSS | Custom Benchmark | — | ~1,200 |

### Function Type Distribution

Each sample is automatically labeled with a functional type based on query pattern matching:

| Func Type | Trigger Keywords | Example |
|-----------|-----------------|---------|
| `WHAT` | what is, define, meaning of | *"What is malpractice?"* |
| `HOW` | how to, how does, steps to | *"How does LoRA reduce parameters?"* |
| `WHY` | why, reason, cause of | *"Why does informed consent matter?"* |
| `COMPARE` | compare, difference, versus | *"Compare civil vs criminal liability"* |

---

## ⚙️ Installation

```bash
git clone https://github.com/kong276818/NexRoute.git
cd NexRoute
pip install -r requirements.txt
```

**Requirements**
- Python 3.10+
- PyTorch 2.0+
- transformers >= 4.40
- peft >= 0.10
- datasets

---

## 🚀 Usage

### 1. Data Preprocessing

```bash
python data/preprocess.py \
  --domains MED LAW STEM SOC \
  --output_dir data/processed/ \
  --min_output_words 50
```

### 2. Training

```bash
# Train domain-specific LoRA expert
python train.py \
  --domain MED \
  --base_model Qwen/Qwen2.5-7B \
  --lora_rank 16 \
  --lora_alpha 32 \
  --epochs 3 \
  --batch_size 4

# Train router
python train.py \
  --mode router \
  --router_type dual \
  --data_dir data/processed/
```

### 3. Evaluation

```bash
# Benchmark evaluation
python evaluate.py \
  --benchmark medqa \
  --checkpoint checkpoints/nexroute_v1

# Cross-domain evaluation
python evaluate.py \
  --benchmark cross_domain \
  --checkpoint checkpoints/nexroute_v1
```

### 4. Inference

```bash
python inference.py \
  --query "What are the legal obligations of a physician in malpractice cases?" \
  --checkpoint checkpoints/nexroute_v1
```

---

## 📈 Results

> 🔄 Experiments currently in progress. Results will be updated upon completion.

### Benchmark Performance (coming soon)

| Model | MedQA | MMLU-Law | SciQ | CommonsenseQA | GPQA (Cross) |
|-------|-------|----------|------|---------------|--------------|
| Qwen2.5 (Base) | -- | -- | -- | -- | -- |
| Single-LoRA | -- | -- | -- | -- | -- |
| Domain-only MoE | -- | -- | -- | -- | -- |
| **NexRoute (Ours)** | -- | -- | -- | -- | -- |

---

## 📝 Citation

If you use NexRoute in your research, please cite:

```bibtex
@article{kong2025nexroute,
  title     = {NexRoute: A Dual-Axis Nexus Routing Framework for Adaptive 
               Expert Selection in Cross-Domain Large Language Models},
  author    = {Kong, Junyoung and Ha, Youngguk},
  journal   = {Expert Systems with Applications},
  year      = {2025},
  publisher = {Elsevier}
}
```

---

## 👥 Authors

| Name | Affiliation | Role |
|------|------------|------|
| **Junyoung Kong** | Konkuk University | First Author |
| **Youngguk Ha** | Konkuk University | Corresponding Author |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

© 2026 Junyoung Kong, Konkuk University, Seoul, Republic of Korea
