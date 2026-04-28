# Can Fine-Tuning Language Models on Bias Itself Introduce Bias?

A research project investigating whether parameter-efficient fine-tuning (LoRA) on bias-annotated data (StereoSet) inadvertently reshapes a model's own representational biases — evaluated on CrowS-Pairs and SST-2 using frozen sentence embeddings.

---

## 🔬 Research Question

Large language models inherit societal biases from their training data. A natural mitigation strategy is to fine-tune models on bias-annotated benchmarks so they can *detect* stereotypical text. But does this exposure itself reshape the model's internal biases?

We investigate this by comparing:
- **Base MiniLM**: Frozen `sentence-transformers/all-MiniLM-L6-v2` encoder
- **LoRA MiniLM**: Same encoder + LoRA adapters (r=8, α=32) fine-tuned on StereoSet for stereotype classification

Both models are evaluated using **frozen sentence embeddings** on two downstream tasks.

## 📊 Key Findings

| Metric | Base MiniLM | LoRA MiniLM | Interpretation |
|--------|:-----------:|:-----------:|----------------|
| **CrowS-Pairs Stereotype Score** | 0.511 | 0.422 | LoRA *over-corrects* → anti-stereotypical preference |
| **CrowS-Pairs F1** | 0.650 | 0.562 | Shifted bias detection behavior |
| **SST-2 Accuracy** | 0.808 | 0.808 | General NLU capability preserved |
| **SST-2 F1** | 0.811 | 0.808 | Negligible difference |
| **Flip Rate** | — | ~40% | 604/1,508 pairs changed preference |

### Key Takeaways

1. **Over-correction, not correction**: LoRA fine-tuning on StereoSet doesn't neutralize bias (SS→0.50) — it inverts it (SS: 0.511→0.422), introducing a mirror-image anti-stereotypical preference.
2. **Category heterogeneity**: Age (−0.276) and religion (−0.228) shift most dramatically; disability is unchanged; socioeconomic bias actually *increases* (+0.093).
3. **Preserved NLU**: SST-2 sentiment accuracy is identical (80.8%), confirming bias-related changes are orthogonal to general task features.

---

## 🏗️ Architecture

```
sentence-transformers/all-MiniLM-L6-v2  (22.7M params, 384-dim)
├── Base MiniLM:  Frozen encoder → Mean pooling → 384-dim embeddings
└── LoRA MiniLM:  Encoder + LoRA(r=8, α=32, target=Q,V) → Mean pooling → 384-dim embeddings
                  LoRA adds 73,728 trainable params (0.32% of total)
```

### Evaluation Pipeline

```
┌─────────────────┐     ┌─────────────────┐
│   Base MiniLM   │     │   LoRA MiniLM    │
│   (frozen)      │     │   (merged LoRA)  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
    Mean-Pooled              Mean-Pooled
    Embeddings               Embeddings
    (384-dim)                (384-dim)
         │                       │
         ├───────────┬───────────┤
         │           │           │
         ▼           ▼           ▼
    CrowS-Pairs   SST-2     Sentence
    Bias Eval     Sentiment  Analysis
                  (LogReg)
```

---

## 📁 Project Structure

```
Can-Fine-Tuning-Language-Models-on-Bias-itself-Introduce-Bias/
│
├── paper/                          # EMNLP 2026 short paper
│   ├── paper.tex                   # LaTeX source
│   └── references.bib              # Bibliography
│
├── models/                         # Saved model weights
│   ├── frozen_minilm/              # Base encoder (full state dict)
│   │   ├── model.pt
│   │   ├── tokenizer.json
│   │   └── ...
│   └── lora_minilm/                # LoRA-adapted encoder
│       ├── adapter/                # PEFT adapter weights
│       │   ├── adapter_config.json
│       │   └── adapter_model.safetensors
│       └── classifier.pt           # StereoSet classifier head
│
├── data/                           # Evaluation datasets
│   ├── CrowS/
│   │   └── crows_pairs_anonymized.csv
│   └── SST-2/
│       ├── train.tsv
│       └── test.tsv
│
├── src/                            # Source code
│   ├── embeddings/                 # Embedding extraction
│   │   ├── __init__.py
│   │   └── extract_embeddings.py   # EmbeddingExtractor class
│   │
│   ├── evaluation/                 # Evaluation modules
│   │   ├── metrics.py              # General classification metrics
│   │   ├── crows_pairs_eval.py     # CrowS-Pairs bias evaluation
│   │   ├── sentiment_eval.py       # SST-2 sentiment evaluation
│   │   └── analysis.py             # Plots & sentence-level analysis
│   │
│   ├── models/                     # Model implementations (StereoSet)
│   │   ├── frozen_minilm.py        # Frozen MiniLM classifier
│   │   ├── lora_minilm.py          # LoRA MiniLM classifier
│   │   └── logistic_regression.py  # TF-IDF baseline
│   │
│   ├── inference/                  # Production inference
│   │   └── predictor.py            # BiasPredictor API
│   │
│   ├── utils/
│   │   ├── config.py               # Configuration & paths
│   │   └── logger.py               # Logging setup
│   │
│   ├── load_data.py                # StereoSet data loading
│   └── preprocess.py               # Data preprocessing & splits
│
├── scripts/                        # Reproducible experiment scripts
│   ├── run_pipeline.py             # Full pipeline (CrowS + SST-2 + analysis)
│   ├── run_crows.py                # CrowS-Pairs evaluation only
│   └── run_sentiment.py            # SST-2 evaluation only
│
├── notebooks/                      # Training notebooks (Colab)
│   ├── frozen_minilm.ipynb
│   └── lora_minilm.ipynb
│
├── results/                        # Generated outputs (not committed)
│   ├── crows_pairs/
│   │   ├── overall_metrics.json
│   │   ├── category_metrics_base.csv
│   │   ├── category_metrics_lora.csv
│   │   ├── flip_analysis.csv
│   │   ├── score_distribution.png
│   │   └── category_comparison.png
│   ├── sentiment/
│   │   ├── metrics.json
│   │   └── confusion_matrices.png
│   ├── analysis/
│   │   ├── representative_examples.csv
│   │   └── behavioral_diff.png
│   └── summary_table.md
│
├── streamlit_app.py                # Interactive demo
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- GPU recommended (CUDA-capable) for embedding extraction

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Can-Fine-Tuning-Language-Models-on-Bias-itself-Introduce-Bias

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

```bash
# Copy the example env file and add your HuggingFace token
cp .env.example .env
# Edit .env → HF_TOKEN=your_huggingface_token_here
```

> **Note**: An HF token is needed to download the base `sentence-transformers/all-MiniLM-L6-v2` model from HuggingFace on first run.

---

## 🧪 Running Experiments

### Full Pipeline (Recommended)

```bash
# GPU
python scripts/run_pipeline.py --device cuda --output-dir results/ --batch-size 64

# CPU (slower, useful for debugging)
python scripts/run_pipeline.py --device cpu --output-dir results/ --batch-size 8
```

This runs all evaluations and generates tables + plots in `results/`.

### Individual Evaluations

```bash
# CrowS-Pairs bias evaluation only
python scripts/run_crows.py --device cuda --output-dir results/crows_pairs/

# SST-2 sentiment evaluation only
python scripts/run_sentiment.py --device cuda --output-dir results/sentiment/
```

### GPU Cluster (SLURM)

```bash
#!/bin/bash
#SBATCH --job-name=bias-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=bias-eval_%j.log

conda activate bias-eval
python scripts/run_pipeline.py --device cuda --output-dir results/ --batch-size 64
```

---

## 📈 Detailed Results

### CrowS-Pairs: Category-wise Stereotype Scores

| Category | N | Base SS | LoRA SS | Δ |
|---|:---:|:---:|:---:|:---:|
| Race-color | 516 | 0.446 | 0.333 | −0.113 |
| Gender | 262 | 0.569 | 0.519 | −0.050 |
| Socioeconomic | 172 | 0.314 | 0.407 | **+0.093** |
| Nationality | 159 | 0.673 | 0.591 | −0.082 |
| Religion | 105 | 0.695 | 0.467 | −0.228 |
| Age | 87 | 0.598 | 0.322 | **−0.276** |
| Sexual orientation | 84 | 0.500 | 0.381 | −0.119 |
| Physical appearance | 63 | 0.508 | 0.381 | −0.127 |
| Disability | 60 | 0.533 | 0.533 | 0.000 |

> An unbiased model scores SS = 0.50. Values below 0.50 indicate anti-stereotypical preference.

### SST-2: Sentiment Classification

| Metric | Base MiniLM | LoRA MiniLM |
|---|:---:|:---:|
| Accuracy | 0.808 | 0.808 |
| Precision | 0.796 | 0.806 |
| Recall | 0.826 | 0.810 |
| F1 | 0.811 | 0.808 |

---

## 🔧 StereoSet Bias Detection (Original Task)

The LoRA model was originally trained to classify StereoSet text into three categories:

| Model | Accuracy | F1 (Macro) |
|---|:---:|:---:|
| Logistic Regression | 39.44% | 39.39% |
| Frozen MiniLM | 54.29% | 54.21% |
| **LoRA MiniLM** | **75.00%** | **74.56%** |

---

## 📝 License

This project is for educational and research purposes.

---

*Built with a focus on responsible AI research and reproducible experimental methodology.*
