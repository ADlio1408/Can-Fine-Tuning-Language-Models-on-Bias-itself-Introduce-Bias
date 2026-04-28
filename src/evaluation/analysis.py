"""
Sentence-level behavioural analysis and publication-quality plots.

1. Identify 5–10 representative CrowS-Pairs examples where base and LoRA
   models differ the most.
2. Plot distributions of score differences.
3. Plot category-wise stereotype-score comparison.
4. Plot side-by-side confusion matrices for SST-2.

All figures are saved at 300 DPI with tight layout for EMNLP / ACL papers.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for cluster / CI

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from ..utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------
STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}
plt.rcParams.update(STYLE)
sns.set_palette("muted")


# ---------------------------------------------------------------------------
# 1.  Representative examples
# ---------------------------------------------------------------------------
def find_representative_examples(
    base_result: Dict,
    lora_result: Dict,
    n: int = 10,
) -> pd.DataFrame:
    """
    Return the *n* CrowS-Pairs examples with the largest absolute
    difference in cosine similarity between base and LoRA models.
    """
    base_df = base_result["df"]
    lora_df = lora_result["df"]

    diff = (lora_df["cosine_sim"] - base_df["cosine_sim"]).abs()

    top_idx = diff.nlargest(n).index

    examples = base_df.loc[top_idx, ["sent_more", "sent_less", "stereo_antistereo", "bias_type"]].copy()
    examples["base_cosine_sim"] = base_df.loc[top_idx, "cosine_sim"].values
    examples["lora_cosine_sim"] = lora_df.loc[top_idx, "cosine_sim"].values
    examples["delta"] = diff.loc[top_idx].values
    examples["base_prefers_stereo"] = base_df.loc[top_idx, "prefers_stereo"].values
    examples["lora_prefers_stereo"] = lora_df.loc[top_idx, "prefers_stereo"].values

    examples = examples.sort_values("delta", ascending=False).reset_index(drop=True)
    return examples


# ---------------------------------------------------------------------------
# 2.  Score-difference distribution
# ---------------------------------------------------------------------------
def plot_score_distributions(
    base_result: Dict,
    lora_result: Dict,
    save_path: str | Path,
) -> None:
    """
    Histogram of per-pair cosine-similarity values for both models
    (overlaid), plus a histogram of the difference.
    """
    base_sim = base_result["df"]["cosine_sim"].values
    lora_sim = lora_result["df"]["cosine_sim"].values
    delta = lora_sim - base_sim

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # -- Panel A: overlaid cosine-sim distributions --
    ax = axes[0]
    ax.hist(base_sim, bins=50, alpha=0.6, label="Base MiniLM", color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.hist(lora_sim, bins=50, alpha=0.6, label="LoRA MiniLM", color="#DD8452", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Cosine Similarity (stereo ↔ anti-stereo)")
    ax.set_ylabel("Count")
    ax.set_title("(a) Pair-wise Cosine Similarity")
    ax.legend()

    # -- Panel B: difference distribution --
    ax = axes[1]
    ax.hist(delta, bins=50, color="#55A868", edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Δ Cosine Sim  (LoRA − Base)")
    ax.set_ylabel("Count")
    ax.set_title("(b) Per-Pair Similarity Change")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved score distribution plot → %s", save_path)


# ---------------------------------------------------------------------------
# 3.  Category comparison
# ---------------------------------------------------------------------------
def plot_category_comparison(
    base_cat_df: pd.DataFrame,
    lora_cat_df: pd.DataFrame,
    save_path: str | Path,
) -> None:
    """
    Grouped bar chart comparing stereotype score per bias category.
    """
    cats = base_cat_df["bias_type"].values
    x = np.arange(len(cats))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - width / 2, base_cat_df["stereotype_score"], width,
                   label="Base MiniLM", color="#4C72B0", edgecolor="white", linewidth=0.4)
    bars2 = ax.bar(x + width / 2, lora_cat_df["stereotype_score"], width,
                   label="LoRA MiniLM", color="#DD8452", edgecolor="white", linewidth=0.4)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="Unbiased (0.50)")
    ax.set_xlabel("Bias Category")
    ax.set_ylabel("Stereotype Score")
    ax.set_title("Stereotype Score by Category")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=35, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved category comparison plot → %s", save_path)


# ---------------------------------------------------------------------------
# 4.  Confusion matrices for SST-2
# ---------------------------------------------------------------------------
def plot_confusion_matrices(
    base_cm: list,
    lora_cm: list,
    save_path: str | Path,
) -> None:
    """
    Side-by-side heatmap confusion matrices for SST-2 sentiment.
    """
    base_cm = np.array(base_cm)
    lora_cm = np.array(lora_cm)
    labels = ["Negative", "Positive"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, cm, title in zip(axes, [base_cm, lora_cm], ["Base MiniLM", "LoRA MiniLM"]):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar=False,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)

    plt.suptitle("SST-2 Confusion Matrices", fontsize=13, y=1.02)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved confusion matrices → %s", save_path)


# ---------------------------------------------------------------------------
# 5.  Behavioural difference scatter
# ---------------------------------------------------------------------------
def plot_behavioral_diff(
    base_result: Dict,
    lora_result: Dict,
    save_path: str | Path,
) -> None:
    """
    Scatter plot: base cosine sim (x) vs LoRA cosine sim (y).
    Points are coloured by bias_type.  Diagonal = no change.
    """
    base_df = base_result["df"]
    lora_df = lora_result["df"]

    fig, ax = plt.subplots(figsize=(7, 6))

    categories = base_df["bias_type"].unique()
    palette = sns.color_palette("husl", len(categories))
    for cat, colour in zip(categories, palette):
        mask = base_df["bias_type"] == cat
        ax.scatter(
            base_df.loc[mask, "cosine_sim"],
            lora_df.loc[mask, "cosine_sim"],
            s=12,
            alpha=0.5,
            label=cat,
            color=colour,
            edgecolors="none",
        )

    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Base MiniLM — Cosine Sim")
    ax.set_ylabel("LoRA MiniLM — Cosine Sim")
    ax.set_title("Pair-wise Cosine Similarity: Base vs LoRA")
    ax.legend(fontsize=7, ncol=2, loc="lower right")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved behavioural diff scatter → %s", save_path)
