#!/usr/bin/env python3
"""
Full experimental pipeline: CrowS-Pairs + SST-2 + sentence-level analysis.

Usage (GPU cluster):
    python scripts/run_pipeline.py --device cuda --output-dir results/ --batch-size 64

Usage (CPU / debug):
    python scripts/run_pipeline.py --device cpu --output-dir results/ --batch-size 8
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so ``src`` is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.extract_embeddings import EmbeddingExtractor
from src.evaluation.crows_pairs_eval import (
    load_crows_pairs,
    evaluate_crows_pairs,
    category_analysis,
    flip_analysis,
)
from src.evaluation.sentiment_eval import load_sst2, run_sentiment_eval
from src.evaluation.analysis import (
    find_representative_examples,
    plot_score_distributions,
    plot_category_comparison,
    plot_confusion_matrices,
    plot_behavioral_diff,
)
from src.utils.logger import get_logger

logger = get_logger("pipeline")

# ---------------------------------------------------------------------------
# Default data paths (relative to PROJECT_ROOT)
# ---------------------------------------------------------------------------
CROWS_CSV = PROJECT_ROOT / "data" / "CrowS" / "crows_pairs_anonymized.csv"
SST2_TRAIN = PROJECT_ROOT / "data" / "SST-2" / "train.tsv"
SST2_TEST = PROJECT_ROOT / "data" / "SST-2" / "test.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    logger.info("Saved → %s", path)


def _save_csv(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved → %s", path)


def _build_summary_table(
    crows_base: dict,
    crows_lora: dict,
    sst_base: dict,
    sst_lora: dict,
) -> str:
    """Build a Markdown comparison table for the paper."""
    header = (
        "| Metric | Base MiniLM | LoRA MiniLM |\n"
        "|--------|-------------|-------------|\n"
    )
    rows = []

    # CrowS-Pairs
    cb = crows_base["metrics"]
    cl = crows_lora["metrics"]
    rows.append(f"| **CrowS-Pairs** | | |")
    rows.append(f"| Stereotype Score | {cb['stereotype_score']:.4f} | {cl['stereotype_score']:.4f} |")
    rows.append(f"| Accuracy | {cb['accuracy']:.4f} | {cl['accuracy']:.4f} |")
    rows.append(f"| Precision | {cb['precision']:.4f} | {cl['precision']:.4f} |")
    rows.append(f"| Recall | {cb['recall']:.4f} | {cl['recall']:.4f} |")
    rows.append(f"| F1 | {cb['f1']:.4f} | {cl['f1']:.4f} |")

    # SST-2
    sb = sst_base["metrics"]
    sl = sst_lora["metrics"]
    rows.append(f"| **SST-2 Sentiment** | | |")
    rows.append(f"| Accuracy | {sb['accuracy']:.4f} | {sl['accuracy']:.4f} |")
    rows.append(f"| Precision | {sb['precision']:.4f} | {sl['precision']:.4f} |")
    rows.append(f"| Recall | {sb['recall']:.4f} | {sl['recall']:.4f} |")
    rows.append(f"| F1 | {sb['f1']:.4f} | {sl['f1']:.4f} |")

    return header + "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    t0 = time.time()
    out = Path(args.output_dir)

    # ---------------------------------------------------------------
    # 1.  Load models
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1 / 5 — Loading models")
    logger.info("=" * 60)

    base_ext = EmbeddingExtractor("base", device=args.device)
    lora_ext = EmbeddingExtractor("lora", device=args.device)

    # ---------------------------------------------------------------
    # 2.  CrowS-Pairs evaluation
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2 / 5 — CrowS-Pairs bias evaluation")
    logger.info("=" * 60)

    crows_df = load_crows_pairs(CROWS_CSV)

    crows_base = evaluate_crows_pairs(base_ext, crows_df, batch_size=args.batch_size)
    crows_lora = evaluate_crows_pairs(lora_ext, crows_df, batch_size=args.batch_size)

    # Overall metrics
    _save_json(
        {"base": crows_base["metrics"], "lora": crows_lora["metrics"]},
        out / "crows_pairs" / "overall_metrics.json",
    )

    # Category analysis
    base_cat = category_analysis(crows_base)
    lora_cat = category_analysis(crows_lora)
    _save_csv(base_cat, out / "crows_pairs" / "category_metrics_base.csv")
    _save_csv(lora_cat, out / "crows_pairs" / "category_metrics_lora.csv")

    # Flip analysis
    flips = flip_analysis(crows_base, crows_lora)
    _save_csv(flips, out / "crows_pairs" / "flip_analysis.csv")

    # Plots
    plot_score_distributions(
        crows_base, crows_lora,
        out / "crows_pairs" / "score_distribution.png",
    )
    plot_category_comparison(
        base_cat, lora_cat,
        out / "crows_pairs" / "category_comparison.png",
    )

    # ---------------------------------------------------------------
    # 3.  SST-2 sentiment evaluation
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3 / 5 — SST-2 sentiment evaluation")
    logger.info("=" * 60)

    train_texts, train_labels, test_texts, test_labels = load_sst2(SST2_TRAIN, SST2_TEST)

    sst_base = run_sentiment_eval(
        base_ext, train_texts, train_labels, test_texts, test_labels,
        batch_size=args.batch_size,
    )
    sst_lora = run_sentiment_eval(
        lora_ext, train_texts, train_labels, test_texts, test_labels,
        batch_size=args.batch_size,
    )

    _save_json(
        {"base": sst_base["metrics"], "lora": sst_lora["metrics"]},
        out / "sentiment" / "metrics.json",
    )

    plot_confusion_matrices(
        sst_base["metrics"]["confusion_matrix"],
        sst_lora["metrics"]["confusion_matrix"],
        out / "sentiment" / "confusion_matrices.png",
    )

    # ---------------------------------------------------------------
    # 4.  Sentence-level analysis
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 4 / 5 — Sentence-level analysis")
    logger.info("=" * 60)

    examples = find_representative_examples(crows_base, crows_lora, n=10)
    _save_csv(examples, out / "analysis" / "representative_examples.csv")

    plot_behavioral_diff(
        crows_base, crows_lora,
        out / "analysis" / "behavioral_diff.png",
    )

    # ---------------------------------------------------------------
    # 5.  Summary
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 5 / 5 — Generating summary")
    logger.info("=" * 60)

    summary = _build_summary_table(crows_base, crows_lora, sst_base, sst_lora)
    summary_path = out / "summary_table.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary, encoding="utf-8")
    logger.info("Saved → %s", summary_path)

    # Print summary to stdout
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(summary)
    print(f"Flip count: {len(flips)} / {len(crows_df)} "
          f"({100 * len(flips) / len(crows_df):.1f}%)")
    print(f"\nAll outputs saved to: {out.resolve()}")
    print(f"Total time: {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full bias-evaluation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="PyTorch device (cuda / cpu).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "results"),
        help="Directory to write all results.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding extraction.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
