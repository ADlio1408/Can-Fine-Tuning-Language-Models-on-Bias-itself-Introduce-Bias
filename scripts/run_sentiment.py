#!/usr/bin/env python3
"""
Standalone SST-2 sentiment evaluation.

Usage:
    python scripts/run_sentiment.py --device cuda --output-dir results/sentiment/
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.extract_embeddings import EmbeddingExtractor
from src.evaluation.sentiment_eval import load_sst2, run_sentiment_eval
from src.evaluation.analysis import plot_confusion_matrices
from src.utils.logger import get_logger

logger = get_logger("sentiment_eval")

SST2_TRAIN = PROJECT_ROOT / "data" / "SST-2" / "train.tsv"
SST2_TEST = PROJECT_ROOT / "data" / "SST-2" / "test.tsv"


def main(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)

    # Load models
    base_ext = EmbeddingExtractor("base", device=args.device)
    lora_ext = EmbeddingExtractor("lora", device=args.device)

    # Load data
    train_texts, train_labels, test_texts, test_labels = load_sst2(SST2_TRAIN, SST2_TEST)

    # Evaluate
    logger.info("Evaluating base model …")
    base_res = run_sentiment_eval(
        base_ext, train_texts, train_labels, test_texts, test_labels,
        batch_size=args.batch_size,
    )

    logger.info("Evaluating LoRA model …")
    lora_res = run_sentiment_eval(
        lora_ext, train_texts, train_labels, test_texts, test_labels,
        batch_size=args.batch_size,
    )

    # Save metrics
    out.mkdir(parents=True, exist_ok=True)
    metrics = {"base": base_res["metrics"], "lora": lora_res["metrics"]}
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrices
    plot_confusion_matrices(
        base_res["metrics"]["confusion_matrix"],
        lora_res["metrics"]["confusion_matrix"],
        out / "confusion_matrices.png",
    )

    # Print results
    print("\n=== SST-2 Sentiment Results ===")
    for model_name in ("base", "lora"):
        m = metrics[model_name]
        print(f"\n{model_name.upper()}:")
        for k in ("accuracy", "precision", "recall", "f1"):
            print(f"  {k}: {m[k]:.4f}")
        print(f"\n  Classification Report:\n{m['classification_report']}")

    print(f"\nAll outputs → {out.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SST-2 sentiment evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "results" / "sentiment"))
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
