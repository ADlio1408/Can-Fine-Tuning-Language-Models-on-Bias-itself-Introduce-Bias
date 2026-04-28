#!/usr/bin/env python3
"""
Standalone CrowS-Pairs bias evaluation.

Usage:
    python scripts/run_crows.py --device cuda --output-dir results/crows_pairs/
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.extract_embeddings import EmbeddingExtractor
from src.evaluation.crows_pairs_eval import (
    load_crows_pairs,
    evaluate_crows_pairs,
    category_analysis,
    flip_analysis,
)
from src.evaluation.analysis import (
    find_representative_examples,
    plot_score_distributions,
    plot_category_comparison,
    plot_behavioral_diff,
)
from src.utils.logger import get_logger

logger = get_logger("crows_eval")

CROWS_CSV = PROJECT_ROOT / "data" / "CrowS" / "crows_pairs_anonymized.csv"


def main(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)

    # Load models
    base_ext = EmbeddingExtractor("base", device=args.device)
    lora_ext = EmbeddingExtractor("lora", device=args.device)

    # Load data
    crows_df = load_crows_pairs(CROWS_CSV)

    # Evaluate
    logger.info("Evaluating base model …")
    base_res = evaluate_crows_pairs(base_ext, crows_df, batch_size=args.batch_size)

    logger.info("Evaluating LoRA model …")
    lora_res = evaluate_crows_pairs(lora_ext, crows_df, batch_size=args.batch_size)

    # Save overall metrics
    out.mkdir(parents=True, exist_ok=True)
    metrics = {"base": base_res["metrics"], "lora": lora_res["metrics"]}
    with open(out / "overall_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Category analysis
    base_cat = category_analysis(base_res)
    lora_cat = category_analysis(lora_res)
    base_cat.to_csv(out / "category_metrics_base.csv", index=False)
    lora_cat.to_csv(out / "category_metrics_lora.csv", index=False)

    # Flip analysis
    flips = flip_analysis(base_res, lora_res)
    flips.to_csv(out / "flip_analysis.csv", index=False)

    # Representative examples
    examples = find_representative_examples(base_res, lora_res, n=10)
    examples.to_csv(out / "representative_examples.csv", index=False)

    # Plots
    plot_score_distributions(base_res, lora_res, out / "score_distribution.png")
    plot_category_comparison(base_cat, lora_cat, out / "category_comparison.png")
    plot_behavioral_diff(base_res, lora_res, out / "behavioral_diff.png")

    # Print results
    print("\n=== CrowS-Pairs Results ===")
    for model_name, m in metrics.items():
        print(f"\n{model_name.upper()}:")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}")
    print(f"\nFlips: {len(flips)} / {len(crows_df)} ({100*len(flips)/len(crows_df):.1f}%)")
    print(f"All outputs → {out.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CrowS-Pairs bias evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "results" / "crows_pairs"))
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
