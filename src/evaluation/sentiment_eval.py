"""
SST-2 sentiment classification using frozen embeddings.

Train identical logistic-regression classifier heads on frozen sentence
embeddings from both the base and LoRA MiniLM models to test whether
bias-aware fine-tuning degrades general-purpose NLU capability.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from ..embeddings.extract_embeddings import EmbeddingExtractor
from ..utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_sst2(
    train_path: str | Path,
    test_path: str | Path,
) -> Tuple[list, np.ndarray, list, np.ndarray]:
    """
    Load SST-2 train and test splits from TSV files.

    Expected format: ``sentence\\tlabel`` (tab-separated, no header).

    Returns
    -------
    train_texts, train_labels, test_texts, test_labels
    """
    def _read(path: str | Path):
        df = pd.read_csv(path, sep="\t", header=None, names=["sentence", "label"])
        df = df.dropna().reset_index(drop=True)
        texts = df["sentence"].tolist()
        labels = df["label"].astype(int).values
        return texts, labels

    train_texts, train_labels = _read(train_path)
    test_texts, test_labels = _read(test_path)

    logger.info(
        "SST-2 loaded: %d train, %d test  |  label dist train=%s test=%s",
        len(train_texts),
        len(test_texts),
        dict(zip(*np.unique(train_labels, return_counts=True))),
        dict(zip(*np.unique(test_labels, return_counts=True))),
    )
    return train_texts, train_labels, test_texts, test_labels


# ---------------------------------------------------------------------------
# Classifier training & evaluation
# ---------------------------------------------------------------------------
def train_sentiment_classifier(
    embeddings: np.ndarray,
    labels: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Train a logistic-regression head on frozen embeddings.

    Hyperparameters are fixed for reproducibility across model comparisons.
    """
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        random_state=random_state,
    )
    clf.fit(embeddings, labels)
    train_acc = clf.score(embeddings, labels)
    logger.info("Classifier trained — train accuracy: %.4f", train_acc)
    return clf


def evaluate_sentiment(
    clf: LogisticRegression,
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict:
    """
    Evaluate the sentiment classifier.

    Returns
    -------
    dict with accuracy, precision, recall, f1, confusion_matrix,
    classification_report.
    """
    y_pred = clf.predict(embeddings)

    metrics = {
        "accuracy": float(accuracy_score(labels, y_pred)),
        "precision": float(precision_score(labels, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(labels, y_pred, average="binary", zero_division=0)),
        "f1": float(f1_score(labels, y_pred, average="binary", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, y_pred).tolist(),
        "classification_report": classification_report(
            labels, y_pred, target_names=["negative", "positive"], zero_division=0
        ),
    }
    return metrics


# ---------------------------------------------------------------------------
# End-to-end convenience
# ---------------------------------------------------------------------------
def run_sentiment_eval(
    extractor: EmbeddingExtractor,
    train_texts: list,
    train_labels: np.ndarray,
    test_texts: list,
    test_labels: np.ndarray,
    batch_size: int = 64,
) -> Dict:
    """
    Full SST-2 pipeline for a single model:
    extract embeddings → train classifier → evaluate.
    """
    logger.info("Extracting train embeddings …")
    train_emb = extractor.extract(train_texts, batch_size=batch_size)

    logger.info("Extracting test embeddings …")
    test_emb = extractor.extract(test_texts, batch_size=batch_size)

    logger.info("Training classifier …")
    clf = train_sentiment_classifier(train_emb, train_labels)

    logger.info("Evaluating …")
    metrics = evaluate_sentiment(clf, test_emb, test_labels)

    return {
        "metrics": metrics,
        "classifier": clf,
        "train_embeddings": train_emb,
        "test_embeddings": test_emb,
    }
