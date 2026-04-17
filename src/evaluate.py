import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report,
)
from sklearn.metrics import top_k_accuracy_score


def illegal_move_rate(y_pred: np.ndarray, legal_masks: np.ndarray) -> float:
    """
    Fraction of predictions that choose an illegal column.
    y_pred: shape (n_samples,)
    legal_masks: shape (n_samples, 7), 1 = legal, 0 = illegal
    """
    illegal = [
        1 if legal_masks[i, y_pred[i]] == 0 else 0
        for i in range(len(y_pred))
    ]
    return float(np.mean(illegal))


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    legal_masks: np.ndarray,
):
    """
    Returns a dictionary of standardized multiclass metrics.
    """
    labels = list(range(7))

    metrics = {
        "top1_accuracy": accuracy_score(y_true, y_pred),
        "top2_accuracy": top_k_accuracy_score(
            y_true, y_score, k=2, labels=labels
        ),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "illegal_move_rate": illegal_move_rate(y_pred, legal_masks),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, digits=4
        ),
    }

    return metrics


def print_metrics(metrics: dict):
    print("\n=== Evaluation Metrics ===")
    print(f"Top-1 Accuracy   : {metrics['top1_accuracy']:.4f}")
    print(f"Top-2 Accuracy   : {metrics['top2_accuracy']:.4f}")
    print(f"Macro-F1         : {metrics['macro_f1']:.4f}")
    print(f"Illegal-move rate: {metrics['illegal_move_rate']:.4f}")

    print("\nConfusion Matrix (rows = true class, cols = predicted class):")
    print(metrics["confusion_matrix"])

    print("\nClassification Report:")
    print(metrics["classification_report"])