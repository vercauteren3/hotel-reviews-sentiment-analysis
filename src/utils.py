# ------------------------------------------------------------------------------
# src/utils.py — helper functions for model evaluation
# ------------------------------------------------------------------------------

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

def _coerce_binary(y):
    """Convert string labels ('0'/'1') to numeric 0/1 arrays."""
    y = np.asarray(y)
    if y.dtype.kind in {"U", "S", "O"}:  # Unicode, String, or Object type
        try:
            y = y.astype(int)
        except Exception:
            pass
    return y


def evaluate_model(true_labels, predicted_classes, positive_class="1"):
    """
    Evaluate a binary classification model.
    Returns Accuracy, Precision, Recall, Specificity, F1_Score, and Balanced_Accuracy.
    """
    y_true = _coerce_binary(true_labels)
    y_pred = _coerce_binary(predicted_classes)
    pos_label = 1 if str(positive_class) == "1" else 0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    balanced_accuracy = (recall + specificity) / 2.0

    return {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "Specificity": float(specificity),
        "F1_Score": float(f1),
        "Balanced_Accuracy": float(balanced_accuracy),
    }


# Optional: self-test if you run this file directly
if __name__ == "__main__":
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    print("Self-test:", evaluate_model(y_true, y_pred))