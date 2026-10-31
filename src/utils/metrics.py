"""Evaluation metrics computation for Real vs. AI-Generated image detection.

Computes:
- Accuracy, Precision, Recall, Specificity, F1-Score
- Confusion Matrix (TN, FP, FN, TP)
- ROC Curve & Area Under the ROC Curve (ROC-AUC)
- Average Precision (mAP)
"""

from typing import Dict, Any, Optional, Union
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


def compute_evaluation_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    y_prob: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """Calculates comprehensive classification metrics.
    
    Args:
        y_true: Ground truth binary labels (0 = Real, 1 = Fake).
        y_pred: Predicted binary labels (0 = Real, 1 = Fake).
        y_prob: Optional predicted probabilities of being Fake (class 1) in [0, 1].
        
    Returns:
        Dictionary of computed metric values.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if y_prob is not None and isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.detach().cpu().numpy()

    y_true = y_true.astype(int).flatten()
    y_pred = y_pred.astype(int).flatten()

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": specificity,
        "f1_score": f1,
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "sample_counts": {
            "total": len(y_true),
            "real": int((y_true == 0).sum()),
            "fake": int((y_true == 1).sum()),
        }
    }

    if y_prob is not None:
        y_prob = y_prob.flatten()
        try:
            # ROC-AUC requires both classes present in y_true
            if len(np.unique(y_true)) > 1:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
                metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
            else:
                metrics["roc_auc"] = None
                metrics["average_precision"] = None
        except Exception:
            metrics["roc_auc"] = None
            metrics["average_precision"] = None

    return metrics


def format_metrics_table(metrics: Dict[str, Any], title: str = "Evaluation Metrics") -> str:
    """Formats metrics dictionary into a clean markdown table."""
    cm = metrics["confusion_matrix"]
    auc_str = f"{metrics['roc_auc'] * 100:.2f}%" if metrics.get("roc_auc") is not None else "N/A"
    ap_str = f"{metrics['average_precision'] * 100:.2f}%" if metrics.get("average_precision") is not None else "N/A"

    table = f"""
### {title}
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Accuracy** | `{metrics['accuracy'] * 100:.2f}%` | Overall correct real and fake predictions |
| **Precision** | `{metrics['precision'] * 100:.2f}%` | How many flagged AI images are genuinely AI |
| **Recall (Sensitivity)** | `{metrics['recall'] * 100:.2f}%` | Proportion of actual AI images detected |
| **Specificity** | `{metrics['specificity'] * 100:.2f}%` | Proportion of actual real images kept genuine |
| **F1-Score** | `{metrics['f1_score'] * 100:.2f}%` | Harmonic mean of Precision and Recall |
| **ROC-AUC** | `{auc_str}` | Area Under Receiver Operating Characteristic Curve |
| **Average Precision** | `{ap_str}` | Area under Precision-Recall curve |

**Confusion Matrix:**
- **True Negatives (TN - Real correctly predicted as Real):** {cm['tn']}
- **False Positives (FP - Real misclassified as AI):** {cm['fp']}
- **False Negatives (FN - AI misclassified as Real):** {cm['fn']}
- **True Positives (TP - AI correctly predicted as AI):** {cm['tp']}
"""
    return table
