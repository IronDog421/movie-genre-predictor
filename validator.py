from typing import Dict, Optional
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, hamming_loss

def compute_metrics(labels: np.ndarray,
                    preds: np.ndarray,
                    threshold: Optional[float] = None) -> Dict[str, float]:
    """
    Calcula precisión/recall/F1 macro, subset accuracy y Hamming loss.
    Si `preds` son probabilidades/logits, se binarizan con `threshold` (por defecto 0.5).

    Parameters
    ----------
    labels : array (n_samples, n_classes)
        Etiquetas verdaderas en {0,1}.
    preds : array (n_samples, n_classes)
        Predicciones en {0,1} o probabilidades/logits.
    threshold : float, opcional
        Umbral para binarizar si `preds` son float. Por defecto 0.5.

    Returns
    -------
    dict
        {'accuracy', 'f1', 'precision', 'recall', 'hamming_loss'}
    """
    y_true = np.asarray(labels)
    y_pred = np.asarray(preds)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: labels {y_true.shape} vs preds {y_pred.shape}")

    # Binariza si las predicciones son float (probabilidades/logits)
    if y_pred.dtype.kind in "fc":
        thr = 0.5 if threshold is None else float(threshold)
        y_pred = (y_pred >= thr).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)   # en multilabel es subset accuracy (exact match)
    hl = hamming_loss(y_true, y_pred)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "hamming_loss": hl,
    }
