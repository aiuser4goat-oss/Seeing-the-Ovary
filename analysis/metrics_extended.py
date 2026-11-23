# analysis/metrics_extended.py

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from scipy.stats import sem, t


def compute_ci(scores, confidence=0.95):
    scores = np.array(scores, dtype=float)
    n = len(scores)
    mean = np.mean(scores)
    if n <= 1:
        return mean, None, None
    se = sem(scores)
    h = se * t.ppf((1 + confidence) / 2.0, n - 1)
    return mean, mean - h, mean + h


def compute_bootstrapped_auc(y_true, y_prob, n_bootstraps=500, seed=42):
    np.random.seed(seed)
    aucs = []
    for _ in range(n_bootstraps):
        idxs = np.random.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idxs])) < 2:
            continue
        aucs.append(
            roc_auc_score(y_true[idxs], y_prob[idxs], multi_class="ovr")
        )
    if len(aucs) == 0:
        return None, None, None
    return compute_ci(aucs)


def compute_metrics(y_true, y_pred, y_prob, class_names):
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")

    try:
        auc_global = roc_auc_score(y_true, y_prob, multi_class="ovr")
        auc_mean, auc_low, auc_high = compute_bootstrapped_auc(y_true, y_prob)
    except Exception:
        auc_global, auc_mean, auc_low, auc_high = None, None, None, None

    return {
        "report": report,
        "confusion_matrix": cm,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc_global,
        "auc_ci": (auc_mean, auc_low, auc_high)
    }