# analysis/plotting.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def plot_training_curves(history, out_path="plots/training_curves.png"):
    ensure_dir(os.path.dirname(out_path) or ".")
    hist = history.history

    plt.figure(figsize=(7, 5))
    plt.plot(hist.get("loss", []), label="Train loss")
    plt.plot(hist.get("val_loss", []), label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_roc_multi(y_true, y_prob, class_names, out_path="plots/roc_curve.png"):
    ensure_dir(os.path.dirname(out_path) or ".")
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_classes = len(class_names)

    # one-hot
    y_onehot = np.eye(n_classes)[y_true]

    plt.figure(figsize=(7, 5))
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_onehot[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()