# train/evaluate.py
import numpy as np
import tensorflow as tf

from utils.dataset_loader import build_datasets
from analysis.metrics_extended import compute_metrics
from analysis.plotting import plot_roc_multi


def run_eval(
    model_path="models/best_model.h5",
    train_dir="data/Train",
    val_dir="data/Val",
    img_size=(224, 224),
    batch_size=16,
):
    print("Loading datasets for evaluation...")
    train_ds, val_ds, class_names = build_datasets(
        train_dir=train_dir,
        val_dir=val_dir,
        img_size=img_size,
        batch_size=batch_size,
        augmentation_layer=None,
    )

    print("Loading model:", model_path)
    model = tf.keras.models.load_model(model_path, compile=False)

    y_true_batches = []
    y_prob_batches = []

    for x, y in val_ds:
        preds = model.predict(x, verbose=0)
        y_prob_batches.append(preds)
        y_true_batches.append(y.numpy())

    y_prob = np.concatenate(y_prob_batches, axis=0)
    y_true_onehot = np.concatenate(y_true_batches, axis=0)
    y_true_idx = np.argmax(y_true_onehot, axis=1)
    y_pred_idx = np.argmax(y_prob, axis=1)

    metrics = compute_metrics(y_true_idx, y_pred_idx, y_prob, class_names)

    print("\n=== VALIDATION REPORT ===")
    print(metrics["report"])
    print("Confusion Matrix:\n", metrics["confusion_matrix"])
    print("AUC:", metrics["auc"])
    print("AUC CI:", metrics["auc_ci"])
    print("F1:", metrics["f1"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])

    # Plot ROC curves for multi-class
    plot_roc_multi(
        y_true_idx,
        y_prob,
        class_names,
        out_path="plots/roc_val.png",
    )

    print("\nROC curve saved to plots/roc_val.png")

    return metrics