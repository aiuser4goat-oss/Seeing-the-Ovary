# analysis/external_validate.py
import numpy as np
import tensorflow as tf

from utils.dataset_loader import build_datasets
from analysis.metrics_extended import compute_metrics
from analysis.plotting import plot_roc_multi


def run_external_validation(
    model_path="models/best_model.h5",
    train_dir="data/Train",
    external_dir="data/External",
    img_size=(224, 224),
    batch_size=16,
):
    """
    Assumes external_dir has SAME subfolder class names as Train.
    """
    print("Loading train + external datasets...")
    # use build_datasets just to keep class_names consistent
    train_ds, ext_ds, class_names = build_datasets(
        train_dir=train_dir,
        val_dir=external_dir,
        img_size=img_size,
        batch_size=batch_size,
        augmentation_layer=None,
    )

    model = tf.keras.models.load_model(model_path, compile=False)

    y_true = []
    y_prob = []

    for x, y in ext_ds:
        preds = model.predict(x, verbose=0)
        y_prob.append(preds)
        y_true.append(y.numpy())

    y_prob = np.concatenate(y_prob, axis=0)
    y_true_onehot = np.concatenate(y_true, axis=0)
    y_true_idx = np.argmax(y_true_onehot, axis=1)
    y_pred_idx = np.argmax(y_prob, axis=1)

    metrics = compute_metrics(y_true_idx, y_pred_idx, y_prob, class_names)

    print("\n=== EXTERNAL DATASET REPORT ===")
    print(metrics["report"])
    print("Global AUC:", metrics["auc"])
    print("AUC CI:", metrics["auc_ci"])
    print("F1:", metrics["f1"])

    plot_roc_multi(
        y_true_idx,
        y_prob,
        class_names,
        out_path="plots/roc_external.png",
    )

    return metrics