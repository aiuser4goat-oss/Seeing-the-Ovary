# analysis/eval_runner.py

import numpy as np
import tensorflow as tf

from utils.dataset_loader import build_datasets
from analysis.metrics_extended import compute_metrics
from model.hybrid_model import build_hybrid_model

def run_eval(
    model_path="models/best_model.h5",
    val_dir="data/Val",
    img_size=(224, 224)
):
    """
    Evaluates the final trained model on the validation/test set.
    Computes classification report, confusion matrix, AUC, CI, etc.
    """

    print("\n=== LOADING SAVED MODEL ===")
    model = tf.keras.models.load_model(model_path, compile=False)

    print("\n=== LOADING DATASET ===")
    _, val_ds, class_names = build_datasets(
        val_dir=val_dir,
        train_dir=None,   # not needed
        img_size=img_size,
        batch_size=16,
        augmentation=None
    )

    print("\n=== RUNNING INFERENCE ===")
    all_labels = []
    all_preds = []
    all_probs = []

    for x_batch, y_batch in val_ds:
        probs = model.predict(x_batch, verbose=0)
        preds = np.argmax(probs, axis=1)
        labels = np.argmax(y_batch.numpy(), axis=1)

        all_labels.extend(labels)
        all_preds.extend(preds)
        all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    print("\n=== COMPUTING METRICS ===")
    metrics = compute_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        y_prob=all_probs,
        class_names=class_names
    )

    print("\n=== FINAL EVALUATION REPORT ===\n")
    print(metrics["report"])
    print("\nConfusion Matrix:\n", metrics["confusion_matrix"])
    print("\nAUC:", metrics["auc"])
    print("AUC 95% CI:", metrics["auc_ci"])
    print("Weighted F1:", metrics["f1"])

    return metrics


if __name__ == "__main__":
    run_eval()