# analysis/kfold_training.py
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical

from model.hybrid_model import build_hybrid_model
from analysis.metrics_extended import compute_metrics

# extra safety (Metal off)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_METAL_ENABLE"] = "0"
os.environ["APPLE_ENABLE_METAL"] = "0"


def load_all_images(train_dir="data/Train", img_size=(224, 224)):
    from pathlib import Path
    import cv2

    class_names = sorted(
        [d.name for d in Path(train_dir).iterdir() if d.is_dir()]
    )
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    X, y = [], []

    for cls in class_names:
        folder = Path(train_dir) / cls
        for f in folder.iterdir():
            if not f.is_file():
                continue
            img = cv2.imread(str(f))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(class_to_idx[cls])

    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y, dtype=int)
    return X, y, class_names


def run_kfold(k=5, train_dir="data/Train", save_models=True):
    print("Loading images...")
    X, y, class_names = load_all_images(train_dir=train_dir)
    num_classes = len(class_names)

    print("Detected classes:", class_names)
    print("Loaded", len(X), "images.")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold = 0
    all_results = []

    for train_idx, val_idx in skf.split(X, y):
        fold += 1
        print(f"\n========= FOLD {fold}/{k} =========")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        y_train_cat = to_categorical(y_train, num_classes)
        y_val_cat = to_categorical(y_val, num_classes)

        model = build_hybrid_model(
            input_shape=(224, 224, 3), num_classes=num_classes
        )

        # use legacy Adam to avoid slowdown warnings on M1/M2
        optimizer = tf.keras.optimizers.Adam(1e-4)

        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        ]

        model.fit(
            X_train,
            y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=20,
            batch_size=16,
            callbacks=callbacks,
            verbose=1,
        )

        y_prob = model.predict(X_val)
        y_pred = np.argmax(y_prob, axis=1)

        fold_metrics = compute_metrics(y_val, y_pred, y_prob, class_names)
        all_results.append(fold_metrics)

        print("Fold classification report:\n", fold_metrics["report"])

        if save_models:
            save_dir = "models/kfold_folds"
            os.makedirs(save_dir, exist_ok=True)

            weight_path = f"{save_dir}/fold{fold}.weights.h5"
            model.save_weights(weight_path)
            print(f"Saved: {weight_path}")

    print("\n\n===== K-FOLD SUMMARY =====")
    f1s = [res["f1"] for res in all_results]
    mean_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)
    print("Mean F1:", mean_f1)
    print("F1 CI (mean ± 1σ):", (mean_f1 - std_f1, mean_f1 + std_f1))

    return all_results