import os
import json
import tensorflow as tf

from model.hybrid_model import build_hybrid_model
from utils.augmentation import build_augmentation_layer
from utils.dataset_loader import build_datasets, compute_class_weights_from_dataset
from analysis.plotting import plot_training_curves


# ==========================================================
# COSINE DECAY LR (safe for Keras)
# ==========================================================
def cosine_lr(step):
    initial_lr = 1e-4
    total_steps = 20000
    cosine_decay = 0.5 * (1 + tf.cos(3.14159 * step / total_steps))
    return initial_lr * cosine_decay


class CosineLRCallback(tf.keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs=None):
        step = tf.cast(self.model.optimizer.iterations, tf.float32)
        lr = cosine_lr(step)
        self.model.optimizer.learning_rate.assign(lr)


# ==========================================================
# TRAIN FUNCTION
# ==========================================================
def train_model(
    stage=1,
    train_dir="data/Train",
    val_dir="data/Val",
    img_size=(224,223),
    batch_size=16,
    epochs_stage1=12,
    epochs_stage2=10
):
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    print("\n======================================")
    print(f"          STARTING TRAINING (STAGE {stage})")
    print("======================================\n")

    # -------------------------
    # DATASETS
    # -------------------------
    aug_layer = build_augmentation_layer()
    train_ds, val_ds, class_names = build_datasets(
        train_dir=train_dir,
        val_dir=val_dir,
        img_size=img_size,
        batch_size=batch_size,
        augmentation_layer=aug_layer
    )

    num_classes = len(class_names)
    class_weights = compute_class_weights_from_dataset(train_ds, num_classes)

    # -------------------------
    # BUILD MODEL
    # -------------------------
    freeze_effnet = True if stage == 1 else False
    model = build_hybrid_model(
        input_shape=(img_size[0], img_size[1], 3),
        num_classes=num_classes,
        eff_trainable=not freeze_effnet
    )

    # -------------------------
    # STAGE 2 LOADS STAGE 1 WEIGHTS
    # -------------------------
    if stage == 2:
        ckpt = "models/stage1_best.weights.h5"
        print(f"Loading Stage 1 weights: {ckpt}")
        model.load_weights(ckpt)

    # -------------------------
    # OPTIMIZER
    # -------------------------
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # -------------------------
    # CALLBACKS
    # -------------------------
    best_path = (
        "models/stage1_best.weights.h5"
        if stage == 1
        else "models/stage2_best.weights.h5"
    )

    callbacks = [
        CosineLRCallback(),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # -------------------------
    # TRAIN
    # -------------------------
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_stage1 if stage == 1 else epochs_stage2,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # -------------------------
    # SAVE HISTORY JSON
    # -------------------------
    hist_path = f"models/history_stage{stage}.json"
    with open(hist_path, "w") as f:
        json.dump(history.history, f)

    # Plot graphs
    plot_training_curves(history, out_path=f"plots/stage{stage}_curves.png")

    print(f"\n=== TRAINING COMPLETE (STAGE {stage}) ===")
    print(f"Saved weights: {best_path}")
    print(f"Saved history: {hist_path}")

    return history, best_path, class_names