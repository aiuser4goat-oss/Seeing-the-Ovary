# utils/dataset_loader.py
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight


def build_datasets(
    train_dir="data/Train",
    val_dir="data/Val",
    img_size=(224, 224),
    batch_size=16,
    augmentation_layer=None,
):
    """
    Builds train & val datasets from folders.

    Returns:
      train_ds, val_ds, class_names
    """
    seed = 42

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    class_names = train_ds.class_names
    print("Detected class order:", class_names)

    if augmentation_layer is not None:
        train_ds = train_ds.map(
            lambda x, y: (augmentation_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


def compute_class_weights_from_dataset(train_ds, num_classes):
    """
    Scan the training dataset once to estimate class weights.
    """
    labels = []

    for _, y in train_ds.unbatch():
        idx = tf.argmax(y, axis=-1).numpy()
        labels.append(idx)

    labels = np.array(labels, dtype=int)
    classes = np.arange(num_classes)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )

    class_weights = {i: float(w) for i, w in enumerate(weights)}
    print("Computed class weights:", class_weights)
    return class_weights