# utils/augmentation.py
import tensorflow as tf
from tensorflow.keras import layers


def build_augmentation_layer():
    """
    Keras preprocessing pipeline applied on-the-fly to training batches.
    """
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.15),
        ],
        name="augmentation"
    )