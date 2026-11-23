# model/fuzzy_layers.py

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="OvarianAI")
class FuzzyConv(layers.Layer):
    """
    Simplified fuzzy convolution layer:
    - Learns membership-like masks via sigmoid
    - Multiplies them with a standard Conv2D
    """

    def __init__(self, filters, kernel_size, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation

        self.conv = layers.Conv2D(filters, kernel_size, padding="same", use_bias=False)
        self.membership = layers.Conv2D(filters, kernel_size, padding="same", activation="sigmoid")
        self.act = layers.Activation(activation)

    def call(self, inputs, **kwargs):
        base = self.conv(inputs)
        mu = self.membership(inputs)  # membership map in [0,1]
        x = base * mu
        return self.act(x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "activation": self.activation,
            }
        )
        return cfg