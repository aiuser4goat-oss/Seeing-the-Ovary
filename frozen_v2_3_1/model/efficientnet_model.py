# model/efficientnet_model.py

import tensorflow as tf
from tensorflow.keras import layers, models


def build_efficientnet_backbone(input_shape=(224, 224, 3), trainable_backbone=False, weights="imagenet"):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        pooling=None  # we will pool manually
    )
    base.trainable = trainable_backbone
    inp = layers.Input(shape=input_shape)
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D(name="effnet_global_pool")(x)
    return models.Model(inputs=inp, outputs=x, name="EffB0_Backbone")


def build_efficientnet_model(input_shape=(224, 224, 3), num_classes=4):
    backbone = build_efficientnet_backbone(input_shape, trainable_backbone=False)
    inp = backbone.input
    x = backbone.output
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs=inp, outputs=out, name="EffB0_Classifier")
