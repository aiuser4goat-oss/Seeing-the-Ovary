import tensorflow as tf
from tensorflow.keras import layers, models
from model.fuzzy_layers import FuzzyConv
from model.follicle_net import build_follicle_cnn
from model.efficientnet_model import build_efficientnet_backbone


def build_hybrid_model(input_shape=(224, 224, 3), num_classes=4, eff_trainable=False, weights="imagenet"):
    inp = layers.Input(shape=input_shape)

    # ----- FUZZY CNN BRANCH -----
    fx = FuzzyConv(32, 3, activation="relu")(inp)
    fx = layers.BatchNormalization()(fx)
    fx = layers.MaxPooling2D()(fx)

    fx = FuzzyConv(64, 3, activation="relu")(fx)
    fx = layers.BatchNormalization()(fx)
    fx = layers.MaxPooling2D()(fx)

    fx = FuzzyConv(128, 3, activation="relu")(fx)
    fx = layers.BatchNormalization()(fx)
    fx = layers.GlobalAveragePooling2D()(fx)
    fuzzy_feat = layers.Dense(128, activation="relu")(fx)

    # ----- FOLLICLE BRANCH -----
    follicle = build_follicle_cnn(input_shape)
    follicle_feat = follicle(inp)

    # ----- EFFICIENTNET -----
    eff = build_efficientnet_backbone(
        input_shape,
        trainable_backbone=eff_trainable,
        weights=weights
    )

    eff_feat = eff(inp)

    # ----- CONCAT -----
    merged = layers.Concatenate()([fuzzy_feat, follicle_feat, eff_feat])

    x = layers.Dense(512, activation="relu")(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inp, outputs=out)
    return model
