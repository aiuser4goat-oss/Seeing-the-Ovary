# model/follicle_net.py

from tensorflow.keras import layers, models


def build_follicle_cnn(input_shape=(224, 224, 3)):
    """
    Lightweight CNN focused on small follicle-like structures.
    """
    inp = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)

    model = models.Model(inputs=inp, outputs=x, name="FollicleNet")
    return model