# utils/losses.py
import tensorflow as tf
from tensorflow.keras import backend as K

class CategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=None, name="categorical_focal_loss"):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        cross_entropy = -y_true * K.log(y_pred)
        focal_weight = K.pow(1 - y_pred, self.gamma)

        if self.alpha is not None:
            alpha_factor = y_true * self.alpha
        else:
            alpha_factor = 1.0

        loss = alpha_factor * focal_weight * cross_entropy
        return K.mean(K.sum(loss, axis=-1))