# utils/gradcam.py
import numpy as np
import tensorflow as tf
import cv2


def make_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    """
    img_array: shape (1, H, W, 3), scaled to [0,1]
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original_img_rgb, heatmap, alpha=0.4):
    """
    original_img_rgb: uint8 RGB image (H,W,3)
    heatmap: 2D float array in [0,1]
    """
    heatmap = cv2.resize(heatmap, (original_img_rgb.shape[1], original_img_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(heatmap_color, alpha, original_img_rgb, 1 - alpha, 0)
    return overlay