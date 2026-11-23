# analysis/gradcam_runner.py
import cv2
import numpy as np
import tensorflow as tf

from utils.gradcam import make_gradcam_heatmap, overlay_heatmap


def run_gradcam(
    image_path,
    model_path="models/best_model.h5",
    last_conv_layer_name="top_conv",  # EfficientNetB0 conv layer
    output_path="gradcam_output.jpg",
):
    if image_path is None:
        raise ValueError("image_path must be provided")

    print("Loading model from:", model_path)
    model = tf.keras.models.load_model(model_path, compile=False)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_arr = np.expand_dims(img_resized, axis=0) / 255.0

    print("Computing Grad-CAM heatmap...")
    heatmap = make_gradcam_heatmap(model, img_arr, last_conv_layer_name)
    overlay = overlay_heatmap(img_rgb, heatmap)

    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print("Grad-CAM saved to:", output_path)

    return overlay