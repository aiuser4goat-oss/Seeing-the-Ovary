#!/usr/bin/env python3
import os
import sys

# Dispatch the continuation before legacy CPU-only environment settings.
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "continuation":
    from continuation.cli import main as continuation_main
    continuation_main(sys.argv[2:])
    raise SystemExit(0)

# ------------------------------------------------------------------
# Environment: force CPU-only on M1/M2 and disable Metal
# ------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_METAL_ENABLE"] = "0"
os.environ["APPLE_ENABLE_METAL"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
from oversampling.malignant_augmentor import augment_malignant
from analysis.kfold_training import run_kfold
from train.train_runner import run_training
from train.evaluate import run_eval
from analysis.gradcam_runner import run_gradcam
from analysis.external_validate import run_external_validation


def print_usage():
    print("\nUsage:")
    print("  python main.py oversample             # augment malignant class")
    print("  python main.py kfold                  # 5-fold stratified CV")
    print("  python main.py train                  # train final model")
    print("  python main.py eval                   # evaluate on Val")
    print("  python main.py gradcam <img>          # Grad-CAM for a single image")
    print("  python main.py external               # external dataset validation")
    print()


def main():
    if len(sys.argv) < 2:
        print("Error: Missing command.\n")
        print_usage()
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "oversample":
        print("\n=== OVERSAMPLING MALIGNANT DATA ===\n")
        augment_malignant()

    elif cmd == "kfold":
        print("\n=== RUNNING STRATIFIED 5-FOLD CROSS-VALIDATION ===\n")
        run_kfold()

    elif cmd == "train":
        print("\n=== TRAINING FINAL HYBRID MODEL ===\n")
        run_training()

    elif cmd == "eval":
        print("\n=== EVALUATING MODEL ON VALIDATION SET ===\n")
        run_eval()

    elif cmd == "gradcam":
        if len(sys.argv) < 3:
            print("Error: No image path provided.")
            print("Usage: python main.py gradcam path/to/image.jpg")
            sys.exit(1)
        image_path = sys.argv[2]
        print(f"\n=== GENERATING GRAD-CAM FOR: {image_path} ===\n")
        run_gradcam(image_path=image_path)

    elif cmd == "external":
        print("\n=== EXTERNAL DATASET VALIDATION ===\n")
        run_external_validation()

    else:
        print(f"Error: Unknown command '{cmd}'.\n")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
