# train/train_runner.py

import sys
from train.train import train_model


def run_training():
    """
    Handles Stage 1 → Stage 2 automatically.
    Usage:
        python main.py train          (runs Stage 1)
        python main.py train --stage2 (runs Stage 2)
    """

    # Detect if user wants Stage 2 manually
    user_stage2 = ("--stage2" in sys.argv)

    print("\n======================================")
    print("           TRAINING PIPELINE")
    print("======================================\n")

    if user_stage2:
        print(">>> User requested: STAGE 2 (Fine-tuning)\n")
        result = train_model(stage=2)
    else:
        print(">>> Running STAGE 1 (Feature Extraction)\n")
        result = train_model(stage=1)

    # train_model returns:
    #   history, best_path, final_path, class_names
    if isinstance(result, tuple) and len(result) == 4:
        history, best_path, final_path, class_names = result

        print("\n======================================")
        print("         TRAINING COMPLETE")
        print("======================================")
        print("Best model :", best_path)
        print("Final model:", final_path)
        print("Classes    :", class_names)
        print("======================================\n")

        return result

    # If train_model returned only history
    else:
        print("\n======================================")
        print("   TRAINING COMPLETE (history only)")
        print("======================================\n")
        return result