# oversampling/malignant_augmentor.py

import os
from pathlib import Path
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator, array_to_img

def augment_malignant(
    malignant_dir="data/Train/Malignant",
    target_total=700,            # final total images in Malignant
    img_size=(224, 224)
):
    """
    Safely augments malignant images directly INTO the Malignant/ folder.
    No subfolders are created. No weird class split. Clean, simple, effective.
    """

    malignant_dir = Path(malignant_dir)
    if not malignant_dir.exists():
        raise ValueError(f"Malignant directory not found: {malignant_dir}")

    # Load only real images (ignore earlier augmented ones)
    real_images = [
        f for f in malignant_dir.iterdir()
        if f.is_file() and not f.name.startswith("aug_")
    ]

    n_real = len(real_images)
    print(f"Found {n_real} real malignant images.")

    # If malignant is already large enough
    if n_real >= target_total:
        print(f"Malignant already has {n_real} images (>= {target_total}). No augmentation needed.")
        return

    to_generate = target_total - n_real
    print(f"Generating {to_generate} augmented malignant images...")

    datagen = ImageDataGenerator(
        rotation_range=12,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.10,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )

    count = 0
    idx = 0

    # Loop until enough synthetic images are created
    while count < to_generate:
        img_path = real_images[idx % n_real]
        idx += 1

        img = load_img(img_path, target_size=img_size)
        arr = img_to_array(img)
        arr = np.expand_dims(arr, axis=0)

        # Generate multiple augmentations per real image
        for _ in range(3):
            if count >= to_generate:
                break

            batch = next(datagen.flow(arr, batch_size=1))
            out_name = f"aug_malignant_{count:04d}.jpg"
            out_path = malignant_dir / out_name

            array_to_img(batch[0]).save(out_path)
            count += 1

    print(f"Created {count} augmented malignant images.")
    print(f"TOTAL malignant images is now ≈ {n_real + count}")
    print("Malignant augmentation complete.")