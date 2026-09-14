import os
import shutil
from pathlib import Path
from PIL import Image

# ============================================================
# AIR DEFENSE DATASET
# YOLO OBJECT DETECTION -> CLASSIFICATION DATASET
# ============================================================

# Dataset location
BASE_DIR = Path(__file__).resolve().parent / "Air_Defense_Object_Detection_Dataset_(YOLOv8)"

# Output classification dataset
OUTPUT_DIR = Path(__file__).resolve().parent / "classification_dataset"

# Class names from data.yaml
CLASS_NAMES = [
    "f16",
    "helicopter",
    "quadcopter",
    "rocket"
]

# YOLO image extensions
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def find_image(image_dir, stem):
    """Find image corresponding to a label file."""
    for ext in IMAGE_EXTENSIONS:
        image_path = image_dir / (stem + ext)
        if image_path.exists():
            return image_path
    return None


def convert_split(split):
    print("\n" + "=" * 60)
    print(f"PROCESSING: {split.upper()}")
    print("=" * 60)

    image_dir = BASE_DIR / "images" / split
    label_dir = BASE_DIR / "labels" / split

    if not image_dir.exists():
        print(f"ERROR: Image directory not found: {image_dir}")
        return

    if not label_dir.exists():
        print(f"ERROR: Label directory not found: {label_dir}")
        return

    # Create class directories
    for class_name in CLASS_NAMES:
        output_class_dir = OUTPUT_DIR / split / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

    label_files = list(label_dir.glob("*.txt"))

    print(f"Label files found: {len(label_files)}")

    total_crops = 0
    skipped = 0

    for index, label_file in enumerate(label_files, start=1):

        # Find matching image
        image_path = find_image(image_dir, label_file.stem)

        if image_path is None:
            print(f"WARNING: Image not found for {label_file.name}")
            skipped += 1
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            image_width, image_height = image.size

            # Read YOLO annotations
            with open(label_file, "r") as f:
                lines = [line.strip() for line in f if line.strip()]

            object_number = 0

            for line in lines:

                parts = line.split()

                if len(parts) < 5:
                    continue

                # Handles both "2" and "2.0"
                class_id = int(float(parts[0]))

                if class_id < 0 or class_id >= len(CLASS_NAMES):
                    continue

                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # YOLO normalized coordinates -> pixel coordinates
                x_center *= image_width
                y_center *= image_height
                width *= image_width
                height *= image_height

                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)

                # Keep crop inside image boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image_width, x2)
                y2 = min(image_height, y2)

                # Ignore invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                # Crop object
                crop = image.crop((x1, y1, x2, y2))

                class_name = CLASS_NAMES[class_id]

                output_dir = OUTPUT_DIR / split / class_name

                # Unique filename
                output_filename = (
                    f"{label_file.stem}_object_{object_number}.jpg"
                )

                output_path = output_dir / output_filename

                crop.save(output_path, quality=95)

                total_crops += 1
                object_number += 1

        except Exception as e:
            print(f"ERROR processing {label_file.name}: {e}")
            skipped += 1

        # Progress
        if index % 100 == 0:
            print(
                f"Processed {index}/{len(label_files)} "
                f"| Crops created: {total_crops}"
            )

    print(f"\nCompleted {split}")
    print(f"Crops created: {total_crops}")
    print(f"Skipped files: {skipped}")


# ============================================================
# MAIN
# ============================================================

print("\nAIR DEFENSE CLASSIFICATION DATASET CREATION")
print("=" * 60)

if not BASE_DIR.exists():
    print("\nERROR: Dataset folder not found!")
    print(f"Expected location:\n{BASE_DIR}")
    print("\nPlease check the dataset folder name.")
    exit()

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Process all three splits
for split in ["train", "val", "test"]:
    convert_split(split)

print("\n" + "=" * 60)
print("CONVERSION COMPLETED!")
print("=" * 60)

print(f"\nClassification dataset created at:")
print(OUTPUT_DIR)

print("\nExpected structure:")
print("""
classification_dataset/
├── train/
│   ├── f16/
│   ├── helicopter/
│   ├── quadcopter/
│   └── rocket/
├── val/
│   ├── f16/
│   ├── helicopter/
│   ├── quadcopter/
│   └── rocket/
└── test/
    ├── f16/
    ├── helicopter/
    ├── quadcopter/
    └── rocket/
""")