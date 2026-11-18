from pathlib import Path
import random
import shutil

SOURCE_DIR = Path(
    r"D:\workspace\smoke_siglip2_no_labels_clean\smoke_siglip2_no_labels_clean\SmolVLM_500_Test\SmolVLM_500_Test"
)

OUTPUT_DIR = Path(
    r"D:\workspace\smoke_siglip2_no_labels_clean\smoke_siglip2_no_labels_clean\SigLIP2_dataset"
)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

CLASSES = [
    "black_smoke",
    "cloud",
    "steam",
    "normal_background",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(SEED)


def copy_images(files, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        shutil.copy2(file, target_dir / file.name)


for class_name in CLASSES:
    class_dir = SOURCE_DIR / class_name

    if not class_dir.exists():
        print(f"[ERROR] 找不到資料夾: {class_dir}")
        continue

    files = [
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    random.shuffle(files)

    n = len(files)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    copy_images(train_files, OUTPUT_DIR / "train" / class_name)
    copy_images(val_files, OUTPUT_DIR / "val" / class_name)
    copy_images(test_files, OUTPUT_DIR / "test" / class_name)

    print(f"\n{class_name}")
    print(f"Total : {n}")
    print(f"Train : {len(train_files)}")
    print(f"Val   : {len(val_files)}")
    print(f"Test  : {len(test_files)}")

print("\nDataset split completed.")
print(f"Output directory: {OUTPUT_DIR}")
