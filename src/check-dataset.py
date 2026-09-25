from pathlib import Path
from PIL import Image

# Lokasi Dataset
dataset_path = Path("dataset")

# Class yang kita gunakan
classes = ["bottle", "apple", "book", "human"]

print("===== DATASET CHECKER =====")

for class_name in classes:
    class_path = dataset_path / class_name

    # Ambil semua gambar
    images = list(class_path.glob("*.jpg"))
    images += list(class_path.glob("*.jpeg"))
    images += list(class_path.glob("*.png"))

    print(f"\n{class_name}: {len(images)} gambar")

    # Check setiap gambar
    for image_path in images:
        try:
            with Image.open(image_path) as image:
                print(
                    f"  ✓ {image_path.name} "
                    f"-> {image.size} {image.format}"
                )
        except Exception as e:
            print(f"  ✗ {image_path.name} ERROR: {e}")