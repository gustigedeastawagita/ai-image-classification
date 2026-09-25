from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

dataset_path = Path("dataset")

classes = ["bottle", "apple", "book", "human"]

fig, axes = plt.subplots(4, 4, figsize=(12, 12))

for row, class_name in enumerate(classes):
    class_path = dataset_path / class_name

    images = []
    images += list(class_path.glob("*jpg"))
    images += list(class_path.glob("*jpeg"))
    images += list(class_path.glob("*png"))

    # Ambil maksimal 3 gambar
    images = images[:4]

    for col, image_path in enumerate(images):
        image = Image.open(image_path)

        axes[row, col].imshow(image)
        axes[row, col].set_title(class_name)
        axes[row, col].axis("off")

plt.tight_layout()
plt.show()