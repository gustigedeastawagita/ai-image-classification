import torch
from PIL import Image
from torchvision import transforms, models, datasets
from torch import nn


# 1. Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# 2. Ambil Nama Class
dataset = datasets.ImageFolder("dataset")

classes = dataset.classes

print("Classes:", classes)


# 3. Buat Arsitektur ResNet18
model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(classes)
)


# 4. Load Model Yang Sudah Di Training
model.load_state_dict(
    torch.load(
        "model.pth",
        map_location="cpu"
    )
)

model.eval()


# 5. Prediction Function
def predict_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")


    # Preprocessing
    image = transform(image)


    # Tambahkan Batch Dimension
    image = image.unsqueeze(0)


    # Prediction
    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )


    # Hasil Prediction
    predicted_class = classes[
        predicted.item()
    ]


    confidence = confidence.item() * 100


    return predicted_class, confidence