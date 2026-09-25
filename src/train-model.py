import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torch import nn

# 1. Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 2. Load Dataset
dataset = datasets.ImageFolder(
    "dataset",
    transform=transform
)

print("Classes:", dataset.classes)
print("Total Images:", len(dataset))

# 3. Split Dataset
train_size = int(0.8 * len(dataset))
validation_size = len(dataset) - train_size

train_dataset, validation_dataset = random_split(
    dataset,
    [train_size, validation_size]
)

print("Training images:", len(train_dataset))
print("Validation images:", len(validation_dataset))

# 4. Data Loader
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=8,
    shuffle=False
)

# 5. Load Pretrained Model
model = models.resnet18(weights="DEFAULT")

# 6. Ganti Output Model
model.fc = nn.Linear(
    model.fc.in_features,
    4
)

print("\nModel Ready!")
print("Output classes:", len(dataset.classes))

# 7. Loss Function
criterion = nn.CrossEntropyLoss()

# 8. Optimizer 
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# 9. Training
epochs = 10

for epoch in range(epochs):
    model.train()

    total_loss = 0

    for images, labels in train_loader:
        # Prediksi 
        outputs = model(images)

        # Hitung Error
        loss = criterion(outputs, labels)

        # Bersihkan Gradient Sebelumnya
        optimizer.zero_grad()

        # Backprogation
        loss.backward()

        # Update Weights
        optimizer.step()

        total_loss += loss.item()

    avarage_loss = total_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {avarage_loss:.4f}"
    )

# 10. Validation
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in validation_loader:
        outputs = model(images)

        _, predicted = torch. max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
print(f"\nValidation Accuracy: {accuracy:.2%}")

# 11. Save Model
torch.save(model.state_dict(), "model.pth")

print("Model Saved!")