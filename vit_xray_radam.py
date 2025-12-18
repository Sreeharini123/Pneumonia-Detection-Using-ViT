# vit_xray_radam.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
from torch_optimizer import RAdam  # pip install torch-optimizer

# -------------------------- Settings --------------------------
train_path = "chest_xray/train"
val_path = "chest_xray/val"
test_path = "chest_xray/test"

img_size = 224
batch_size = 16
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# -------------------------- Data Transforms --------------------------
train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------- Datasets & Loaders --------------------------
train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
val_dataset = datasets.ImageFolder(val_path, transform=val_test_transform)
test_dataset = datasets.ImageFolder(test_path, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# -------------------------- Model --------------------------
model = timm.create_model("vit_base_patch16_224", pretrained=True)
model.head = nn.Linear(model.head.in_features, 1)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = RAdam([
    {'params': model.patch_embed.parameters(), 'lr': 2e-5, 'weight_decay': 0.01},
    {'params': model.blocks.parameters(), 'lr': 2e-5, 'weight_decay': 0.01},
    {'params': model.head.parameters(), 'lr': 1e-4, 'weight_decay': 0.0}
])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# -------------------------- Training --------------------------
def train_model():
    best_val_acc = 0.0
    patience = 10
    counter = 0

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).int()
            correct += (preds == labels.int()).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # Validation
        model.eval()
        correct, total = 0, 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).int()
                correct += (preds == labels.int()).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        scheduler.step(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] Train Acc: {train_acc*100:.2f}%, Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), "vit_radam_best.pth")
            print("Saved best model!")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered!")
                break

# -------------------------- Evaluation --------------------------
def evaluate_model():
    model.load_state_dict(torch.load("vit_radam_best.pth", map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {acc*100:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes, zero_division=0))

# -------------------------- Main --------------------------
if _name_ == "_main_":
    parser = argparse.ArgumentParser(description="ViT Pneumonia Detection with RAdam")
    parser.add_argument("--mode", type=str, default="train", choices=["train","evaluate"])
    args = parser.parse_args()

    if args.mode == "train":
        train_model()
        evaluate_model()
    else:
        evaluate_model()
