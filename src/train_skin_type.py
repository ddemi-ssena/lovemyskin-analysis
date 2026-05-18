import os, cv2, torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

BASE_DIR    = r"data\skin_type_classification_dataset"
NUM_CLASSES = 4
BATCH_SIZE  = 16
EPOCHS      = 20
LR          = 1e-4
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["combination", "dry", "normal", "oily"]

print(f"Device: {DEVICE}")

train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.3),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])

class SkinTypeDataset(Dataset):
    def __init__(self, split, transform):
        self.transform = transform
        self.samples   = []
        split_dir = os.path.join(BASE_DIR, split)
        for label_idx, cls in enumerate(CLASS_NAMES):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(cls_dir, fname), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except:
            img = None
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug = self.transform(image=img)["image"]
        return aug, label

train_ds = SkinTypeDataset("train", train_transform)
val_ds   = SkinTypeDataset("valid", val_transform)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=NUM_CLASSES)
model = model.to(DEVICE)

# Weighted loss
counts  = torch.tensor([248, 833, 976, 815], dtype=torch.float)
weights = (1.0 / counts)
weights = weights / weights.sum()
weights = weights.to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler    = torch.amp.GradScaler("cuda")

best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss, correct, total = 0, 0, 0
    for imgs, labels in train_dl:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            out  = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        correct    += (out.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    train_acc = correct / total * 100

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast("cuda"):
                out  = model(imgs)
                loss = criterion(out, labels)
            val_loss    += loss.item()
            val_correct += (out.argmax(1) == labels).sum().item()
            val_total   += labels.size(0)

    val_acc = val_correct / val_total * 100
    scheduler.step()

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Train Loss: {train_loss/len(train_dl):.3f} | Train Acc: {train_acc:.1f}% | "
          f"Val Loss: {val_loss/len(val_dl):.3f} | Val Acc: {val_acc:.1f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/skin_type_model.pt")
        print(f"  >> Model kaydedildi (Val Acc: {val_acc:.1f}%)")

print(f"\nEgitim tamamlandi! En iyi Val Acc: {best_val_acc:.1f}%")