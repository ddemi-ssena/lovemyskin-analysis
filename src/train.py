import json, os, cv2, torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================
SPLIT_DIR  = r"data\split"
NUM_CLASSES = 8
BATCH_SIZE  = 8
EPOCHS      = 20
LR          = 1e-4
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ============================================================
# TRANSFORMS
# ============================================================
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

# ============================================================
# DATASET
# ============================================================
class SkinDataset(Dataset):
    def __init__(self, split, transform):
        anno_path = os.path.join(SPLIT_DIR, split, "annotations.json")
        img_dir   = os.path.join(SPLIT_DIR, split, "images")
        with open(anno_path, encoding="utf-8") as f:
            data = json.load(f)

        self.img_dir   = img_dir
        self.transform = transform
        self.cat_ids   = sorted(set(c["id"] for c in data["categories"]))
        self.cat_to_idx = {cid: i for i, cid in enumerate(self.cat_ids)}

        # Her gorsel icin en buyuk annotation'i al (multi-label yerine ana sinif)
        img_to_cats = defaultdict(set)
        for ann in data["annotations"]:
            img_to_cats[ann["image_id"]].add(ann["category_id"])

        self.samples = []
        id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
        for img_id, cats in img_to_cats.items():
            if img_id in id_to_file:
                self.samples.append((id_to_file[img_id], list(cats)[0]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, cat_id = self.samples[idx]
        img_path = os.path.join(self.img_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug = self.transform(image=img)["image"]
        label = self.cat_to_idx[cat_id]
        return aug, label

# ============================================================
# MODEL
# ============================================================
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=NUM_CLASSES)
model = model.to(DEVICE)

# ============================================================
# TRAINING
# ============================================================
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()
scaler    = GradScaler()

train_ds = SkinDataset("train", train_transform)
val_ds   = SkinDataset("val",   val_transform)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

best_val_acc = 0.0

for epoch in range(EPOCHS):
    # --- TRAIN ---
    model.train()
    train_loss, correct, total = 0, 0, 0
    for imgs, labels in train_dl:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            out  = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        correct    += (out.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    train_acc = correct / total * 100

    # --- VAL ---
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with autocast():
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
        torch.save(model.state_dict(), "models/best_model.pt")
        print(f"  >> Model kaydedildi (Val Acc: {val_acc:.1f}%)")

print(f"\nEgitim tamamlandi! En iyi Val Acc: {best_val_acc:.1f}%")