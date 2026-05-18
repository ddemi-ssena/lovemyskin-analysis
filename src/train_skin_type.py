import os, cv2, torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

IMG_DIR    = r"data\Skin types.multiclass\train"
CSV_PATH   = r"data\Skin types.multiclass\train\_classes.csv"
NUM_CLASSES = 4
BATCH_SIZE  = 8
EPOCHS      = 30
LR          = 1e-4
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["combination", "dry", "normal", "oily"]

print(f"Device: {DEVICE}")

# Train transform'u guclendir
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.HueSaturationValue(p=0.3),
    A.GaussNoise(p=0.2),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])

class SkinTypeDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        fname = row["filename"]
        label = int(np.argmax(row[CLASS_NAMES].values.astype(float)))
    
        # Unicode path sorunu icin
        img_path = os.path.join(self.img_dir, fname)
        try:
            img = cv2.imdecode(
                np.fromfile(img_path, dtype=np.uint8), 
                cv2.IMREAD_COLOR
            )
        except:
            img = None
        
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug = self.transform(image=img)["image"]
        return aug, label

df = pd.read_csv(CSV_PATH)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42,
                                     stratify=df[CLASS_NAMES].idxmax(axis=1))

train_ds = SkinTypeDataset(train_df, IMG_DIR, train_transform)
val_ds   = SkinTypeDataset(val_df,   IMG_DIR, val_transform)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# Modele dropout ekle
model = timm.create_model("efficientnet_b0", pretrained=True, 
                           num_classes=NUM_CLASSES, drop_rate=0.4)
model = model.to(DEVICE)

# Learning rate'i duşur, weight decay ekle
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()
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