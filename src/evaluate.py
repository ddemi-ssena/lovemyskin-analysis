import json, os, cv2, torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

SPLIT_DIR   = r"data\split"
MODEL_PATH  = r"models\best_model.pt"
NUM_CLASSES = 8
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["acne", "dark_circles", "eczema", "enlarged_pores",
               "fine_lines", "hyperpigmentation", "rosacea", "wrinkle"]

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])

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
        img = cv2.imread(os.path.join(self.img_dir, fname))
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug = self.transform(image=img)["image"]
        return aug, self.cat_to_idx[cat_id]

# Model yukle
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Test seti
test_ds = SkinDataset("test", transform)
test_dl = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in test_dl:
        imgs = imgs.to(DEVICE)
        out  = model(imgs)
        preds = out.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# Rapor
print("\n=== SINIF BAZLI SONUCLAR ===")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix")
plt.ylabel("Gercek")
plt.xlabel("Tahmin")
plt.tight_layout()
plt.savefig("src/confusion_matrix.png", dpi=120)
plt.show()
print("\nConfusion matrix kaydedildi: src/confusion_matrix.png")