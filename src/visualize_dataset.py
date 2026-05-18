import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

anno_path = r"data\archive (2)\Skin problems.coco\train\_annotations.coco.json"
img_dir   = r"data\archive (2)\Skin problems.coco\train"

with open(anno_path) as f:
    data = json.load(f)

id_to_name = {cat['id']: cat['name'] for cat in data['categories']}

# Her sınıftan 1 örnek göster
samples = {}
img_lookup = {img['id']: img['file_name'] for img in data['images']}

for ann in data['annotations']:
    cid = ann['category_id']
    if cid not in samples:
        samples[cid] = ann

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

for i, (cid, ann) in enumerate(sorted(samples.items())):
    fname = img_lookup[ann['image_id']]
    img_path = os.path.join(img_dir, fname)
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Bounding box çiz
    x, y, w, h = [int(float(v)) for v in ann['bbox']]
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    axes[i].imshow(img)
    axes[i].set_title(id_to_name[cid], fontsize=13, fontweight='bold')
    axes[i].axis('off')

plt.suptitle("Dataset Örnekleri — Her Sınıftan 1 Görsel", fontsize=15)
plt.tight_layout()
plt.savefig("src/dataset_samples.png", dpi=120)
plt.show()
print("Görsel kaydedildi: src/dataset_samples.png")