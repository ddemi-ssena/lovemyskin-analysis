import json, os, shutil

MERGED_ANNO = r"data\merged\annotations.json"
OUTPUT_IMGS = r"data\merged\images"

EXTRA_CLASSES = {
    "eczema":  r"data\DATA\train\Eczemaa",
    "rosacea": r"data\DATA\train\Rosacea",
    "acne":    r"data\DATA\train\Acne",
}

with open(MERGED_ANNO, encoding="utf-8") as f:
    merged = json.load(f)

cat_name_to_id = {c["name"]: c["id"] for c in merged["categories"]}
print("Kategoriler:", cat_name_to_id)

max_img_id = max(img["id"] for img in merged["images"])
max_ann_id = max(ann["id"] for ann in merged["annotations"])
img_id_counter = max_img_id + 1
ann_id_counter = max_ann_id + 1

for class_name, folder in EXTRA_CLASSES.items():
    cat_id = cat_name_to_id.get(class_name)
    if cat_id is None:
        print(f"UYARI: {class_name} kategorisi bulunamadi, atlaniyor")
        continue

    copied = 0
    for fname in os.listdir(folder):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        src = os.path.join(folder, fname)
        new_fname = f"extra_{class_name}_{fname}"
        dst = os.path.join(OUTPUT_IMGS, new_fname)
        shutil.copy2(src, dst)

        merged["images"].append({
            "id": img_id_counter,
            "file_name": new_fname,
            "width": 224,
            "height": 224
        })
        merged["annotations"].append({
            "id": ann_id_counter,
            "image_id": img_id_counter,
            "category_id": cat_id,
            "bbox": [0, 0, 224, 224],
            "area": 224 * 224,
            "iscrowd": 0
        })

        img_id_counter += 1
        ann_id_counter += 1
        copied += 1

    print(f"{class_name}: {copied} gorsel eklendi")

with open(MERGED_ANNO, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2)

from collections import Counter
id_to_name = {c["id"]: c["name"] for c in merged["categories"]}
counts = Counter(id_to_name[a["category_id"]] for a in merged["annotations"])
print("\nFinal dagilim:")
for k, v in sorted(counts.items()):
    print(f"  {k}: {v}")