import json, os, shutil

ECZEMA_DIR  = r"data\DATA\train\Eczemaa"
MERGED_ANNO = r"data\merged\annotations.json"
OUTPUT_IMGS = r"data\merged\images"

with open(MERGED_ANNO, encoding="utf-8") as f:
    merged = json.load(f)

# eczema category id bul
eczema_cat_id = None
for cat in merged["categories"]:
    if cat["name"] == "eczema":
        eczema_cat_id = cat["id"]
        break

print(f"eczema category id: {eczema_cat_id}")

max_img_id = max(img["id"] for img in merged["images"])
max_ann_id = max(ann["id"] for ann in merged["annotations"])

img_id_counter = max_img_id + 1
ann_id_counter = max_ann_id + 1
copied = 0

for fname in os.listdir(ECZEMA_DIR):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    src = os.path.join(ECZEMA_DIR, fname)
    new_fname = "ecz_" + fname
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
        "category_id": eczema_cat_id,
        "bbox": [0, 0, 224, 224],
        "area": 224 * 224,
        "iscrowd": 0
    })

    img_id_counter += 1
    ann_id_counter += 1
    copied += 1

with open(MERGED_ANNO, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2)

print(f"Eklenen: {copied} gorsel")

from collections import Counter
id_to_name = {c["id"]: c["name"] for c in merged["categories"]}
counts = Counter(id_to_name[a["category_id"]] for a in merged["annotations"])
print("\nYeni dagilim:")
for k, v in sorted(counts.items()):
    print(f"  {k}: {v}")