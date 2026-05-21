import json, os, shutil

DARK_ANNO  = r"data\Dark Circles.coco\train\_annotations.coco.json"
DARK_IMGS  = r"data\Dark Circles.coco\train"
MERGED_ANNO = r"data\merged\annotations.json"
OUTPUT_IMGS = r"data\merged\images"

# Mevcut merged dataseti yukle
with open(MERGED_ANNO, encoding="utf-8") as f:
    merged = json.load(f)

# dark_circles category id'sini bul
dark_cat_id = None
for cat in merged["categories"]:
    if cat["name"] == "dark_circles":
        dark_cat_id = cat["id"]
        break

print(f"dark_circles category id: {dark_cat_id}")

# Yeni dataseti yukle
with open(DARK_ANNO, encoding="utf-8") as f:
    dark_data = json.load(f)

# Mevcut max id'leri bul
max_img_id = max(img["id"] for img in merged["images"])
max_ann_id = max(ann["id"] for ann in merged["annotations"])

print(f"Mevcut max image id: {max_img_id}")
print(f"Mevcut max annotation id: {max_ann_id}")

img_id_counter = max_img_id + 1
ann_id_counter = max_ann_id + 1

old_to_new = {}
copied = skipped = 0

for img in dark_data["images"]:
    src = os.path.join(DARK_IMGS, img["file_name"])
    if not os.path.exists(src):
        skipped += 1
        continue

    new_id    = img_id_counter
    img_id_counter += 1
    old_to_new[img["id"]] = new_id
    new_fname = "dc_" + img["file_name"]
    dst = os.path.join(OUTPUT_IMGS, new_fname)
    shutil.copy2(src, dst)
    copied += 1

    merged["images"].append({
        "id": new_id,
        "file_name": new_fname,
        "width": img["width"],
        "height": img["height"]
    })

print(f"Kopyalanan: {copied} | Atlanan: {skipped}")

# Annotation ekle
added = 0
for ann in dark_data["annotations"]:
    if ann["image_id"] not in old_to_new:
        continue
    merged["annotations"].append({
        "id": ann_id_counter,
        "image_id": old_to_new[ann["image_id"]],
        "category_id": dark_cat_id,
        "bbox": ann["bbox"],
        "area": float(ann["bbox"][2]) * float(ann["bbox"][3]),
        "iscrowd": 0
    })
    ann_id_counter += 1
    added += 1

print(f"Eklenen annotation: {added}")

# Kaydet
with open(MERGED_ANNO, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2)

# Yeni dagilim
from collections import Counter
id_to_name = {c["id"]: c["name"] for c in merged["categories"]}
counts = Counter(id_to_name[a["category_id"]] for a in merged["annotations"])
print("\nYeni dagilim:")
for k, v in sorted(counts.items()):
    print(f"  {k}: {v}")