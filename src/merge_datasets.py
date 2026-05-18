import json, os, shutil

DATASET1_ANNO = r"data\archive (2)\Skin Problems.coco\train\_annotations.coco.json"
DATASET1_IMGS = r"data\archive (2)\Skin Problems.coco\train"
DATASET2_ANNO = r"data\Skin Problems.coco\train\_annotations.coco.json"
DATASET2_IMGS = r"data\Skin Problems.coco\train"
OUTPUT_ANNO   = r"data\merged\annotations.json"
OUTPUT_IMGS   = r"data\merged\images"

LABEL_MAP = {
    "blackhead": "acne", "comedonical-acne": "acne",
    "pustule-acne": "acne", "pimples": "acne",
    "rosacea": "rosacea", "eczema": "eczema",
    "wrinkle": "wrinkle", "wrinkles": "wrinkle",
    "hyperpigmentation": "hyperpigmentation",
    "enlarged_pores": "enlarged_pores",
    "fine_lines": "fine_lines",
    "dark_circles": "dark_circles",
    "skin-problems": None, "classes": None,
}

os.makedirs(OUTPUT_IMGS, exist_ok=True)

final_classes = sorted(set(v for v in LABEL_MAP.values() if v))
cat_name_to_id = {n: i+1 for i, n in enumerate(final_classes)}

merged_images, merged_annotations = [], []
img_id_counter, ann_id_counter = 1, 1

def process(anno_path, img_dir, prefix):
    global img_id_counter, ann_id_counter
    with open(anno_path, encoding="utf-8") as f:
        data = json.load(f)
    old_cat = {c['id']: c['name'] for c in data['categories']}
    old_to_new = {}
    copied = skipped = 0
    for img in data['images']:
        src = os.path.join(img_dir, img['file_name'])
        if not os.path.exists(src):
            skipped += 1
            continue
        new_id = img_id_counter
        img_id_counter += 1
        old_to_new[img['id']] = new_id
        new_fname = prefix + img['file_name']
        shutil.copy2(src, os.path.join(OUTPUT_IMGS, new_fname))
        merged_images.append({"id": new_id, "file_name": new_fname,
                               "width": img['width'], "height": img['height']})
        copied += 1
    print(f"  Copied: {copied} | Skipped: {skipped}")
    for ann in data['annotations']:
        if ann['image_id'] not in old_to_new:
            continue
        new_cat = LABEL_MAP.get(old_cat.get(ann['category_id'], ""))
        if not new_cat:
            continue
        merged_annotations.append({
            "id": ann_id_counter, "image_id": old_to_new[ann['image_id']],
            "category_id": cat_name_to_id[new_cat],
            "bbox": ann['bbox'],
            "area": ann.get('area', float(ann['bbox'][2]) * float(ann['bbox'][3])),
            "iscrowd": 0
        })
        ann_id_counter += 1

print("Dataset 1...")
process(DATASET1_ANNO, DATASET1_IMGS, "d1_")
print("Dataset 2...")
process(DATASET2_ANNO, DATASET2_IMGS, "d2_")

final = {
    "images": merged_images,
    "annotations": merged_annotations,
    "categories": [{"id": cid, "name": n} for n, cid in cat_name_to_id.items()]
}
with open(OUTPUT_ANNO, "w", encoding="utf-8") as f:
    json.dump(final, f, indent=2)

print(f"\nDone!")
print(f"Total images     : {len(merged_images)}")
print(f"Total annotations: {len(merged_annotations)}")
print(f"Classes          : {list(cat_name_to_id.keys())}")