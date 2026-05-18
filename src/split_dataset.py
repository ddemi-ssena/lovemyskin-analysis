import json, os, shutil, random

random.seed(42)

ANNO_PATH  = r"data\merged\annotations.json"
IMGS_DIR   = r"data\merged\images"
OUTPUT_DIR = r"data\split"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

with open(ANNO_PATH, encoding="utf-8") as f:
    data = json.load(f)

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)

image_ids = [img["id"] for img in data["images"]]
random.shuffle(image_ids)

n = len(image_ids)
n_train = int(n * TRAIN_RATIO)
n_val   = int(n * VAL_RATIO)

splits = {
    "train": set(image_ids[:n_train]),
    "val":   set(image_ids[n_train:n_train+n_val]),
    "test":  set(image_ids[n_train+n_val:])
}

id_to_img = {img["id"]: img for img in data["images"]}

for split_name, split_ids in splits.items():
    split_imgs = [id_to_img[i] for i in split_ids]
    split_anns = [a for a in data["annotations"] if a["image_id"] in split_ids]

    for img in split_imgs:
        src = os.path.join(IMGS_DIR, img["file_name"])
        dst = os.path.join(OUTPUT_DIR, split_name, "images", img["file_name"])
        shutil.copy2(src, dst)

    split_coco = {
        "images": split_imgs,
        "annotations": split_anns,
        "categories": data["categories"]
    }
    with open(os.path.join(OUTPUT_DIR, split_name, "annotations.json"), "w", encoding="utf-8") as f:
        json.dump(split_coco, f, indent=2)

    print(f"{split_name}: {len(split_imgs)} gorsel, {len(split_anns)} annotation")