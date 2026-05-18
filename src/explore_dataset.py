import json
import os
from collections import Counter

# Annotation dosyasını oku
anno_path = r"data\archive (2)\Skin problems.coco\train\_annotations.coco.json"

with open(anno_path, "r") as f:
    data = json.load(f)

# Kaç görsel var?
print(f"Toplam görsel sayısı: {len(data['images'])}")

# Hangi sınıflar var?
print(f"\nSınıflar:")
for cat in data['categories']:
    print(f"  ID {cat['id']}: {cat['name']}")

# Her sınıftan kaç annotation var?
class_counts = Counter()
for ann in data['annotations']:
    class_counts[ann['category_id']] += 1

print(f"\nSınıf dağılımı:")
id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
for cat_id, count in sorted(class_counts.items()):
    print(f"  {id_to_name[cat_id]}: {count} annotation")