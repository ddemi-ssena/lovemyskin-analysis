import cv2
import numpy as np
import torch
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

PROBLEM_CLASSES = ["acne", "dark_circles", "eczema", "enlarged_pores",
                   "fine_lines", "hyperpigmentation", "rosacea", "wrinkle"]

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])

# Landmark indexleri — her bolge icin
REGIONS = {
    "alin":      [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                  361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                  176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162,
                  21, 54, 103, 67, 109],
    "yanak_sol": [116, 123, 147, 213, 192, 214, 210, 211, 32, 208,
                  199, 428, 262, 431, 432, 436, 433, 416, 376, 352, 345],
    "yanak_sag": [345, 352, 376, 416, 433, 436, 432, 431, 262, 428,
                  199, 208, 32, 211, 210, 214, 192, 213, 147, 123, 116],
    "burun":     [1, 2, 5, 4, 19, 94, 164, 0, 11, 12, 13, 14, 15, 16, 17, 18],
    "goz_alti":  [362, 382, 381, 380, 374, 373, 390, 249, 263,
                  466, 388, 387, 386, 385, 384, 398]
}

def load_model(path, num_classes):
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model.to(DEVICE)

def get_region_bbox(landmarks, indices, h, w, padding=10):
    points = []
    for idx in indices:
        if idx < len(landmarks):
            lm = landmarks[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            points.append((x, y))
    if not points:
        return None
    pts  = np.array(points)
    x1   = max(0, pts[:,0].min() - padding)
    y1   = max(0, pts[:,1].min() - padding)
    x2   = min(w, pts[:,0].max() + padding)
    y2   = min(h, pts[:,1].max() + padding)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return (x1, y1, x2, y2)

def predict_region(model, img_rgb, bbox):
    x1, y1, x2, y2 = bbox
    crop   = img_rgb[y1:y2, x1:x2]
    tensor = transform(image=crop)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out   = torch.softmax(model(tensor), dim=1)[0]
        idx   = out.argmax().item()
        conf  = round(out[idx].item() * 100, 1)
        scores = {PROBLEM_CLASSES[i]: round(out[i].item()*100, 1)
                  for i in range(len(PROBLEM_CLASSES))}
    return {
        "main": PROBLEM_CLASSES[idx],
        "conf": conf,
        "scores": scores
    }

class RegionAnalyzer:
    def __init__(self):
        model_path = "models/face_landmarker.task"
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options, num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.model    = load_model("models/best_model.pt", 8)
        print("RegionAnalyzer hazir!")

    def analyze(self, img_path):
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Gorsel okunamadi"}

        img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w     = img.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result   = self.detector.detect(mp_image)

        if not result.face_landmarks:
            return {"error": "Yuz bulunamadi!"}

        landmarks    = result.face_landmarks[0]
        region_results = {}

        for region_name, indices in REGIONS.items():
            bbox = get_region_bbox(landmarks, indices, h, w)
            if bbox is None:
                continue
            region_results[region_name] = predict_region(self.model, img_rgb, bbox)

        return {"regions": region_results}


if __name__ == "__main__":
    analyzer = RegionAnalyzer()
    path     = input("Gorsel yolu: ")
    result   = analyzer.analyze(path)

    if "error" in result:
        print(result["error"])
    else:
        print("\n=== BOLGE ANALIZI ===")
        for region, data in result["regions"].items():
            print(f"\n{region.upper()}:")
            print(f"  Ana sorun: {data['main']} (%{data['conf']})")
            top3 = sorted(data['scores'].items(), key=lambda x: -x[1])[:3]
            for prob, score in top3:
                print(f"  {prob}: %{score}")