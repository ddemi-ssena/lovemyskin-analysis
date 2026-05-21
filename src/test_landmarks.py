import cv2
import mediapipe as mp
import numpy as np

# Yeni API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

# Model dosyasini indir
model_path = "models/face_landmarker.task"
if not os.path.exists(model_path):
    print("Model indiriliyor...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Indirildi!")

# Landmark detector olustur
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# Test gorseli
img_path = input("Test gorsel yolu: ")
img = cv2.imread(img_path)
if img is None:
    print("Gorsel okunamadi!")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
result  = detector.detect(mp_image)

if not result.face_landmarks:
    print("Yuz bulunamadi!")
    exit()

print(f"Yuz bulundu! {len(result.face_landmarks[0])} landmark tespit edildi.")

# Yuz bolgeleri icin landmark indexleri
REGIONS = {
    "alın":   [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
    "yanak_sol": [116, 123, 147, 213, 192, 214, 210, 211, 32, 208, 199, 428, 262, 431, 432, 436, 433, 416, 376, 352, 345, 372],
    "yanak_sag": [345, 352, 376, 416, 433, 436, 432, 431, 262, 428, 199, 208, 32, 211, 210, 214, 192, 213, 147, 123, 116],
    "burun":  [1, 2, 5, 4, 19, 94, 2, 164, 0, 11, 12, 13, 14, 15, 16, 17, 18],
    "alti_goz": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
}

landmarks = result.face_landmarks[0]
h, w = img.shape[:2]

# Her bolgeyi gorselde goster
img_copy = img.copy()
colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255)]

for (region_name, indices), color in zip(REGIONS.items(), colors):
    points = []
    for idx in indices:
        if idx < len(landmarks):
            lm = landmarks[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            points.append((x, y))

    if points:
        pts = np.array(points, np.int32)
        cv2.polylines(img_copy, [pts], True, color, 2)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(img_copy, region_name, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

cv2.imshow("Yuz Bolgeleri", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()