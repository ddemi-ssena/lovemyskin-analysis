import tkinter as tk
from tkinter import filedialog
import cv2, torch
import numpy as np
from PIL import Image, ImageTk
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading
from dotenv import load_dotenv
import os

load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

PROBLEM_CLASSES   = ["acne", "dark_circles", "eczema", "enlarged_pores",
                      "fine_lines", "hyperpigmentation", "rosacea", "wrinkle"]
SKIN_TYPE_CLASSES = ["combination", "dry", "normal", "oily"]

REGION_TR = {
    "alin": "Alın",
    "yanak_sol": "Sol Yanak",
    "yanak_sag": "Sağ Yanak",
    "burun": "Burun",
    "goz_alti": "Göz Altı"
}

PROBLEM_TR = {
    "acne": "Akne",
    "dark_circles": "Göz Altı Morluğu",
    "eczema": "Egzama",
    "enlarged_pores": "Büyük Gözenek",
    "fine_lines": "İnce Çizgi",
    "hyperpigmentation": "Hiperpigmentasyon",
    "rosacea": "Rozasea",
    "wrinkle": "Kırışıklık"
}

SKIN_TYPE_TR = {
    "combination": "Karma",
    "dry": "Kuru",
    "normal": "Normal",
    "oily": "Yağlı"
}

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])

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

REGION_VALID_PROBLEMS = {
    "alin":      ["acne", "fine_lines", "wrinkle", "hyperpigmentation", "enlarged_pores"],
    "yanak_sol": ["acne", "rosacea", "hyperpigmentation", "enlarged_pores", "eczema"],
    "yanak_sag": ["acne", "rosacea", "hyperpigmentation", "enlarged_pores", "eczema"],
    "burun":     ["acne", "enlarged_pores", "rosacea", "hyperpigmentation"],
    "goz_alti":  ["dark_circles", "fine_lines", "wrinkle", "eczema"]
}

def load_model(path, num_classes):
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model.to(DEVICE)

print("Modeller yukleniyor...")
problem_model   = load_model("models/best_model.pt", 8)
skin_type_model = load_model("models/skin_type_model.pt", 4)

base_options = python.BaseOptions(model_asset_path="models/face_landmarker.task")
options      = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector     = vision.FaceLandmarker.create_from_options(options)
print("Hazir!")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def get_region_bbox(landmarks, indices, h, w, padding=10):
    points = []
    for idx in indices:
        if idx < len(landmarks):
            lm = landmarks[idx]
            points.append((int(lm.x * w), int(lm.y * h)))
    if not points:
        return None
    pts = np.array(points)
    x1  = max(0, pts[:,0].min() - padding)
    y1  = max(0, pts[:,1].min() - padding)
    x2  = min(w, pts[:,0].max() + padding)
    y2  = min(h, pts[:,1].max() + padding)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return (x1, y1, x2, y2)

def predict_region(img_rgb, bbox, region_name):
    x1, y1, x2, y2 = bbox
    crop   = img_rgb[y1:y2, x1:x2]
    tensor = transform(image=crop)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = torch.softmax(problem_model(tensor), dim=1)[0]

    # Sadece bu bolge icin gecerli siniflar
    valid = REGION_VALID_PROBLEMS.get(region_name, PROBLEM_CLASSES)
    valid_indices = [PROBLEM_CLASSES.index(p) for p in valid]
    
    best_idx  = max(valid_indices, key=lambda i: out[i].item())
    best_conf = round(out[best_idx].item() * 100, 1)
    
    return PROBLEM_CLASSES[best_idx], best_conf

def predict(img_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Gorsel okunamadi"}

    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w     = img.shape[:2]

    # Yuz tespiti
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result   = detector.detect(mp_image)

    if not result.face_landmarks:
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) == 0:
            return {"error": "Yuzunuzun gorundugu bir fotograf yukleyin!"}
        x, y, fw, fh = max(faces, key=lambda f: f[2]*f[3])
        face_img = img_rgb[y:y+fh, x:x+fw]
        tensor   = transform(image=face_img)["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            type_out = torch.softmax(skin_type_model(tensor), dim=1)[0]
            prob_out = torch.softmax(problem_model(tensor), dim=1)[0]
        return {
            "skin_type": SKIN_TYPE_CLASSES[type_out.argmax().item()],
            "type_conf": round(type_out.max().item()*100, 1),
            "main_problem": PROBLEM_CLASSES[prob_out.argmax().item()],
            "regions": {}
        }

    landmarks = result.face_landmarks[0]

    # Cilt tipi — tam yuz
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda f: f[2]*f[3])
        face_img = img_rgb[y:y+fh, x:x+fw]
    else:
        face_img = img_rgb

    tensor = transform(image=face_img)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        type_out = torch.softmax(skin_type_model(tensor), dim=1)[0]

    # Bolge analizi
    region_results = {}
    for region_name, indices in REGIONS.items():
        bbox = get_region_bbox(landmarks, indices, h, w)
        if bbox:
            problem, conf = predict_region(img_rgb, bbox, region_name)
            region_results[region_name] = (problem, conf)

    return {
        "skin_type": SKIN_TYPE_CLASSES[type_out.argmax().item()],
        "type_conf": round(type_out.max().item()*100, 1),
        "regions": region_results
    }

def browse_and_predict():
    path = filedialog.askopenfilename(
        filetypes=[("Gorsel", "*.jpg *.jpeg *.png")])
    if not path:
        return

    pil_img = Image.open(path).resize((260, 260))
    photo   = ImageTk.PhotoImage(pil_img)
    canvas.create_image(130, 130, image=photo)
    canvas.image = photo

    status_label.config(text="Analiz ediliyor...", fg="#888")
    root.update()

    def run():
        result = predict(path)

        if "error" in result:
            status_label.config(text=result["error"], fg="#e94560")
            return

        skin_tr = SKIN_TYPE_TR.get(result["skin_type"], result["skin_type"])
        status_label.config(
            text=f"Cilt Tipi: {skin_tr.upper()} (%{result['type_conf']})",
            fg="#e94560")

        for w in regions_frame.winfo_children():
            w.destroy()

        if result["regions"]:
            tk.Label(regions_frame, text="Bolge Bazli Analiz:",
                     font=("Helvetica", 10, "bold"),
                     bg="#1a1a2e", fg="white").pack(anchor="w", pady=(5,3))

            for region_key, (problem, conf) in result["regions"].items():
                region_tr  = REGION_TR.get(region_key, region_key)
                problem_tr = PROBLEM_TR.get(problem, problem)

                row = tk.Frame(regions_frame, bg="#16213e", pady=4, padx=8)
                row.pack(fill="x", pady=2)

                tk.Label(row, text=region_tr, font=("Helvetica", 9, "bold"),
                         bg="#16213e", fg="#a8dadc", width=12, anchor="w").pack(side="left")

                color = "#e94560" if conf > 70 else "#ffd700" if conf > 40 else "#4ecdc4"
                tk.Label(row, text=f"{problem_tr}",
                         font=("Helvetica", 9), bg="#16213e",
                         fg=color).pack(side="left", padx=5)

                tk.Label(row, text=f"%{conf}",
                         font=("Courier", 9), bg="#16213e",
                         fg="white").pack(side="right")

    threading.Thread(target=run, daemon=True).start()

# --- UI ---
root = tk.Tk()
root.title("Cilt Analizi")
root.geometry("500x700")
root.resizable(False, True)
root.configure(bg="#1a1a2e")

tk.Label(root, text="Cilt Analizi", font=("Helvetica", 18, "bold"),
         bg="#1a1a2e", fg="#e94560").pack(pady=10)

canvas = tk.Canvas(root, width=260, height=260, bg="#16213e", highlightthickness=0)
canvas.pack(pady=5)
canvas.create_text(130, 130, text="Fotograf secin", fill="#555",
                   font=("Helvetica", 11))

tk.Button(root, text="Fotograf Sec ve Analiz Et",
          font=("Helvetica", 11, "bold"),
          bg="#e94560", fg="white", relief="flat",
          padx=15, pady=8, cursor="hand2",
          command=browse_and_predict).pack(pady=8)

status_label = tk.Label(root, text="",
                         font=("Helvetica", 11, "bold"),
                         bg="#1a1a2e", fg="#e94560")
status_label.pack()

regions_frame = tk.Frame(root, bg="#1a1a2e")
regions_frame.pack(fill="x", padx=20, pady=5)

tk.Label(root, text="Bu analiz tibbi tani degildir.",
         font=("Helvetica", 8), bg="#1a1a2e", fg="#555").pack(side="bottom", pady=5)

root.mainloop()