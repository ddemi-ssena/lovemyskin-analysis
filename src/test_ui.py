import tkinter as tk
from tkinter import filedialog
import cv2, torch
import numpy as np
from PIL import Image, ImageTk
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
PROBLEM_CLASSES   = ["acne", "dark_circles", "eczema", "enlarged_pores",
                      "fine_lines", "hyperpigmentation", "rosacea", "wrinkle"]
SKIN_TYPE_CLASSES = ["combination", "dry", "normal", "oily"]

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])

# Yuz tespiti
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def load_model(path, num_classes):
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model.to(DEVICE)

print("Modeller yukleniyor...")
problem_model   = load_model("models/best_model.pt", 8)
skin_type_model = load_model("models/skin_type_model.pt", 4)
print("Hazir!")

def predict(img_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Gorsel okunamadi"}
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Yuz tespiti
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return {"error": "Yuzunuzun gorundugu bir fotograf yukleyin!"}

    # En buyuk yuzu al ve kirp
    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
    face_img    = img_rgb[y:y+h, x:x+w]

    tensor = transform(image=face_img)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        type_out = torch.softmax(skin_type_model(tensor), dim=1)[0]
        type_idx = type_out.argmax().item()
        prob_out = torch.softmax(problem_model(tensor), dim=1)[0]
        all_probs = {PROBLEM_CLASSES[i]: round(prob_out[i].item()*100, 1)
                     for i in range(len(PROBLEM_CLASSES))}
    return {
        "skin_type": SKIN_TYPE_CLASSES[type_idx],
        "type_conf": round(type_out[type_idx].item()*100, 1),
        "all_scores": dict(sorted(all_probs.items(), key=lambda x: -x[1]))
    }

def browse_and_predict():
    path = filedialog.askopenfilename(
        filetypes=[("Gorsel", "*.jpg *.jpeg *.png")])
    if not path:
        return

    pil_img = Image.open(path).resize((220, 220))
    photo   = ImageTk.PhotoImage(pil_img)
    canvas.create_image(110, 110, image=photo)
    canvas.image = photo

    result = predict(path)

    if "error" in result:
        skin_type_label.config(text=result["error"])
        for w in bars_frame.winfo_children():
            w.destroy()
        return

    skin_type_label.config(
        text=f"Cilt Tipi: {result['skin_type'].upper()} (%{result['type_conf']})")

    for w in bars_frame.winfo_children():
        w.destroy()

    for problem, score in result["all_scores"].items():
        row = tk.Frame(bars_frame, bg="#1a1a2e")
        row.pack(fill="x", pady=2)
        tk.Label(row, text=f"{problem:<18}", font=("Courier", 9),
                 bg="#1a1a2e", fg="#a8dadc").pack(side="left")
        tk.Label(row, text=f"%{score:>5}", font=("Courier", 9),
                 bg="#1a1a2e", fg="white").pack(side="left")

root = tk.Tk()
root.title("Cilt Analizi")
root.geometry("480x620")
root.resizable(False, False)
root.configure(bg="#1a1a2e")

tk.Label(root, text="Cilt Analizi", font=("Helvetica", 18, "bold"),
         bg="#1a1a2e", fg="#e94560").pack(pady=10)

canvas = tk.Canvas(root, width=220, height=220, bg="#16213e", highlightthickness=0)
canvas.pack(pady=5)
canvas.create_text(110, 110, text="Fotograf secin", fill="#555", font=("Helvetica", 11))

tk.Button(root, text="Fotograf Sec ve Analiz Et",
          font=("Helvetica", 11, "bold"),
          bg="#e94560", fg="white", relief="flat",
          padx=15, pady=8, cursor="hand2",
          command=browse_and_predict).pack(pady=8)

skin_type_label = tk.Label(root, text="Cilt tipi: —",
                            font=("Helvetica", 11, "bold"),
                            bg="#1a1a2e", fg="#e94560")
skin_type_label.pack()

bars_frame = tk.Frame(root, bg="#1a1a2e")
bars_frame.pack(fill="x", padx=30, pady=5)

tk.Label(root, text="Bu analiz tibbi tani degildir.",
         font=("Helvetica", 8), bg="#1a1a2e", fg="#555").pack(side="bottom", pady=5)

root.mainloop()