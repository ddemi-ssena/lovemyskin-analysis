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
        return None
    img    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transform(image=img)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        type_out  = torch.softmax(skin_type_model(tensor), dim=1)[0]
        type_idx  = type_out.argmax().item()
        prob_out  = torch.softmax(problem_model(tensor), dim=1)[0]
        all_probs = {PROBLEM_CLASSES[i]: round(prob_out[i].item()*100, 1)
                     for i in range(len(PROBLEM_CLASSES))}
    return {
        "skin_type": SKIN_TYPE_CLASSES[type_idx],
        "type_conf": round(type_out[type_idx].item()*100, 1),
        "all_scores": dict(sorted(all_probs.items(), key=lambda x: -x[1]))
    }

def show_result(result):
    if not result:
        skin_type_var.set("Gorsel okunamadi!")
        return
    skin_type_var.set(f"Cilt Tipi: {result['skin_type'].upper()}  (%{result['type_conf']})")
    for widget in bars_frame.winfo_children():
        widget.destroy()
    tk.Label(bars_frame, text="Cilt Sorunlari:", font=("Helvetica", 12, "bold"),
             bg="#1a1a2e", fg="white").pack(anchor="w", pady=5)
    for problem, score in result["all_scores"].items():
        row = tk.Frame(bars_frame, bg="#1a1a2e")
        row.pack(fill="x", pady=2)
        tk.Label(row, text=f"{problem:<20}", font=("Courier", 10),
                 bg="#1a1a2e", fg="#a8dadc", width=20, anchor="w").pack(side="left")
        bar_bg = tk.Frame(row, bg="#16213e", width=200, height=16)
        bar_bg.pack(side="left", padx=5)
        bar_bg.pack_propagate(False)
        color = "#e94560" if score > 50 else "#4ecdc4" if score > 25 else "#45b7d1"
        tk.Frame(bar_bg, bg=color, width=int(score*2), height=16).place(x=0, y=0)
        tk.Label(row, text=f"%{score}", font=("Courier", 10),
                 bg="#1a1a2e", fg="white").pack(side="left")

def browse_and_predict():
    path = filedialog.askopenfilename(
        filetypes=[("Gorsel dosyalari", "*.jpg *.jpeg *.png")])
    if not path:
        return
    pil_img = Image.open(path).resize((300, 300))
    photo   = ImageTk.PhotoImage(pil_img)
    img_label.configure(image=photo, width=300, height=300)
    img_label.image = photo
    show_result(predict(path))

root = tk.Tk()
root.title("Cilt Analizi")
root.geometry("700x900")
root.resizable(True, True)
root.configure(bg="#1a1a2e")

tk.Label(root, text="Cilt Analizi", font=("Helvetica", 22, "bold"),
         bg="#1a1a2e", fg="#e94560").pack(pady=20)

tk.Label(root, text="Fotograf secmek icin asagidaki butona tiklayin",
         font=("Helvetica", 11), bg="#1a1a2e", fg="#888").pack()

img_label = tk.Label(root, bg="#16213e", width=300, height=200,
                     text="Fotograf buraya gelecek", fg="#444",
                     font=("Helvetica", 12))
img_label.pack(pady=15)

skin_type_var = tk.StringVar(value="")
tk.Label(root, textvariable=skin_type_var, font=("Helvetica", 14, "bold"),
         bg="#1a1a2e", fg="#e94560").pack(pady=5)

bars_frame = tk.Frame(root, bg="#1a1a2e")
bars_frame.pack(fill="x", padx=40, pady=5)

tk.Button(root, text="Fotograf Sec ve Analiz Et",
          font=("Helvetica", 13, "bold"),
          bg="#e94560", fg="white", relief="flat",
          padx=20, pady=10, cursor="hand2",
          command=browse_and_predict).pack(pady=15)

tk.Label(root, text="Bu analiz tibbi tani degildir.",
         font=("Helvetica", 9), bg="#1a1a2e", fg="#555").pack()

root.mainloop()