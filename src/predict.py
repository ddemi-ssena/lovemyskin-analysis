import cv2, torch
import numpy as np
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

PROBLEM_CLASSES  = ["acne", "dark_circles", "eczema", "enlarged_pores",
                     "fine_lines", "hyperpigmentation", "rosacea", "wrinkle"]
SKIN_TYPE_CLASSES = ["combination", "dry", "normal", "oily"]

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])

def load_model(path, num_classes):
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

problem_model   = load_model("models/best_model.pt", 8)
skin_type_model = load_model("models/skin_type_model.pt", 4)

def predict(img_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Gorsel okunamadi"}
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transform(image=img)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # Cilt tipi
        type_out   = torch.softmax(skin_type_model(tensor), dim=1)[0]
        type_idx   = type_out.argmax().item()
        type_conf  = type_out[type_idx].item()

        # Cilt sorunlari
        prob_out   = torch.softmax(problem_model(tensor), dim=1)[0]
        prob_idx   = prob_out.argmax().item()
        prob_conf  = prob_out[prob_idx].item()

        # Tum sorunlarin skorlari
        all_problems = {PROBLEM_CLASSES[i]: round(prob_out[i].item(), 3)
                        for i in range(len(PROBLEM_CLASSES))}

    return {
        "skin_type":   SKIN_TYPE_CLASSES[type_idx],
        "type_conf":   round(type_conf, 3),
        "main_problem": PROBLEM_CLASSES[prob_idx],
        "prob_conf":   round(prob_conf, 3),
        "all_scores":  all_problems
    }

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else input("Gorsel yolu: ")
    result = predict(path)
    print("\n=== SONUC ===")
    print(f"Cilt Tipi  : {result['skin_type']} (%{result['type_conf']*100:.1f})")
    print(f"Ana Sorun  : {result['main_problem']} (%{result['prob_conf']*100:.1f})")
    print(f"\nTum Skorlar:")
    for k, v in sorted(result['all_scores'].items(), key=lambda x: -x[1]):
        bar = "█" * int(v * 20)
        print(f"  {k:<20} {bar} {v:.3f}")