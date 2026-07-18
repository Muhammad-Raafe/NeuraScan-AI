"""
app.py
-------
Flask backend for the brain tumor classification website.

- Serves the frontend (templates/index.html + static assets)
- Loads the trained CNN (model.pth, produced by train_model.py)
- Exposes POST /predict which takes an uploaded MRI image and
  returns real class probabilities from the trained model.

Run:
    python train_model.py   (once, to generate model.pth)
    python app.py           (starts the website on http://localhost:5000)
"""

import os
import io
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify

from model_def import CNN

app = Flask(__name__)

MODEL_PATH = "model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
class_names = []

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]
    model = CNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model loaded. Classes:", class_names)
else:
    print("WARNING: model.pth not found. Run `python train_model.py` first.")


def prettify_label(label: str) -> str:
    return label.replace("_", " ").replace("-", " ").title()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1)[0].cpu().numpy()

        results = [
            {"label": prettify_label(class_names[i]), "raw_label": class_names[i], "probability": round(float(probs[i]) * 100, 2)}
            for i in range(len(class_names))
        ]
        results.sort(key=lambda r: r["probability"], reverse=True)

        top = results[0]
        is_tumor = "notumor" not in top["raw_label"].lower() and "no_tumor" not in top["raw_label"].lower() and "no-tumor" not in top["raw_label"].lower()

        return jsonify({
            "prediction": top["label"],
            "confidence": top["probability"],
            "is_tumor": is_tumor,
            "all_probabilities": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
