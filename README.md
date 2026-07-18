# 🧠 NeuraScan AI — Neuro-Imaging Assistant

A full-stack brain tumor classification web application: a custom-designed
clinical-themed HTML/CSS/JS frontend backed by a real Flask API serving a
PyTorch CNN. Upload an MRI slice and get an instant 4-class probability
breakdown (Glioma, Meningioma, Pituitary, or No Tumor).

⚠️ **Disclaimer:** Educational/portfolio project only. This is **not** a
certified medical device and must never be used for real diagnostic or
clinical decisions. Always consult a qualified radiologist/physician.

---

## 🧠 Overview

This project trains a convolutional neural network (PyTorch) on brain MRI
scans to classify them into one of four categories, then deploys it behind
a real API with a custom frontend designed to feel like a clinical imaging
tool rather than a bare data-science demo.

**Architecture:**

```
Browser (HTML/CSS/JS)  ──POST /predict (image)──▶  Flask API  ──▶  model.pth (CNN)
        ▲                                                              │
        └───────────────── JSON: class probabilities ◀─────────────────┘
```

---

## ✨ Features

- 🎨 **Custom-designed frontend** — clinical teal & amber visual identity,
  Space Grotesk display type, drag-and-drop MRI upload
- 🧠 **Real CNN inference** — the "Analyze Scan" button sends the uploaded
  image to a genuine Flask endpoint, which runs it through the trained
  PyTorch model and returns real softmax probabilities for all 4 classes
- 📊 **Full probability breakdown** — animated confidence bars for every
  class, not just the top prediction
- 🔍 **Scanning animation** — a brief "running convolutional layers…" beam
  animation while the model performs inference
- 📱 **Fully responsive**, accessible focus states, respects reduced-motion
- 🔐 **Clean separation of concerns** — `model_def.py` (shared architecture),
  `train_model.py` (offline training), `app.py` (API + serving)

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Frontend | HTML5, CSS3 (custom, no framework), vanilla JavaScript |
| Backend | Flask (Python) |
| ML Model | Custom CNN (PyTorch) — 2 conv blocks + batch norm + dropout |
| Image Processing | Pillow, torchvision transforms |
| Fonts | Space Grotesk (display), Inter (body), IBM Plex Mono (data) |

---

## 🧠 Model Architecture

```
Input (128×128×3)
   → Conv2d(3→32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
   → Conv2d(32→64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
   → Flatten
   → Linear(64×30×30 → 128) → ReLU → Dropout(0.5)
   → Linear(128 → num_classes)
```

Trained for 10 epochs with Adam (lr=0.001) and cross-entropy loss.

---

## 📂 Dataset

Expects the standard brain tumor MRI dataset folder layout:

```
Training/
  ├── glioma/
  ├── meningioma/
  ├── notumor/
  └── pituitary/
Testing/
  ├── glioma/
  ├── meningioma/
  ├── notumor/
  └── pituitary/
```

(Class names are auto-detected from folder names via `ImageFolder` — no
hardcoding, so this works with any similarly-structured dataset.)

---

## 🚀 Running Locally

**1. Clone the repository**
```bash
git clone https://github.com/Muhammad-Raafe/NeuraScan-AI.git
cd NeuraScan-AI
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your dataset**
Place the `Training/` and `Testing/` folders in the project root.

**4. Train the model** (generates `model.pth`)
```bash
python train_model.py
```

**5. Run the website**
```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## 📁 Project Structure

```
NeuraScan-AI/
│
├── app.py                  # Flask backend — serves the site + /predict API
├── train_model.py           # Trains the CNN, exports model.pth
├── model_def.py               # Shared CNN architecture (used by both)
├── Training/ , Testing/         # Dataset (add your own)
├── model.pth                      # Generated after running train_model.py
├── requirements.txt
├── templates/
│   └── index.html                   # Main page (upload + results)
├── static/
│   ├── css/style.css                  # Full design system
│   └── js/script.js                     # Upload logic, API calls, animations
└── README.md
```

---

## 🌐 Deployment

Needs a Python process running (PyTorch inference isn't static), so it's
best deployed on **Render** or **Railway**:

1. Push this repo to GitHub (include the trained `model.pth` so the host
   doesn't need to retrain — training a CNN on free-tier CPU is slow)
2. Create a new Web Service, point it at the repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `python app.py`

**Note:** PyTorch is a large dependency — free-tier build/RAM limits can be
tight. If deployment struggles, consider `torch` CPU-only wheels to shrink
the install size.

---

## 🔮 Future Improvements

- Add Grad-CAM visualization to highlight which region of the scan drove
  the prediction (huge trust/interpretability win for a medical demo)
- Add batch upload for multiple scans at once
- Add a confusion matrix / per-class precision-recall view from training
- Swap the CNN for a fine-tuned pretrained backbone (ResNet/EfficientNet)
  and compare accuracy

---

## 👤 Author

**Muhammad Raafe**
AI/ML Enthusiast | Building a portfolio in Machine Learning & Data Science

GitHub: [@Muhammad-Raafe](https://github.com/Muhammad-Raafe)
