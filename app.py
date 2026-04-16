"""
FruitLens Backend — app.py
Fixed: CORS headers, model loading error handling, wake endpoint, correct preprocessing
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io
import os
import traceback

app = Flask(__name__)

# ─── CORS: allow ALL origins (needed for Netlify ↔ Render) ───────────────────
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

# ─── Also manually set CORS headers on every response ────────────────────────
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

@app.route("/", methods=["GET", "OPTIONS"])
def index():
    return jsonify({"status": "FruitLens backend is running", "model_loaded": model is not None})

# ─── Handle preflight OPTIONS requests ───────────────────────────────────────
@app.route("/predict", methods=["OPTIONS"])
def predict_preflight():
    return "", 204

# ─── Load model + class names ─────────────────────────────────────────────────
MODEL_PATH       = os.environ.get("MODEL_PATH", "fruit_classifier.h5")
CLASS_NAMES_PATH = os.environ.get("CLASS_NAMES_PATH", "class_names.json")

model       = None
class_names = []

def load_model_once():
    global model, class_names
    try:
        print(f"Loading model from {MODEL_PATH} ...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded.")
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        traceback.print_exc()

    try:
        with open(CLASS_NAMES_PATH) as f:
            class_names = json.load(f)
        print(f"✅ Classes loaded: {class_names}")
    except Exception as e:
        print(f"❌ class_names.json load failed: {e}")

load_model_once()


# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Load image bytes → resize to 224×224 → convert to float32.
    NOTE: mobilenet_v2.preprocess_input is BAKED INTO the model graph,
    so we only need to pass raw [0-255] pixels here.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)       # shape (224, 224, 3), range [0, 255]
    arr = np.expand_dims(arr, axis=0)            # shape (1, 224, 224, 3)
    return arr


# ─── Predict endpoint ─────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file field in request. Send as multipart/form-data with key 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        image_bytes = file.read()
        arr = preprocess_image(image_bytes)

        preds = model.predict(arr, verbose=0)[0]   # shape (num_classes,)

        top_k_idx = np.argsort(preds)[::-1][:5]
        top_k = [
            {"label": class_names[i], "score": round(float(preds[i]), 6)}
            for i in top_k_idx
        ]

        return jsonify({
            "prediction": class_names[int(np.argmax(preds))],
            "confidence": round(float(np.max(preds)), 6),
            "top_k": top_k
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─── Health / wake endpoint (keeps Render from timing out) ───────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "num_classes": len(class_names),
        "classes": class_names
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
