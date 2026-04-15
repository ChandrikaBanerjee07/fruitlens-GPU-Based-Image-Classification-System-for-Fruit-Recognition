from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io
import os


app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("fruit_classifier.keras")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.route("/", methods=["GET"])
def home():
    return "FruitLens backend is running"
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    image_bytes = request.files["file"].read()
    arr = preprocess_image(image_bytes)
    preds = model.predict(arr, verbose=0)[0]

    top_k_idx = np.argsort(preds)[::-1][:5]
    top_k = [{"label": class_names[i], "score": float(preds[i])} for i in top_k_idx]

    return jsonify({
        "prediction": class_names[int(np.argmax(preds))],
        "confidence": float(np.max(preds)),
        "top_k": top_k
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


