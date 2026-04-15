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
    try:
        print("Request received")

        if "file" not in request.files:
            print("No file uploaded")
            return jsonify({"error": "No file uploaded"}), 400

        image_file = request.files["file"]
        image_bytes = image_file.read()
        print("File received")

        arr = preprocess_image(image_bytes)
        print("Image preprocessed")

        preds = model.predict(arr, verbose=0)[0]
        print("Prediction done")

        top_k_idx = np.argsort(preds)[::-1][:5]
        top_k = [{"label": class_names[i], "score": float(preds[i])} for i in top_k_idx]

        result = {
            "prediction": class_names[int(np.argmax(preds))],
            "confidence": float(np.max(preds)),
            "top_k": top_k
        }

        print("Sending response:", result)
        return jsonify(result)

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


