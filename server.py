# server.py
import os
import pickle
import numpy as np
from flask import Flask, request, jsonify

MODEL_PATH = "asl_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run train_model.py first.")

app = Flask(__name__)

with open("asl_model.pkl", "rb") as f:
    bundle = pickle.load(f)
clf = bundle["model"]
le  = bundle["label_encoder"]

MIN_CONFIDENCE = 0.3

def normalize(coords):
    lm = np.array(coords).reshape(-1, 3)
    lm -= lm[0]           # wrist-relative
    max_val = np.max(np.abs(lm))
    if max_val != 0:
        lm /= max_val
    return lm.flatten()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    landmarks = data.get("landmarks", [])
    if len(landmarks) != 63:
        return jsonify({"error": f"Expected 63 values, got {len(landmarks)}"}), 400

    features = normalize(landmarks).reshape(1, -1)   # normalize raw input
    pred = clf.predict(features)[0]
    prob = float(clf.predict_proba(features).max())
    letter = le.inverse_transform([pred])[0].upper()

    if prob < MIN_CONFIDENCE:
        return jsonify({"prediction": None, "confidence": prob, "error": "Low confidence"})

    return jsonify({"prediction": letter, "confidence": prob})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)