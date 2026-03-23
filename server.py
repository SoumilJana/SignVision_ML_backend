# server.py
import os
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_sock import Sock

MODEL_PATH = "asl_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run train_model.py first.")

app = Flask(__name__)
sock = Sock(app)

with open("asl_model.pkl", "rb") as f:
    bundle = pickle.load(f)
clf = bundle["model"]
le  = bundle["label_encoder"]

MIN_CONFIDENCE = 0.65

def normalize(coords):
    lm = np.array(coords).reshape(-1, 3)
    lm -= lm[0]           # wrist-relative
    max_val = np.max(np.abs(lm))
    if max_val != 0:
        lm /= max_val
    return lm.flatten()

def run_prediction(landmarks):
    """Shared prediction logic used by both HTTP and WebSocket endpoints."""
    if len(landmarks) != 63:
        return {"error": f"Expected 63 values, got {len(landmarks)}"}

    features = normalize(landmarks).reshape(1, -1)
    pred = clf.predict(features)[0]
    prob = float(clf.predict_proba(features).max())
    letter = le.inverse_transform([pred])[0].upper()

    if prob < MIN_CONFIDENCE:
        return {"prediction": None, "confidence": prob, "error": "Low confidence"}

    return {"prediction": letter, "confidence": prob}


# ── Existing HTTP endpoint (unchanged behavior) ─────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    landmarks = data.get("landmarks", [])
    result = run_prediction(landmarks)
    status = 400 if "error" in result and result.get("prediction") is None and "Expected" in result.get("error", "") else 200
    return jsonify(result), status


# ── WebSocket endpoint with frame dropping ───────────────────────────────────

@sock.route('/ws')
def ws_predict(ws):
    while True:
        try:
            raw = ws.receive()
            if raw is None:
                break

            # Frame dropping: drain any queued messages, only process the latest
            latest = raw
            while True:
                try:
                    queued = ws.receive(timeout=0)
                    if queued is None:
                        break
                    latest = queued
                except Exception:
                    break

            data = json.loads(latest)
            landmarks = data.get("landmarks", [])
            request_id = data.get("id")

            result = run_prediction(landmarks)
            if request_id is not None:
                result["id"] = request_id

            ws.send(json.dumps(result))

        except Exception as e:
            # Connection closed or unexpected error — exit cleanly
            break


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
