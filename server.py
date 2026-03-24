# server.py
import os
import json
import pickle
import functools
import numpy as np
import jwt as pyjwt
import requests as req
from flask import Flask, request, jsonify, g
from flask_sock import Sock

MODEL_PATH = "asl_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run train_model.py first.")

app = Flask(__name__)
sock = Sock(app)

# ── Environment config (set in Render.com dashboard) ────────────────────────
SUPABASE_JWT_SECRET  = os.environ.get("SUPABASE_JWT_SECRET", "")
SUPABASE_URL         = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

# Role → feature flag mapping (mirrors Supabase get_my_feature_flags RPC)
ROLE_FLAGS: dict[str, dict[str, bool]] = {
    "admin": {
        "dev_mode": True, "data_collection": True, "premium_content": True,
        "unlimited_translations": True, "offline_mode": True, "debug_overlay": True,
    },
    "dev": {
        "dev_mode": True, "data_collection": True, "premium_content": True,
        "unlimited_translations": True, "offline_mode": False, "debug_overlay": True,
    },
    "pro": {
        "dev_mode": False, "data_collection": False, "premium_content": True,
        "unlimited_translations": True, "offline_mode": True, "debug_overlay": False,
    },
    "free": {
        "dev_mode": False, "data_collection": False, "premium_content": False,
        "unlimited_translations": False, "offline_mode": False, "debug_overlay": False,
    },
}


def _verify_jwt(token: str) -> dict | None:
    """Validates the Supabase JWT signature. Returns payload or None on failure."""
    if not SUPABASE_JWT_SECRET:
        return None
    try:
        return pyjwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            options={"require": ["sub", "exp", "iss"]},
        )
    except pyjwt.InvalidTokenError:
        return None


def _get_role_for_user(user_id: str) -> str:
    """Fetches the authoritative role from profiles table via Supabase service key."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return "free"
    try:
        resp = req.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"id": f"eq.{user_id}", "select": "role"},
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            },
            timeout=3,
        )
        if resp.status_code == 200:
            rows = resp.json()
            if rows and rows[0].get("role") in ROLE_FLAGS:
                return rows[0]["role"]
    except Exception:
        pass
    return "free"


def require_auth(f):
    """Decorator: validates Supabase JWT, populates g.user_id and g.role."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return jsonify({"error": "Missing or malformed Authorization header"}), 401
        payload = _verify_jwt(header[7:])
        if payload is None:
            return jsonify({"error": "Invalid or expired token"}), 401
        g.user_id = payload.get("sub")
        g.role = _get_role_for_user(g.user_id)
        return f(*args, **kwargs)
    return decorated


def require_flag(flag_name: str):
    """Decorator (stacked after require_auth): returns 403 if role lacks the flag."""
    def decorator(f):
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            role = getattr(g, "role", "free")
            if not ROLE_FLAGS.get(role, {}).get(flag_name, False):
                return jsonify({"error": f"Feature '{flag_name}' not available for role '{role}'"}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator


# ── Feature flags endpoint ───────────────────────────────────────────────────

@app.route("/api/features", methods=["GET"])
@require_auth
def get_features():
    """
    Returns the authenticated user's role and feature flags.
    Called by the React Native app on login to populate FeatureFlagsContext.

    Response: { "role": "dev", "features": { "dev_mode": true, ... } }
    """
    role = g.role
    return jsonify({"role": role, "features": ROLE_FLAGS.get(role, ROLE_FLAGS["free"])})


# ── ML model ─────────────────────────────────────────────────────────────────

with open("asl_model.pkl", "rb") as f:
    bundle = pickle.load(f)
clf = bundle["model"]
le  = bundle["label_encoder"]

MIN_CONFIDENCE = 0.55

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
    proba = clf.predict_proba(features)[0]

    # Top-3 candidates — always included so frontend can vote across frames
    top3_indices = proba.argsort()[::-1][:3]
    top3 = [
        {"letter": le.inverse_transform([i])[0].upper(), "confidence": float(proba[i])}
        for i in top3_indices
    ]

    pred = top3_indices[0]
    prob = float(proba[pred])
    letter = top3[0]["letter"]

    if prob < MIN_CONFIDENCE:
        return {"prediction": None, "confidence": prob, "top3": top3, "error": "Low confidence"}

    return {"prediction": letter, "confidence": prob, "top3": top3}


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
