"""
Provenance Guard - Flask app (Milestone 4).

Routes:
  POST /submit  - accepts {text, creator_id}, runs BOTH signals, fuses them into a
                  calibrated confidence + verdict, returns a structured response,
                  and writes an audit entry capturing both signals + the combination.
  GET  /log     - returns recent audit entries as JSON.
  GET  /health  - liveness check.

NOTE: `label` is still a PLACEHOLDER in M4. Real 3-variant transparency labels +
appeals + rate limiting arrive in Milestone 5. See planning.md for the full spec.
"""

import uuid
from flask import Flask, request, jsonify
from dotenv import load_dotenv

from llm_signal import get_llm_signal
from stylometry import get_stylometric_signal
from scoring import fuse_signals
from audit import write_entry, get_log, utc_now_iso

load_dotenv()

app = Flask(__name__)

MIN_WORDS = 5  # reject trivially short input; the 40-word short-text guard lives in scoring.py


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/submit", methods=["POST"])
def submit():
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    creator_id = (body.get("creator_id") or "").strip()

    # --- validation ---
    if not text:
        return jsonify({"error": "Field 'text' is required."}), 400
    if not creator_id:
        return jsonify({"error": "Field 'creator_id' is required."}), 400
    n_words = len(text.split())
    if n_words < MIN_WORDS:
        return jsonify({"error": f"Text must be at least {MIN_WORDS} words."}), 400

    content_id = str(uuid.uuid4())

    # --- Signal 1: LLM (Groq, semantic) ---
    try:
        s1 = get_llm_signal(text)
    except Exception as e:
        return jsonify({"error": f"Detection failed: {e}"}), 502
    llm_score = s1["llm_score"]

    # --- Signal 2: Stylometry (pure Python, structural) ---
    s2 = get_stylometric_signal(text)
    stylo_score = s2["stylo_score"]

    # --- Fusion: combine into calibrated confidence + verdict ---
    fused = fuse_signals(llm_score, stylo_score, n_words)
    attribution = fused["verdict"]
    confidence = fused["confidence"]
    ai_likelihood = fused["ai_likelihood"]

    # --- PLACEHOLDER label (real 3-variant labels arrive in M5) ---
    label = {
        "variant": "placeholder",
        "text": "Transparency label generation arrives in Milestone 5.",
    }

    # --- audit entry: both signals + the combined result ---
    entry = {
        "content_id": content_id,
        "creator_id": creator_id,
        "timestamp": utc_now_iso(),
        "attribution": attribution,
        "confidence": confidence,
        "ai_likelihood": ai_likelihood,
        "signals": {
            "llm_score": llm_score,
            "stylo_score": stylo_score,
        },
        "rules_applied": fused["rules_applied"],
        "status": "classified",
    }
    write_entry(entry)

    # --- response ---
    return jsonify({
        "content_id": content_id,
        "attribution": attribution,
        "confidence": confidence,
        "ai_likelihood": ai_likelihood,
        "label": label,
        "signals": {
            "llm": {"llm_score": llm_score, "rationale": s1["rationale"]},
            "stylometry": {"stylo_score": stylo_score, "features": s2["features"]},
        },
        "rules_applied": fused["rules_applied"],
    })


@app.route("/log", methods=["GET"])
def log():
    return jsonify({"entries": get_log()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
