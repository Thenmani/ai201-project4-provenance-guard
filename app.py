"""
Provenance Guard - Flask app (Milestone 3).

Routes:
  POST /submit  - accepts {text, creator_id}, runs Signal 1 (LLM), returns a
                  structured response and writes an audit entry.
  GET  /log     - returns recent audit entries as JSON.
  GET  /health  - liveness check.

NOTE: `confidence` and `label` are PLACEHOLDERS in Milestone 3.
  - Real calibrated confidence (fusion of two signals) arrives in Milestone 4.
  - Real 3-variant transparency labels + appeals + rate limiting arrive in M5.
See planning.md for the full spec these implement against.
"""

import uuid
from flask import Flask, request, jsonify
from dotenv import load_dotenv

from llm_signal import get_llm_signal
from audit import write_entry, get_log, utc_now_iso

load_dotenv()

app = Flask(__name__)

MIN_WORDS = 5  # basic guard; real short-text handling (40-word guard) comes in M4


def derive_placeholder_attribution(llm_score):
    """Milestone-3 stand-in attribution from the single LLM signal.

    Replaced in Milestone 4 by fusion over the asymmetric bands defined in
    planning.md (>= 0.80 likely_ai, <= 0.40 likely_human, else uncertain).
    """
    if llm_score >= 0.80:
        return "likely_ai"
    if llm_score <= 0.40:
        return "likely_human"
    return "uncertain"


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
    if len(text.split()) < MIN_WORDS:
        return jsonify({"error": f"Text must be at least {MIN_WORDS} words."}), 400

    content_id = str(uuid.uuid4())

    # --- Signal 1: LLM (Groq) ---
    try:
        signal1 = get_llm_signal(text)
    except Exception as e:  # API/key/parse failures surface as 502
        return jsonify({"error": f"Detection failed: {e}"}), 502

    llm_score = signal1["llm_score"]
    attribution = derive_placeholder_attribution(llm_score)

    # --- PLACEHOLDERS (replaced in M4 / M5) ---
    confidence = llm_score  # placeholder: real calibrated confidence in M4
    label = {
        "variant": "placeholder",
        "text": "Transparency label generation arrives in Milestone 5.",
    }

    # --- audit entry ---
    entry = {
        "content_id": content_id,
        "creator_id": creator_id,
        "timestamp": utc_now_iso(),
        "attribution": attribution,
        "confidence": round(confidence, 4),
        "llm_score": round(llm_score, 4),
        "status": "classified",
    }
    write_entry(entry)

    # --- response ---
    return jsonify({
        "content_id": content_id,
        "attribution": attribution,
        "confidence": round(confidence, 4),
        "label": label,
        "signals": {
            "llm": {
                "llm_score": round(llm_score, 4),
                "rationale": signal1["rationale"],
            }
        },
    })


@app.route("/log", methods=["GET"])
def log():
    return jsonify({"entries": get_log()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
