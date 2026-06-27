"""
Provenance Guard - Flask app (Milestone 5, production layer).

Routes:
  POST /submit  - accepts {text, creator_id}, runs both signals, fuses them into a
                  calibrated confidence + verdict, generates the transparency label,
                  writes an audit entry, returns the structured response.
                  Rate limited (see RATE_LIMIT_SUBMIT).
  POST /appeal  - accepts {content_id, creator_reasoning}, sets the content's status
                  to "under_review", records the reasoning in the audit log, returns
                  a confirmation. No automated re-classification (human reviewer owns it).
  GET  /log     - returns recent audit entries as JSON.
  GET  /health  - liveness check.

See planning.md for the full spec these implement against.
"""

import uuid
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

from llm_signal import get_llm_signal
from stylometry import get_stylometric_signal
from scoring import fuse_signals
from labels import generate_label
from audit import write_entry, update_entry, get_log, utc_now_iso

load_dotenv()

app = Flask(__name__)

# --- Rate limiting (see README for the reasoning behind these numbers) ---
# /submit : a real writer submits their own work a handful of times per session;
#           10/min gives generous headroom for editing + resubmitting, while a
#           flooding script (hundreds/min) is throttled to a trickle. 100/day caps
#           sustained abuse from one IP without blocking a prolific legitimate user.
# /appeal : appeals are rare and human-driven; 30/hour is plenty for a real creator
#           and stops an appeal-spam script.
RATE_LIMIT_SUBMIT = "10 per minute;100 per day"
RATE_LIMIT_APPEAL = "30 per hour"

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

MIN_WORDS = 5  # reject trivially short input; the 40-word short-text guard lives in scoring.py


@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "Rate limit exceeded. Please slow down and try again later.",
        "detail": str(e.description),
    }), 429


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/submit", methods=["POST"])
@limiter.limit(RATE_LIMIT_SUBMIT)
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

    # --- Transparency label (real, 3-variant) ---
    label = generate_label(attribution, confidence)

    # --- audit entry: full record (both signals + combined + appeal slot) ---
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
        "label_variant": label["variant"],
        "rules_applied": fused["rules_applied"],
        "status": "classified",
        "appeal_reasoning": None,   # populated if/when an appeal is filed
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


@app.route("/appeal", methods=["POST"])
@limiter.limit(RATE_LIMIT_APPEAL)
def appeal():
    body = request.get_json(silent=True) or {}
    content_id = (body.get("content_id") or "").strip()
    reasoning = (body.get("creator_reasoning") or "").strip()

    # --- validation ---
    if not content_id:
        return jsonify({"error": "Field 'content_id' is required."}), 400
    if not reasoning:
        return jsonify({"error": "Field 'creator_reasoning' is required."}), 400

    # --- update the content's audit entry in place ---
    updated = update_entry(content_id, {
        "status": "under_review",
        "appeal_reasoning": reasoning,
        "appeal_timestamp": utc_now_iso(),
    })
    if updated is None:
        return jsonify({"error": f"No content found with content_id '{content_id}'."}), 404

    return jsonify({
        "content_id": content_id,
        "status": "under_review",
        "message": ("Your appeal has been received. This content is now under review "
                    "by a human moderator. No automated re-classification is performed."),
        "appeal_reasoning": reasoning,
    })


@app.route("/log", methods=["GET"])
def log():
    return jsonify({"entries": get_log()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
