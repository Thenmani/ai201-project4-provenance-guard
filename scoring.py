"""
Confidence fusion + verdict banding.  [planning.md > Fusion + Uncertainty Representation]

Combines the two signal scores (each on the axis 0 = human-like, 1 = AI-like)
into a single calibrated `ai_likelihood`, then maps it to a verdict using
DELIBERATELY ASYMMETRIC bands (it is harder to assert "AI" than "human").

Rules implemented (all from planning.md):
  1. Weighted fusion:  ai_likelihood = 0.65*llm + 0.35*stylo   (LLM weighted higher; stylometry is noisier)
  2. Short-text guard: < 40 words -> stylometry is unreliable. Shift weight to the
     LLM (0.90/0.10) and WIDEN the uncertain band, so the system is slower to make
     a confident claim on short text.
  3. Disagreement->uncertain: if |llm - stylo| > 0.40 the signals contradict each
     other; clamp the result into the uncertain band rather than trusting either.

Bands (normal):      ai>=0.80 likely_ai | 0.40<ai<0.80 uncertain | ai<=0.40 likely_human
Bands (short text):  ai>=0.85 likely_ai | 0.35<ai<0.85 uncertain | ai<=0.35 likely_human
"""

# Fusion weights
W_LLM, W_STYLO = 0.65, 0.35
W_LLM_SHORT, W_STYLO_SHORT = 0.90, 0.10

# Verdict band cutoffs
AI_CUT, HUMAN_CUT = 0.75, 0.40              # normal
AI_CUT_SHORT, HUMAN_CUT_SHORT = 0.82, 0.35  # short text (wider uncertain band)

SHORT_TEXT_WORDS = 40
DISAGREEMENT_THRESHOLD = 0.40


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def fuse_signals(llm_score, stylo_score, n_words):
    """Combine the two signals into a calibrated verdict.

    Returns a dict:
      {
        "ai_likelihood": float,     # 0..1, the system's P(AI)
        "verdict": str,             # likely_ai | uncertain | likely_human
        "confidence": float,        # confidence IN the verdict (see planning.md)
        "rules_applied": [str],     # which rules fired (for the audit log)
        "weights": {"llm":..,"stylo":..}
      }
    """
    rules = []
    short = n_words < SHORT_TEXT_WORDS

    # --- Rule 2: short-text guard picks the weights & cutoffs ---
    if short:
        w_llm, w_stylo = W_LLM_SHORT, W_STYLO_SHORT
        ai_cut, human_cut = AI_CUT_SHORT, HUMAN_CUT_SHORT
        rules.append("short_text_guard")
    else:
        w_llm, w_stylo = W_LLM, W_STYLO
        ai_cut, human_cut = AI_CUT, HUMAN_CUT

    # --- Rule 1: weighted fusion ---
    ai_likelihood = w_llm * llm_score + w_stylo * stylo_score

    # --- Rule 3: disagreement -> clamp into the uncertain band ---
    if abs(llm_score - stylo_score) > DISAGREEMENT_THRESHOLD:
        ai_likelihood = _clamp(ai_likelihood, human_cut + 0.01, ai_cut - 0.01)
        rules.append("disagreement_to_uncertain")

    ai_likelihood = round(_clamp(ai_likelihood, 0.0, 1.0), 4)

    # --- verdict from asymmetric bands ---
    if ai_likelihood >= ai_cut:
        verdict = "likely_ai"
        confidence = ai_likelihood
    elif ai_likelihood <= human_cut:
        verdict = "likely_human"
        confidence = 1.0 - ai_likelihood
    else:
        verdict = "uncertain"
        # Confidence in an "uncertain" verdict = how central it is (peaks at 0.5).
        confidence = 1.0 - 2 * abs(ai_likelihood - 0.5)

    return {
        "ai_likelihood": ai_likelihood,
        "verdict": verdict,
        "confidence": round(_clamp(confidence, 0.0, 1.0), 4),
        "rules_applied": rules,
        "weights": {"llm": w_llm, "stylo": w_stylo},
    }
