"""
Signal 2 - Stylometric heuristics (pure Python).  [planning.md > Detection Signals > Signal 2]

Structural / statistical detector. Measures quantifiable variance that tends to
differ between human and AI writing, on the axis 0 = human-like, 1 = AI-like.

Sub-metrics (each mapped to a 0-1 "AI-likeness" value):
  - burstiness  : sentence-length variation (coefficient of variation). Human
                  writing is bursty/uneven; AI clusters around uniform lengths.
                  LOW variation -> AI-like.  [most reliable -> highest weight]
  - diversity   : type-token ratio (unique words / total). LOW diversity -> AI-like.
  - punctuation : punctuation variety. LOW variety -> AI-like.

Calibration note: exploration on the test corpus showed TTR and punctuation barely
separate short samples (TTR clustered 0.83-0.90), while sentence-length burstiness
separated human / AI / formal text clearly. The three sub-metrics are therefore
combined with a BURSTINESS-DOMINANT weighting (0.70 / 0.15 / 0.15) rather than a
flat average, so the weak short-text metrics can nudge but not wash out the strong one.

Output shape:
    {
        "stylo_score": float in [0, 1],   # 1.0 = AI-like, 0.0 = human-like
        "features": { raw measured values, for the audit log & debugging }
    }
"""

import re
import statistics

# Sub-metric weights (burstiness-dominant; see calibration note above).
W_BURST, W_DIV, W_PUNCT = 0.70, 0.15, 0.15


def _split_sentences(text):
    return [p.strip() for p in re.split(r"[.!?]+", text) if p.strip()]


def _words(text):
    return re.findall(r"[A-Za-z']+", text.lower())


def _clamp01(x):
    return max(0.0, min(1.0, x))


def get_stylometric_signal(text):
    """Return {"stylo_score": float, "features": {...}} for `text`."""
    sentences = _split_sentences(text)
    toks = _words(text)
    n_words = len(toks)

    # --- raw features ---
    sent_lens = [len(_words(s)) for s in sentences]
    mean_len = statistics.mean(sent_lens) if sent_lens else 0.0
    std_len = statistics.pstdev(sent_lens) if len(sent_lens) > 1 else 0.0
    cv = (std_len / mean_len) if mean_len else 0.0          # burstiness
    ttr = (len(set(toks)) / n_words) if n_words else 0.0    # vocabulary diversity
    distinct_punct = len(set(re.findall(r"[,.;:!?\"'()\-]", text)))

    # --- map each raw feature to 0-1 AI-likeness ---
    # Burstiness: CV >= 0.58 -> human (0);  CV <= 0.18 -> AI (1).
    burst_ai = _clamp01((0.58 - cv) / 0.40)
    # Diversity: TTR >= 0.82 -> human (0);  TTR <= 0.57 -> AI (1).
    div_ai = _clamp01((0.82 - ttr) / 0.25)
    # Punctuation variety: >=3 distinct types -> human (0);  <=0 -> AI (1).
    punct_ai = _clamp01((3 - distinct_punct) / 3)

    stylo_score = W_BURST * burst_ai + W_DIV * div_ai + W_PUNCT * punct_ai

    return {
        "stylo_score": round(_clamp01(stylo_score), 4),
        "features": {
            "n_words": n_words,
            "n_sentences": len(sentences),
            "sentence_length_cv": round(cv, 4),
            "type_token_ratio": round(ttr, 4),
            "distinct_punctuation": distinct_punct,
            "sub_scores": {
                "burstiness_ai": round(burst_ai, 4),
                "diversity_ai": round(div_ai, 4),
                "punctuation_ai": round(punct_ai, 4),
            },
        },
    }
