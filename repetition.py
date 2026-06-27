"""
Signal 3 - Repetition / redundancy heuristic (pure Python).  [planning.md > Stretch: Ensemble]

Third detector for the ensemble. Measures how much a text repeats itself, on the
same axis as the other signals: 0 = human-like (varied), 1 = AI-like (repetitive).

Distinct from the other two signals: the LLM reads meaning, stylometry measures
structural variance (sentence length), and this measures REDUNDANCY — reused
phrasing and formulaic filler. AI writing leans on the same transitions and
phrasings ("Furthermore", "Moreover", "It is important to note"); human writing
repeats less mechanically.

Sub-metrics (each mapped to 0-1 AI-likeness):
  - transition_density : how densely the text uses formulaic connectors. [most
                         reliable at short lengths -> highest weight]
  - ngram_repetition   : fraction of repeated word-pairs/triples (matters more on
                         longer text).
  - opener_variety     : how varied the sentence openings are (low variety -> AI-like).

Calibration note: on short samples, n-gram repetition and opener variety barely move
(texts are too short to repeat phrases), while transition density cleanly separates
formulaic AI text from human writing. The sub-metrics are therefore combined
transition-dominant (0.60 / 0.25 / 0.15). A useful property fell out of this: formal
/ academic human writing scores ~0 here (it is formal but not full of cheap filler),
so this signal tends to PROTECT the false-positive trap case rather than worsen it.

Output shape:
    {
        "repetition_score": float in [0, 1],   # 1.0 = AI-like (repetitive)
        "features": { raw measured values }
    }
"""

import re

# Sub-metric weights (transition-dominant; see calibration note).
W_TRANS, W_NGRAM, W_OPENER = 0.60, 0.25, 0.15

# Formulaic connectors / filler phrases that AI text over-uses.
TRANSITION_PHRASES = [
    "furthermore", "moreover", "additionally", "in addition", "however",
    "therefore", "consequently", "in conclusion", "overall", "ultimately",
    "it is important to note", "it is essential", "it is important to recognize",
    "in today's", "fast-paced world", "plays a crucial role", "significantly enhance",
    "by prioritizing", "equally essential", "a wide range of", "it is worth noting",
]


def _words(t):
    return re.findall(r"[a-z']+", t.lower())


def _sentences(t):
    return [s.strip() for s in re.split(r"[.!?]+", t) if s.strip()]


def _clamp01(x):
    return max(0.0, min(1.0, x))


def _ngram_repeat_rate(toks, n):
    """Fraction of n-grams that are repeats (0 = all unique, higher = more repetition)."""
    if len(toks) < n + 1:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return 1 - (len(set(grams)) / len(grams))


def get_repetition_signal(text):
    """Return {"repetition_score": float, "features": {...}} for `text`."""
    toks = _words(text)
    sents = _sentences(text)
    low = text.lower()

    # --- raw features ---
    bigram_rep = _ngram_repeat_rate(toks, 2)
    trigram_rep = _ngram_repeat_rate(toks, 3)
    ngram_rep = max(bigram_rep, trigram_rep)  # take the stronger repetition signal

    openers = [s.split()[0].lower() for s in sents if s.split()]
    opener_variety = (len(set(openers)) / len(openers)) if openers else 1.0

    transition_hits = sum(low.count(p) for p in TRANSITION_PHRASES)
    transition_density = (transition_hits / len(toks)) if toks else 0.0

    # --- map each to 0-1 AI-likeness ---
    # Transition density: 0 -> human (0); >= 0.10 -> AI (1).
    trans_ai = _clamp01(transition_density / 0.10)
    # N-gram repetition: 0 -> human (0); >= 0.05 -> AI (1).
    ngram_ai = _clamp01(ngram_rep / 0.05)
    # Opener variety: 1.0 -> human (0); <= 0.4 -> AI (1).
    opener_ai = _clamp01((1.0 - opener_variety) / 0.6)

    repetition_score = W_TRANS * trans_ai + W_NGRAM * ngram_ai + W_OPENER * opener_ai

    return {
        "repetition_score": round(_clamp01(repetition_score), 4),
        "features": {
            "n_words": len(toks),
            "transition_hits": transition_hits,
            "transition_density": round(transition_density, 4),
            "ngram_repeat_rate": round(ngram_rep, 4),
            "opener_variety": round(opener_variety, 4),
            "sub_scores": {
                "transition_ai": round(trans_ai, 4),
                "ngram_ai": round(ngram_ai, 4),
                "opener_ai": round(opener_ai, 4),
            },
        },
    }
