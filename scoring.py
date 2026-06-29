"""
Ensemble voting + verdict.  [planning.md > Stretch: Ensemble detection (voting)]

Combines THREE signal scores (each on the axis 0 = human-like, 1 = AI-like) by
having each signal cast a vote, then tallying the votes with DELIBERATELY
ASYMMETRIC rules (it is harder to reach "AI" than "human").

How a signal votes (per-signal threshold on its 0-1 score):
    score >= VOTE_AI_AT     -> votes "ai"
    score <= VOTE_HUMAN_AT  -> votes "human"
    otherwise               -> "abstain"  (the signal can't tell)

Abstaining also absorbs the old short-text guard: stylometry and the repetition
signal are unreliable on very short text, so on < 40 words they abstain instead of
voting. (The LLM signal still votes — it does not depend on length the same way.)

Tally rules (asymmetric, because a false accusation is worse than a missed one):
    likely_ai     : at least 2 "ai" votes AND zero "human" votes
    likely_human  : "human" votes are the strict majority of cast (non-abstain) votes
    uncertain     : anything else (a split, too many abstentions, or AI votes that
                    are contradicted by a human vote)

Confidence reflects how DECISIVE the vote was, so the graded confidence from
Milestone 4 survives (we don't collapse to a blunt tally):
    unanimous (3 agree)        -> ~0.90
    2 agree + 1 abstain        -> ~0.78
    2 agree + 1 dissent        -> ~0.60  (-> usually lands uncertain anyway)
    mostly abstain / split     -> low
"""

# Per-signal vote thresholds.
VOTE_AI_AT = 0.60      # score >= this -> the signal votes "ai"
VOTE_HUMAN_AT = 0.40   # score <= this -> the signal votes "human"

SHORT_TEXT_WORDS = 40  # below this, structural signals (stylometry, repetition) abstain


def _vote(score):
    if score >= VOTE_AI_AT:
        return "ai"
    if score <= VOTE_HUMAN_AT:
        return "human"
    return "abstain"


def _confidence_from_votes(verdict, votes):
    """Confidence = how decisive the vote was (see module docstring)."""
    cast = [v for v in votes.values() if v != "abstain"]
    n_ai = list(votes.values()).count("ai")
    n_human = list(votes.values()).count("human")

    if verdict == "likely_ai":
        # decisiveness scales with how many agreed and whether anyone abstained
        if n_ai == 3:
            return 0.90
        if n_ai == 2:
            return 0.78
        return 0.70
    if verdict == "likely_human":
        if n_human == 3:
            return 0.90
        if n_human == 2:
            return 0.78
        return 0.70
    # uncertain: low confidence, lower the more the voters were split / silent
    if not cast:
        return 0.50
    agreement = abs(n_ai - n_human) / max(len(cast), 1)
    return round(0.45 + 0.10 * agreement, 4)  # ~0.45-0.55 band


def vote_signals(llm_score, stylo_score, repetition_score, n_words):
    """Ensemble-vote the three signals into a verdict.

    Returns:
      {
        "verdict": str,                 # likely_ai | uncertain | likely_human
        "confidence": float,            # confidence IN the verdict (decisiveness)
        "votes": {"llm":.., "stylometry":.., "repetition":..},
        "tally": {"ai":n, "human":n, "abstain":n},
      }
    """
    short = n_words < SHORT_TEXT_WORDS

    votes = {
        "llm": _vote(llm_score),
        # structural signals abstain on short text (old short-text guard)
        "stylometry": "abstain" if short else _vote(stylo_score),
        "repetition": "abstain" if short else _vote(repetition_score),
    }

    n_ai = list(votes.values()).count("ai")
    n_human = list(votes.values()).count("human")
    cast = n_ai + n_human

    # --- asymmetric tally ---
    if n_ai >= 2 and n_human == 0:
        verdict = "likely_ai"
    elif n_human > n_ai and n_human > (cast - n_human):
        # human votes are the strict majority of cast votes
        verdict = "likely_human"
    else:
        verdict = "uncertain"

    confidence = _confidence_from_votes(verdict, votes)

    return {
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "votes": votes,
        "tally": {"ai": n_ai, "human": n_human, "abstain": list(votes.values()).count("abstain")},
    }
