"""
Standalone test for Signal 1 - run BEFORE wiring it into the endpoint.

    python test_llm_signal.py

Inspect that clearly-human text scores low (human-like) and generic AI-sounding
text scores higher. This is the "test the function directly" step from Milestone 3.
"""

from dotenv import load_dotenv
from llm_signal import get_llm_signal

load_dotenv()

SAMPLES = {
    "human_casual": (
        "The sun dipped below the horizon, painting the sky in hues of amber and "
        "rose. I sat on the porch, coffee in hand, watching the neighborhood slowly "
        "go quiet."
    ),
    "ai_sounding": (
        "In today's fast-paced world, effective time management is essential for "
        "achieving success. By prioritizing tasks, setting clear goals, and "
        "maintaining focus, individuals can significantly enhance their productivity "
        "and overall well-being."
    ),
}

if __name__ == "__main__":
    for name, text in SAMPLES.items():
        result = get_llm_signal(text)
        print(f"\n[{name}]")
        print(f"  llm_score: {result['llm_score']:.2f}")
        print(f"  rationale: {result['rationale']}")
