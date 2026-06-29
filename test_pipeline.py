"""
Ensemble calibration test - the deliberately chosen inputs, with VOTES shown.

    python test_pipeline.py

Runs the full three-signal ensemble (LLM + stylometry + repetition) and prints
each signal's score, each signal's vote, the tally, the verdict, and confidence -
so you can see the deliberation, not just the final label.

Re-calibration checks (from planning.md > Stretch: Ensemble):
  1. clearly-human text  -> likely_human
  2. clearly-AI (signals agree) -> likely_ai
  3. formal / academic human (the false-positive trap) -> uncertain, never accused
"""

from dotenv import load_dotenv
from llm_signal import get_llm_signal
from stylometry import get_stylometric_signal
from repetition import get_repetition_signal
from scoring import vote_signals

load_dotenv()

CASES = {
    "1. CLEARLY AI": "Artificial intelligence represents a transformative paradigm shift in modern society. It is important to note that while the benefits of AI are numerous, it is equally essential to consider the ethical implications. Furthermore, stakeholders across various sectors must collaborate to ensure responsible deployment.",
    "2. CLEARLY HUMAN": "ok so i finally tried that new ramen place downtown and honestly? underwhelming. the broth was fine but they put WAY too much sodium in it and i was thirsty for like three hours after. my friend got the spicy version and said it was better. probably won't go back unless someone drags me there",
    "3. BORDERLINE: formal human": "The relationship between monetary policy and asset price inflation has been extensively studied in the literature. Central banks face a fundamental tension between their mandate for price stability and the unintended consequences of prolonged low interest rates on equity and real estate valuations.",
    "4. BORDERLINE: edited AI": "I've been thinking a lot about remote work lately. There are genuine tradeoffs — flexibility and no commute on one side, isolation and blurred work-life boundaries on the other. Studies show productivity varies widely by individual and role type.",
    "5. CLEARLY AI (signals agree)": "In today's fast-paced world, effective time management is essential for achieving success. By prioritizing tasks, setting clear goals, and maintaining focus, individuals can significantly enhance their productivity and overall well-being. Moreover, it is important to recognize that work-life balance plays a crucial role.",
}

if __name__ == "__main__":
    for label, text in CASES.items():
        n_words = len(text.split())
        s1 = get_llm_signal(text)
        s2 = get_stylometric_signal(text)
        s3 = get_repetition_signal(text)
        r = vote_signals(s1["llm_score"], s2["stylo_score"], s3["repetition_score"], n_words)
        v = r["votes"]

        print(f"\n{'='*72}\n{label}  ({n_words} words)")
        print(f"  scores : llm={s1['llm_score']:.2f}  stylo={s2['stylo_score']:.2f}  repetition={s3['repetition_score']:.2f}")
        print(f"  votes  : llm={v['llm']}  stylometry={v['stylometry']}  repetition={v['repetition']}")
        print(f"  tally  : {r['tally']}")
        print(f"  --> VERDICT = {r['verdict']}  (confidence {r['confidence']:.2f})")
