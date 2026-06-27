"""
Milestone 4 calibration test - the four deliberately chosen inputs.

    python test_pipeline.py

Runs the FULL pipeline (LLM signal + stylometry + fusion) on a clearly-AI sample,
a clearly-human sample, and two borderline cases. Prints both signal scores
separately alongside the fused result, so if a verdict looks wrong you can see
which signal is responsible (per the milestone's debugging hint).

Expected shape of results:
  - clearly AI    -> high ai_likelihood, likely_ai
  - clearly human -> low ai_likelihood, likely_human
  - formal human  -> stylometry leans AI (the trap); fusion/asymmetry should keep
                     it OUT of a confident AI accusation (uncertain or human)
  - edited AI     -> mid-range / uncertain
  - clearly AI (signals agree) -> high ai_likelihood, likely_ai (demonstrates the
                     AI band is reachable when BOTH signals agree)
"""

from dotenv import load_dotenv
from llm_signal import get_llm_signal
from stylometry import get_stylometric_signal
from scoring import fuse_signals

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
        fused = fuse_signals(s1["llm_score"], s2["stylo_score"], n_words)

        print(f"\n{'='*70}\n{label}  ({n_words} words)")
        print(f"  llm_score   = {s1['llm_score']:.2f}   ({s1['rationale']})")
        print(f"  stylo_score = {s2['stylo_score']:.2f}   "
              f"(CV={s2['features']['sentence_length_cv']:.2f})")
        print(f"  --> ai_likelihood = {fused['ai_likelihood']:.2f}  "
              f"VERDICT = {fused['verdict']}  (confidence {fused['confidence']:.2f})")
        if fused["rules_applied"]:
            print(f"      rules fired: {', '.join(fused['rules_applied'])}")
