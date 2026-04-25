"""
Script : test_tasks.py
Description :
    Vérifie que les tâches sont bien définies et que les prompts
    sont bien générés.

Utilisation :
    uv run python scripts/test_tasks.py
"""

from llm_eval.tasks import (
    SUMMARIZATION_SAMPLES,
    QA_SAMPLES,
    HALLUCINATION_SAMPLES,
    get_summarization_prompt,
    get_qa_prompt,
    get_hallucination_prompt,
)

if __name__ == "__main__":

    print("=" * 55)
    print("Test des tâches d'évaluation")
    print("=" * 55)

    print(f"\nTâche 1 — Résumé scientifique : {len(SUMMARIZATION_SAMPLES)} exemples")
    print(f"Tâche 2 — Question-Answering  : {len(QA_SAMPLES)} exemples")
    print(f"Tâche 3 — Hallucinations      : {len(HALLUCINATION_SAMPLES)} exemples")

    print("\n--- Exemple prompt résumé ---")
    print(get_summarization_prompt(SUMMARIZATION_SAMPLES[0])[:300])

    print("\n--- Exemple prompt QA ---")
    print(get_qa_prompt(QA_SAMPLES[0])[:300])

    print("\n--- Exemple prompt hallucination ---")
    print(get_hallucination_prompt(HALLUCINATION_SAMPLES[0])[:300])

    print("\nTout est OK !")