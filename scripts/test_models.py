"""
Script : test_models.py
Description :
    Vérifie que le client API fonctionne correctement
    en appelant chaque modèle avec une question simple.

Utilisation :
    uv run python scripts/test_models.py
"""

from llm_eval.models import LLMClient, MODELS

if __name__ == "__main__":

    print("=" * 50)
    print("Test du client LLM")
    print("=" * 50)
    print(f"Modèles : {list(MODELS.keys())}\n")

    client    = LLMClient()
    responses = client.call_all(
        prompt = "In one sentence, what is a large language model?",
        delay  = 5.0,
    )

    print("\n" + "=" * 50)
    print("Résultats :")
    print("=" * 50)
    for r in responses:
        if r.success:
            print(f"\n{r.model_id} ({r.latency_s}s, {r.tokens_per_second} tok/s):")
            print(f"  {r.text[:150]}...")
        else:
            print(f"\n{r.model_id} : ERREUR — {r.error}")