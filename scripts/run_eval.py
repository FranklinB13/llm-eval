"""
Script : run_eval.py
Description :
    Lance l'évaluation complète des 5 modèles sur les 3 tâches.

    Durée estimée : 30-60 minutes selon le rate limit Groq.
    Les résultats sont sauvegardés au fur et à mesure dans results/
    donc on peut interrompre et reprendre sans tout refaire.

Utilisation :
    uv run python scripts/run_eval.py
"""

from llm_eval.evaluator import Evaluator

if __name__ == "__main__":

    evaluator = Evaluator()

    # Lancement de l'évaluation complète
    # skip_existing=True : reprend là où on s'est arrêté si interrompu
    results = evaluator.run_all(skip_existing=True)

    # Affichage du résumé final
    evaluator.print_summary(results)

    print("\nRésultats sauvegardés dans results/")
    print("Lance scripts/app.py pour voir le dashboard !")