"""
Script : app.py
Auteur : Franklin
Date   : 2026-04

Description :
    Dashboard Gradio interactif pour visualiser les résultats
    de l'évaluation des 5 LLMs sur les 3 tâches.

    L'interface propose :
        - Vue d'ensemble : heatmap comparatif de tous les modèles
        - Tâche 1 : graphique résumé scientifique
        - Tâche 2 : graphique QA + hallucination rate
        - Tâche 3 : graphique détection d'hallucinations
        - Latence  : comparaison des temps de réponse
        - Exemples : inspection des réponses individuelles

Utilisation :
    uv run python scripts/app.py
    Ouvrir http://127.0.0.1:7860
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import json
from pathlib import Path
import gradio as gr
from llm_eval.visualizer import Visualizer
from llm_eval.models import MODELS


# ==============================================================================
# INITIALISATION
# ==============================================================================

# On charge le visualizer une seule fois au démarrage
# Il lit tous les JSON de résultats dans results/
viz = Visualizer()

print("Dashboard initialisé. Chargement des résultats...")


# ==============================================================================
# FONCTIONS DU DASHBOARD
# ==============================================================================

def get_overview():
    """Retourne le heatmap vue d'ensemble."""
    return viz.plot_overview()


def get_summarization_plot():
    """Retourne le graphique de résumé."""
    return viz.plot_summarization()


def get_qa_plot():
    """Retourne le graphique QA."""
    return viz.plot_qa()


def get_hallucination_plot():
    """Retourne le graphique de détection."""
    return viz.plot_hallucination_detection()


def get_latency_plot():
    """Retourne le graphique de latence."""
    return viz.plot_latency()


def get_model_responses(model_id: str, task: str) -> str:
    """
    Retourne les réponses individuelles d'un modèle sur une tâche.

    Permet d'inspecter manuellement la qualité des réponses
    au-delà des métriques agrégées.

    Args:
        model_id : identifiant du modèle
        task     : nom de la tâche

    Returns:
        Texte formaté avec les réponses et références
    """

    # Mapping des noms affichés vers les identifiants internes
    task_map = {
        "Résumé scientifique"       : "summarization",
        "Question-Answering"        : "qa",
        "Détection d'hallucinations": "hallucination_detection",
    }
    task_key = task_map.get(task, "summarization")

    result_file = Path("results") / f"{model_id}_{task_key}.json"
    if not result_file.exists():
        return "Pas de résultats disponibles pour cette combinaison."

    data      = json.loads(result_file.read_text(encoding="utf-8"))
    responses = data.get("responses", [])

    if not responses:
        return "Aucune réponse trouvée."

    # Formatage des réponses pour l'affichage
    output = f"**Modèle : {model_id} | Tâche : {task}**\n"
    output += f"**Scores : {json.dumps(data['scores'], indent=2)}**\n\n"
    output += "─" * 60 + "\n\n"

    for i, resp in enumerate(responses[:3]):   # on affiche les 3 premiers
        output += f"**Exemple {i+1}**\n\n"

        if task_key == "summarization":
            output += f"**Résumé généré :**\n{resp.get('prediction', '')}\n\n"
            output += f"**Référence :**\n{resp.get('reference', '')}\n\n"
            output += f"Latence : {resp.get('latency_s', 0):.2f}s\n\n"

        elif task_key == "qa":
            output += f"**Question :** {resp.get('question', '')}\n\n"
            output += f"**Réponse générée :**\n{resp.get('prediction', '')}\n\n"
            output += f"**Référence :**\n{resp.get('reference', '')}\n\n"
            output += f"Hallucination rate : {resp.get('hallucination_rate', 0):.2f}\n\n"

        elif task_key == "hallucination_detection":
            output += f"**Erreurs détectées par le modèle :**\n{resp.get('prediction', '')}\n\n"
            output += f"**Vraies hallucinations :**\n"
            for h in resp.get("known_hallucinations", []):
                output += f"  - {h}\n"
            output += f"\nF1 : {resp.get('detection_f1', 0):.3f}\n\n"

        output += "─" * 40 + "\n\n"

    return output


# ==============================================================================
# CONSTRUCTION DU DASHBOARD
# ==============================================================================

def build_dashboard():
    """Construit l'interface Gradio."""

    with gr.Blocks(title="LLM Evaluation Framework") as dashboard:

        # En-tête
        gr.Markdown("""
        # LLM Evaluation Framework
        Comparaison de **5 LLMs open source** sur **3 tâches NLP** :
        résumé scientifique, question-answering et détection d'hallucinations.

        **Modèles évalués :** Llama 3.1 8B · Llama 3.3 70B · Llama 4 Scout · Qwen3 32B · GPT-OSS 120B

        **Métriques :** ROUGE · BERTScore · Hallucination Rate · Latence
        """)

        # Onglets
        with gr.Tabs():

            # Onglet 1 : Vue d'ensemble
            with gr.Tab("Vue d'ensemble"):
                gr.Markdown("### Heatmap comparatif — tous les modèles, toutes les métriques")
                gr.Markdown("Vert = bon score | Rouge = score faible | Les scores de hallucination sont inversés (bas = bien)")
                overview_plot = gr.Plot()
                gr.Button("Afficher").click(
                    fn      = get_overview,
                    outputs = overview_plot,
                )

            # Onglet 2 : Résumé
            with gr.Tab("Résumé scientifique"):
                gr.Markdown("### ROUGE-1 et BERTScore F1 par modèle")
                gr.Markdown("ROUGE-1 mesure le chevauchement de mots. BERTScore mesure la similarité sémantique.")
                sum_plot = gr.Plot()
                gr.Button("Afficher").click(
                    fn      = get_summarization_plot,
                    outputs = sum_plot,
                )

            # Onglet 3 : QA
            with gr.Tab("Question-Answering"):
                gr.Markdown("### BERTScore F1 et Hallucination Rate par modèle")
                gr.Markdown("BERTScore ↑ = meilleures réponses. Hallucination Rate ↓ = moins d'inventions.")
                qa_plot = gr.Plot()
                gr.Button("Afficher").click(
                    fn      = get_qa_plot,
                    outputs = qa_plot,
                )

            # Onglet 4 : Détection
            with gr.Tab("Détection d'hallucinations"):
                gr.Markdown("### F1 Score de détection par modèle")
                gr.Markdown("Mesure la capacité du modèle à identifier des affirmations incorrectes dans un texte.")
                det_plot = gr.Plot()
                gr.Button("Afficher").click(
                    fn      = get_hallucination_plot,
                    outputs = det_plot,
                )

            # Onglet 5 : Latence
            with gr.Tab("Latence"):
                gr.Markdown("### Temps de réponse moyen par modèle et par tâche")
                gr.Markdown("Cruciale en production : un modèle excellent mais lent peut être inutilisable.")
                lat_plot = gr.Plot()
                gr.Button("Afficher").click(
                    fn      = get_latency_plot,
                    outputs = lat_plot,
                )

            # Onglet 6 : Inspection des réponses
            with gr.Tab("Réponses individuelles"):
                gr.Markdown("### Inspection manuelle des réponses")
                gr.Markdown("Sélectionne un modèle et une tâche pour voir les réponses détaillées.")

                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices = list(MODELS.keys()),
                        value   = "llama-3.1-8b",
                        label   = "Modèle",
                    )
                    task_dropdown = gr.Dropdown(
                        choices = [
                            "Résumé scientifique",
                            "Question-Answering",
                            "Détection d'hallucinations",
                        ],
                        value = "Résumé scientifique",
                        label = "Tâche",
                    )

                inspect_btn    = gr.Button("Afficher les réponses", variant="primary")
                responses_text = gr.Markdown()

                inspect_btn.click(
                    fn      = get_model_responses,
                    inputs  = [model_dropdown, task_dropdown],
                    outputs = responses_text,
                )

    return dashboard


# ==============================================================================
# POINT D'ENTRÉE
# ==============================================================================

if __name__ == "__main__":

    dashboard = build_dashboard()
    dashboard.launch(
        share       = False,
        server_port = 7860,
        show_error  = True,
    )