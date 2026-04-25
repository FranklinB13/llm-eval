"""
Module : visualizer.py
Auteur : Franklin
Date   : 2026-04

Description :
    Génération des graphiques comparatifs pour le dashboard.

    Ce module lit les résultats JSON sauvegardés par l'évaluateur
    et produit des graphiques matplotlib comparant les 5 modèles
    sur les 3 tâches.

    Graphiques produits :
        1. Radar chart      : vue globale de chaque modèle sur toutes les métriques
        2. Bar chart résumé : ROUGE-1 et BERTScore par modèle
        3. Bar chart QA     : BERTScore et hallucination rate par modèle
        4. Bar chart détect : F1 de détection par modèle
        5. Latence          : temps de réponse moyen par modèle et tâche
        6. Vitesse          : tokens/seconde par modèle

Utilisation :
    from llm_eval.visualizer import Visualizer
    viz = Visualizer()
    fig = viz.plot_summarization()
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from llm_eval.models import MODELS


# ==============================================================================
# CONFIGURATION VISUELLE
# ==============================================================================

# Couleurs distinctes pour chaque modèle
# On choisit des couleurs bien contrastées pour la lisibilité
MODEL_COLORS = {
    "llama-3.1-8b"  : "#2196F3",   # bleu
    "llama-3.3-70b" : "#4CAF50",   # vert
    "llama-4-scout" : "#FF9800",   # orange
    "qwen3-32b"     : "#E91E63",   # rose
    "gpt-oss-120b"  : "#9C27B0",   # violet
}

# Noms courts pour l'affichage dans les graphiques
MODEL_LABELS = {
    "llama-3.1-8b"  : "Llama 3.1\n8B",
    "llama-3.3-70b" : "Llama 3.3\n70B",
    "llama-4-scout" : "Llama 4\nScout",
    "qwen3-32b"     : "Qwen3\n32B",
    "gpt-oss-120b"  : "GPT-OSS\n120B",
}

# Style global matplotlib
plt.rcParams.update({
    "figure.facecolor" : "white",
    "axes.facecolor"   : "white",
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "font.size"        : 11,
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 11,
})


# ==============================================================================
# CLASSE VISUALIZER
# ==============================================================================

class Visualizer:
    """
    Génère les graphiques comparatifs à partir des résultats JSON.

    Utilisation :
        viz = Visualizer()
        fig = viz.plot_summarization()
        fig = viz.plot_qa()
        fig = viz.plot_hallucination_detection()
        fig = viz.plot_latency()
        fig = viz.plot_overview()
    """

    def __init__(self, results_dir: Path = Path("results")):
        """
        Charge tous les résultats JSON disponibles.

        Args:
            results_dir : dossier contenant les fichiers JSON des résultats
        """
        self.results_dir = results_dir
        self.results      = self._load_results()

    def _load_results(self) -> dict:
        """
        Charge tous les fichiers JSON de résultats dans un dictionnaire.

        Structure retournée :
            {
                "llama-3.1-8b": {
                    "summarization": {...scores...},
                    "qa": {...scores...},
                    "hallucination_detection": {...scores...},
                },
                ...
            }

        Returns:
            Dictionnaire des résultats par modèle et tâche
        """
        results = {}

        for model_id in MODELS.keys():
            results[model_id] = {}

            for task in ["summarization", "qa", "hallucination_detection"]:
                result_file = self.results_dir / f"{model_id}_{task}.json"

                if result_file.exists():
                    data = json.loads(result_file.read_text(encoding="utf-8"))
                    results[model_id][task] = data
                else:
                    results[model_id][task] = None

        return results

    def _get_models_with_data(self, task: str) -> list[str]:
        """
        Retourne les modèles qui ont des résultats pour une tâche donnée.

        Args:
            task : nom de la tâche

        Returns:
            Liste des model_id qui ont des données
        """
        return [
            m for m in MODELS.keys()
            if self.results[m].get(task) is not None
        ]

    def plot_summarization(self) -> plt.Figure:
        """
        Graphique comparatif pour la tâche de résumé scientifique.

        Affiche deux métriques côte à côte pour chaque modèle :
            - ROUGE-1 : chevauchement de mots avec la référence
            - BERTScore F1 : similarité sémantique

        On affiche les deux car elles sont complémentaires :
        ROUGE mesure la correspondance exacte, BERTScore le sens.
        Un modèle peut avoir un bon BERTScore mais un ROUGE faible
        s'il paraphrase bien mais avec des mots différents.

        Returns:
            Figure matplotlib
        """
        models = self._get_models_with_data("summarization")
        if not models:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Pas de données disponibles",
                    ha="center", va="center")
            return fig

        # Extraction des scores
        rouge1_scores = [
            self.results[m]["summarization"]["scores"].get("rouge1", 0)
            for m in models
        ]
        bertscore_scores = [
            self.results[m]["summarization"]["scores"].get("bertscore_f1", 0)
            for m in models
        ]

        # Création du graphique avec deux barres par modèle
        fig, ax = plt.subplots(figsize=(12, 6))

        x      = np.arange(len(models))
        width  = 0.35   # largeur de chaque barre

        # Barres ROUGE-1
        bars1 = ax.bar(
            x - width/2,
            rouge1_scores,
            width,
            label   = "ROUGE-1",
            color   = [MODEL_COLORS[m] for m in models],
            alpha   = 0.7,
        )

        # Barres BERTScore (plus transparentes pour les distinguer)
        bars2 = ax.bar(
            x + width/2,
            bertscore_scores,
            width,
            label   = "BERTScore F1",
            color   = [MODEL_COLORS[m] for m in models],
            alpha   = 1.0,
            hatch   = "//",   # hachures pour distinguer des ROUGE
        )

        # Annotations : valeur au dessus de chaque barre
        for bar in bars1:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=9,
            )
        for bar in bars2:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=9,
            )

        # Mise en forme
        ax.set_title("Tâche 1 : Résumé Scientifique", fontweight="bold")
        ax.set_ylabel("Score (0–1)")
        ax.set_ylim(0, 1.1)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in models])

        # Légende avec hachures
        legend_patches = [
            mpatches.Patch(facecolor="gray", alpha=0.7, label="ROUGE-1"),
            mpatches.Patch(facecolor="gray", hatch="//", label="BERTScore F1"),
        ]
        ax.legend(handles=legend_patches)

        plt.tight_layout()
        return fig

    def plot_qa(self) -> plt.Figure:
        """
        Graphique comparatif pour la tâche de QA.

        Affiche deux métriques :
            - BERTScore F1 : qualité de la réponse vs référence
            - Hallucination Rate : taux d'affirmations non fondées
              (inversé : 0 = pas d'hallucination = bien)

        Returns:
            Figure matplotlib
        """
        models = self._get_models_with_data("qa")
        if not models:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Pas de données disponibles",
                    ha="center", va="center")
            return fig

        bertscore_scores = [
            self.results[m]["qa"]["scores"].get("bertscore_f1", 0)
            for m in models
        ]
        hall_scores = [
            self.results[m]["qa"]["scores"].get("hallucination_rate", 0)
            for m in models
        ]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Graphique 1 : BERTScore
        colors = [MODEL_COLORS[m] for m in models]
        bars   = ax1.bar(
            [MODEL_LABELS[m] for m in models],
            bertscore_scores,
            color = colors,
        )
        for bar in bars:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=10,
            )
        ax1.set_title("BERTScore F1 (↑ mieux)")
        ax1.set_ylabel("Score (0–1)")
        ax1.set_ylim(0, 1.0)

        # Graphique 2 : Hallucination Rate
        # On utilise une couleur rouge pour symboliser le danger
        # Plus le score est bas, mieux c'est
        bars2 = ax2.bar(
            [MODEL_LABELS[m] for m in models],
            hall_scores,
            color = ["#ff4444" if s > 0.1 else "#44bb44" for s in hall_scores],
        )
        for bar in bars2:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=10,
            )
        ax2.set_title("Hallucination Rate (↓ mieux)")
        ax2.set_ylabel("Taux (0–1)")
        ax2.set_ylim(0, 0.5)

        fig.suptitle("Tâche 2 : Question-Answering avec Contexte",
                     fontweight="bold", fontsize=14)
        plt.tight_layout()
        return fig

    def plot_hallucination_detection(self) -> plt.Figure:
        """
        Graphique comparatif pour la tâche de détection d'hallucinations.

        Affiche précision, rappel et F1 pour chaque modèle.
        Le F1 est la métrique principale car il équilibre
        précision (pas de faux positifs) et rappel (toutes les
        hallucinations détectées).

        Returns:
            Figure matplotlib
        """
        models = self._get_models_with_data("hallucination_detection")
        if not models:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Pas de données disponibles",
                    ha="center", va="center")
            return fig

        f1_scores = [
            self.results[m]["hallucination_detection"]["scores"].get("detection_f1", 0)
            for m in models
        ]

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = [MODEL_COLORS[m] for m in models]
        bars   = ax.bar(
            [MODEL_LABELS[m] for m in models],
            f1_scores,
            color = colors,
        )

        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=11,
                fontweight="bold",
            )

        ax.set_title("Tâche 3 : Détection d'Hallucinations — F1 Score",
                     fontweight="bold")
        ax.set_ylabel("F1 Score (0–1)")
        ax.set_ylim(0, 1.1)

        # Ligne de référence à 0.8 (seuil "bon")
        ax.axhline(y=0.8, color="red", linestyle="--", alpha=0.5,
                   label="Seuil acceptable (0.8)")
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_latency(self) -> plt.Figure:
        """
        Graphique de la latence moyenne par modèle et par tâche.

        La latence est cruciale en production : un modèle excellent
        mais lent peut être inutilisable. Ce graphique permet de
        voir le trade-off qualité/vitesse.

        Returns:
            Figure matplotlib
        """
        tasks  = ["summarization", "qa", "hallucination_detection"]
        labels = ["Résumé", "QA", "Détection"]
        models = list(MODELS.keys())

        fig, ax = plt.subplots(figsize=(12, 6))

        x     = np.arange(len(tasks))
        width = 0.15   # largeur pour 5 modèles côte à côte

        for i, model_id in enumerate(models):
            latencies = []
            for task in tasks:
                data = self.results[model_id].get(task)
                if data:
                    latencies.append(data.get("latency_mean", 0))
                else:
                    latencies.append(0)

            ax.bar(
                x + i * width - 2 * width,
                latencies,
                width,
                label = model_id,
                color = MODEL_COLORS[model_id],
            )

        ax.set_title("Latence moyenne par tâche (secondes/requête)",
                     fontweight="bold")
        ax.set_ylabel("Latence (s)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(fontsize=9)

        plt.tight_layout()
        return fig

    def plot_overview(self) -> plt.Figure:
        """
        Vue d'ensemble : tableau récapitulatif de toutes les métriques.

        Affiche un heatmap coloré pour comparer facilement
        tous les modèles sur toutes les métriques d'un coup.
        Vert = bon, Rouge = mauvais (relatif aux autres modèles).

        Returns:
            Figure matplotlib
        """
        models  = list(MODELS.keys())
        metrics = [
            ("summarization", "rouge1",           "Résumé\nROUGE-1",     True),
            ("summarization", "bertscore_f1",      "Résumé\nBERTScore",   True),
            ("qa",            "bertscore_f1",      "QA\nBERTScore",       True),
            ("qa",            "hallucination_rate","QA\nHallucination",   False),
            ("hallucination_detection", "detection_f1", "Détection\nF1",  True),
        ]
        # True = plus c'est haut mieux c'est
        # False = plus c'est bas mieux c'est (hallucination rate)

        # Construction de la matrice de scores
        data = []
        for model_id in models:
            row = []
            for task, metric, _, higher_is_better in metrics:
                result = self.results[model_id].get(task)
                if result:
                    score = result["scores"].get(metric, 0)
                    # Pour hallucination rate, on inverse pour que
                    # vert = bien dans le heatmap
                    if not higher_is_better:
                        score = 1 - score
                else:
                    score = 0
                row.append(score)
            data.append(row)

        data_array = np.array(data)

        # Création du heatmap
        fig, ax = plt.subplots(figsize=(12, 5))

        im = ax.imshow(data_array, cmap="RdYlGn", aspect="auto",
                       vmin=0, vmax=1)

        # Axes
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([m[2] for m in metrics], fontsize=10)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([MODEL_LABELS[m] for m in models], fontsize=10)

        # Valeurs dans chaque cellule
        for i in range(len(models)):
            for j in range(len(metrics)):
                task, metric, _, higher_is_better = metrics[j]
                result = self.results[models[i]].get(task)
                if result:
                    raw_score = result["scores"].get(metric, 0)
                    # On affiche le score original (pas inversé)
                    text = ax.text(j, i, f"{raw_score:.3f}",
                                   ha="center", va="center",
                                   color="black", fontsize=10,
                                   fontweight="bold")

        # Colorbar
        plt.colorbar(im, ax=ax, shrink=0.8, label="Score normalisé")

        ax.set_title("Vue d'ensemble — Comparaison des 5 modèles",
                     fontweight="bold", pad=15)

        plt.tight_layout()
        return fig