"""
Module : evaluator.py
Auteur : Franklin
Date   : 2026-04

Description :
    Orchestration du pipeline d'évaluation complet.

    Ce module est le chef d'orchestre du projet. Il connecte :
        - LLMClient (models.py)    : appels API vers les 5 modèles
        - Tasks + Dataset (tasks.py) : les 3 tâches et les données
        - Métriques (metrics.py)   : ROUGE, BERTScore, hallucination rate

    Pour chaque modèle et chaque tâche, l'évaluateur :
        1. Génère les réponses via l'API Groq
        2. Calcule les métriques appropriées
        3. Sauvegarde les résultats en JSON

    Structure des résultats :
        {
            "model_id": "llama-3.1-8b",
            "task": "summarization",
            "scores": {
                "rouge1": 0.42,
                "rouge2": 0.18,
                "rougeL": 0.38,
                "bertscore_f1": 0.87,
            },
            "latency_mean": 1.2,
            "tokens_per_second_mean": 45.0,
            "responses": [...]
        }

    Idempotence :
        Si un fichier de résultats existe déjà pour un (modèle, tâche),
        on le skipte. On peut interrompre et reprendre sans tout refaire.

Utilisation :
    uv run python scripts/run_eval.py
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from tqdm import tqdm                    # barres de progression

from llm_eval.models import LLMClient, MODELS, ModelResponse
from llm_eval.tasks import (
    SUMMARIZATION_SAMPLES, QA_SAMPLES, HALLUCINATION_SAMPLES,
    get_summarization_prompt, get_qa_prompt, get_hallucination_prompt,
    SummarizationSample, QASample, HallucinationSample,
)
from llm_eval.metrics import (
    compute_rouge,
    compute_bertscore,
    compute_hallucination_rate,
    compute_hallucination_detection_score,
)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Dossier de sauvegarde des résultats
RESULTS_DIR = Path("results")

# Délai entre les appels API pour respecter le rate limit Groq
# 5s entre chaque modèle sur la même tâche
INTER_MODEL_DELAY = 5.0

# Délai entre les exemples d'une même tâche
INTER_SAMPLE_DELAY = 3.0

# Modèle juge pour le hallucination rate
JUDGE_MODEL = "gpt-oss-120b"


# ==============================================================================
# STRUCTURE DE DONNÉES : EvalResult
# ==============================================================================

@dataclass
class EvalResult:
    """
    Résultat d'évaluation pour un modèle sur une tâche.

    Cette structure est sérialisée en JSON et sauvegardée sur disque.
    Elle contient tout ce dont on a besoin pour le dashboard :
        - les scores agrégés (pour les graphiques)
        - les réponses individuelles (pour l'inspection manuelle)
        - les métriques de performance (latence, vitesse)

    Champs :
        model_id          : identifiant court du modèle
        task              : nom de la tâche ("summarization", "qa", "hallucination")
        scores            : dictionnaire des métriques agrégées
        latency_mean      : latence moyenne en secondes
        tok_per_sec_mean  : vitesse moyenne en tokens/seconde
        n_samples         : nombre d'exemples évalués
        responses         : liste des réponses individuelles avec détails
    """
    model_id        : str
    task            : str
    scores          : dict
    latency_mean    : float
    tok_per_sec_mean: float
    n_samples       : int
    responses       : list

    def save(self, results_dir: Path = RESULTS_DIR) -> Path:
        """
        Sauvegarde le résultat en JSON.

        Nom du fichier : {model_id}_{task}.json
        Exemple : llama-3.1-8b_summarization.json

        Args:
            results_dir : dossier de sauvegarde

        Returns:
            Chemin du fichier sauvegardé
        """
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"{self.model_id}_{self.task}.json"
        output_path.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path


# ==============================================================================
# ÉVALUATION TÂCHE 1 : RÉSUMÉ SCIENTIFIQUE
# ==============================================================================

def evaluate_summarization(
    model_id : str,
    client   : LLMClient,
) -> EvalResult:
    """
    Évalue un modèle sur la tâche de résumé scientifique.

    Pipeline :
        Pour chaque article dans SUMMARIZATION_SAMPLES :
            1. Génère un résumé via le modèle
            2. Calcule ROUGE entre résumé généré et référence
        Calcule BERTScore sur tous les résumés en batch
        Retourne les scores agrégés

    Pourquoi ROUGE + BERTScore ensemble ?
        ROUGE est rapide et standard — tout le monde le comprend.
        BERTScore capture les synonymes que ROUGE rate.
        Les deux ensemble donnent une image plus complète de la qualité.

    Args:
        model_id : identifiant du modèle à évaluer
        client   : client LLM initialisé

    Returns:
        EvalResult avec les scores ROUGE et BERTScore moyens
    """

    predictions = []   # résumés générés par le modèle
    references  = []   # résumés de référence (ground truth)
    latencies   = []   # temps de réponse par exemple
    tok_speeds  = []   # vitesse de génération par exemple
    responses   = []   # détails de chaque réponse (pour l'inspection)

    print(f"\n  Résumé scientifique — {model_id}")

    # Barre de progression sur les exemples
    for sample in tqdm(SUMMARIZATION_SAMPLES, desc="  Exemples", leave=False):

        # Génération du résumé
        prompt   = get_summarization_prompt(sample)
        response = client.call(model_id, prompt)

        if response.success:
            # Nettoyage : qwen3 ajoute des balises <think>...</think>
            # On les supprime avant de calculer les métriques
            text = clean_model_output(response.text)

            predictions.append(text)
            references.append(sample.reference)
            latencies.append(response.latency_s)
            tok_speeds.append(response.tokens_per_second)

            responses.append({
                "sample_id" : sample.id,
                "topic"     : sample.topic,
                "prediction": text,
                "reference" : sample.reference,
                "latency_s" : response.latency_s,
            })
        else:
            print(f"\n    Erreur sur {sample.id} : {response.error[:60]}")

        # Délai entre exemples
        time.sleep(INTER_SAMPLE_DELAY)

    # Calcul des métriques ROUGE (rapide, sur tous les exemples)
    rouge_scores = {}
    if predictions:
        from llm_eval.metrics import compute_rouge_batch
        rouge_scores = compute_rouge_batch(predictions, references)

    # Calcul BERTScore (plus lent, fait en batch pour l'efficacité)
    bertscore = {}
    if predictions:
        print(f"    Calcul BERTScore ({len(predictions)} exemples)...")
        bertscore = compute_bertscore(predictions, references)

    # Agrégation de toutes les métriques
    scores = {**rouge_scores, **bertscore}

    return EvalResult(
        model_id         = model_id,
        task             = "summarization",
        scores           = scores,
        latency_mean     = round(sum(latencies) / len(latencies), 3) if latencies else 0,
        tok_per_sec_mean = round(sum(tok_speeds) / len(tok_speeds), 1) if tok_speeds else 0,
        n_samples        = len(predictions),
        responses        = responses,
    )


# ==============================================================================
# ÉVALUATION TÂCHE 2 : QUESTION-ANSWERING
# ==============================================================================

def evaluate_qa(
    model_id : str,
    client   : LLMClient,
) -> EvalResult:
    """
    Évalue un modèle sur la tâche de question-answering avec contexte.

    Pipeline :
        Pour chaque exemple QA :
            1. Génère une réponse en s'appuyant sur le contexte
            2. Calcule BERTScore (réponse vs référence)
            3. Calcule le hallucination rate (réponse vs contexte)

    Pourquoi BERTScore + hallucination rate ensemble ?
        BERTScore mesure si la réponse dit la même chose que la référence.
        Hallucination rate mesure si la réponse invente des informations
        non présentes dans le contexte.
        Les deux sont complémentaires pour évaluer un système RAG.

    Args:
        model_id : identifiant du modèle
        client   : client LLM

    Returns:
        EvalResult avec BERTScore et hallucination rate moyens
    """

    predictions     = []
    references      = []
    contexts        = []
    latencies       = []
    tok_speeds      = []
    hall_scores     = []
    responses       = []

    print(f"\n  Question-Answering — {model_id}")

    for sample in tqdm(QA_SAMPLES, desc="  Exemples", leave=False):

        prompt   = get_qa_prompt(sample)
        response = client.call(model_id, prompt)

        if response.success:
            text = clean_model_output(response.text)

            predictions.append(text)
            references.append(sample.reference)
            contexts.append(sample.context)
            latencies.append(response.latency_s)
            tok_speeds.append(response.tokens_per_second)

            # Calcul du hallucination rate pour cet exemple
            hall_score = compute_hallucination_rate(
                context  = sample.context,
                response = text,
                client   = client,
            )
            hall_scores.append(hall_score)
            time.sleep(2.0)   # délai après l'appel au juge

            responses.append({
                "sample_id"        : sample.id,
                "topic"            : sample.topic,
                "question"         : sample.question,
                "prediction"       : text,
                "reference"        : sample.reference,
                "hallucination_rate": hall_score,
                "latency_s"        : response.latency_s,
            })

        time.sleep(INTER_SAMPLE_DELAY)

    # BERTScore en batch
    bertscore = {}
    if predictions:
        print(f"    Calcul BERTScore ({len(predictions)} exemples)...")
        bertscore = compute_bertscore(predictions, references)

    # Hallucination rate moyen
    hall_mean = round(sum(hall_scores) / len(hall_scores), 4) if hall_scores else 0.0

    scores = {
        **bertscore,
        "hallucination_rate": hall_mean,
    }

    return EvalResult(
        model_id         = model_id,
        task             = "qa",
        scores           = scores,
        latency_mean     = round(sum(latencies) / len(latencies), 3) if latencies else 0,
        tok_per_sec_mean = round(sum(tok_speeds) / len(tok_speeds), 1) if tok_speeds else 0,
        n_samples        = len(predictions),
        responses        = responses,
    )


# ==============================================================================
# ÉVALUATION TÂCHE 3 : DÉTECTION D'HALLUCINATIONS
# ==============================================================================

def evaluate_hallucination_detection(
    model_id : str,
    client   : LLMClient,
) -> EvalResult:
    """
    Évalue la capacité d'un modèle à détecter des hallucinations.

    Pipeline :
        Pour chaque exemple :
            1. On donne au modèle un texte contenant des hallucinations
            2. On lui demande de lister les erreurs factuelles
            3. On vérifie si les vraies hallucinations ont été détectées

    C'est une évaluation de la "conscience factuelle" du modèle —
    sa capacité à raisonner sur la vérité d'un texte.

    Args:
        model_id : identifiant du modèle
        client   : client LLM

    Returns:
        EvalResult avec précision, rappel et F1 de détection
    """

    precision_scores = []
    recall_scores    = []
    f1_scores        = []
    latencies        = []
    tok_speeds       = []
    responses        = []

    print(f"\n  Détection d'hallucinations — {model_id}")

    for sample in tqdm(HALLUCINATION_SAMPLES, desc="  Exemples", leave=False):

        prompt   = get_hallucination_prompt(sample)
        response = client.call(model_id, prompt)

        if response.success:
            text = clean_model_output(response.text)
            latencies.append(response.latency_s)
            tok_speeds.append(response.tokens_per_second)

            # Évaluation de la détection
            det_scores = compute_hallucination_detection_score(
                predicted_text       = text,
                known_hallucinations = sample.hallucinations,
            )
            precision_scores.append(det_scores["precision"])
            recall_scores.append(det_scores["recall"])
            f1_scores.append(det_scores["f1"])

            responses.append({
                "sample_id"           : sample.id,
                "prediction"          : text,
                "known_hallucinations": sample.hallucinations,
                "detection_f1"        : det_scores["f1"],
                "latency_s"           : response.latency_s,
            })

        time.sleep(INTER_SAMPLE_DELAY)

    def mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    scores = {
        "detection_precision": mean(precision_scores),
        "detection_recall"   : mean(recall_scores),
        "detection_f1"       : mean(f1_scores),
    }

    return EvalResult(
        model_id         = model_id,
        task             = "hallucination_detection",
        scores           = scores,
        latency_mean     = mean(latencies),
        tok_per_sec_mean = mean(tok_speeds),
        n_samples        = len(responses),
        responses        = responses,
    )


# ==============================================================================
# FONCTION UTILITAIRE : NETTOYAGE DES RÉPONSES
# ==============================================================================

def clean_model_output(text: str) -> str:
    """
    Nettoie la réponse d'un modèle avant de calculer les métriques.

    Problème observé : qwen3-32b génère des balises <think>...</think>
    qui contiennent son raisonnement interne avant la réponse finale.
    Ces balises faussent les métriques car elles ne font pas partie
    de la réponse.

    On supprime également les espaces en début et fin de texte.

    Args:
        text : réponse brute du modèle

    Returns:
        Texte nettoyé
    """
    import re

    # Suppression des balises <think>...</think> (qwen3, deepseek)
    # re.DOTALL : le point matche aussi les sauts de ligne
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Suppression des espaces et sauts de ligne en début/fin
    text = text.strip()

    return text


# ==============================================================================
# PIPELINE D'ÉVALUATION COMPLET
# ==============================================================================

class Evaluator:
    """
    Orchestre l'évaluation complète des 5 modèles sur les 3 tâches.

    Utilisation :
        evaluator = Evaluator()
        results   = evaluator.run_all()
        evaluator.print_summary(results)
    """

    def __init__(self, results_dir: Path = RESULTS_DIR):
        """
        Initialise l'évaluateur.

        Args:
            results_dir : dossier de sauvegarde des résultats
        """
        self.client      = LLMClient()
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_model(self, model_id: str) -> list[EvalResult]:
        """
        Évalue un modèle sur les 3 tâches.

        Args:
            model_id : identifiant du modèle à évaluer

        Returns:
            Liste de 3 EvalResult (un par tâche)
        """
        results = []

        # Tâche 1 : résumé scientifique
        result = evaluate_summarization(model_id, self.client)
        result.save(self.results_dir)
        results.append(result)
        print(f"    ✅ Résumé — ROUGE-1: {result.scores.get('rouge1', 0):.3f}")
        time.sleep(INTER_MODEL_DELAY)

        # Tâche 2 : question-answering
        result = evaluate_qa(model_id, self.client)
        result.save(self.results_dir)
        results.append(result)
        print(f"    ✅ QA — BERTScore F1: {result.scores.get('bertscore_f1', 0):.3f}")
        time.sleep(INTER_MODEL_DELAY)

        # Tâche 3 : détection d'hallucinations
        result = evaluate_hallucination_detection(model_id, self.client)
        result.save(self.results_dir)
        results.append(result)
        print(f"    ✅ Hallucination — F1: {result.scores.get('detection_f1', 0):.3f}")

        return results

    def run_all(self, skip_existing: bool = True) -> list[EvalResult]:
        """
        Évalue tous les modèles sur toutes les tâches.

        Idempotence :
            Si skip_existing=True, on skippe les (modèle, tâche)
            dont les résultats existent déjà. Permet de reprendre
            une évaluation interrompue.

        Args:
            skip_existing : True pour skipper les résultats existants

        Returns:
            Liste de tous les EvalResult
        """
        all_results = []

        print("=" * 55)
        print("Évaluation complète — 5 modèles × 3 tâches")
        print("=" * 55)

        for model_id in tqdm(MODELS.keys(), desc="Modèles"):

            print(f"\n{'─'*40}")
            print(f"Modèle : {model_id}")
            print(f"{'─'*40}")

            model_results = []

            for task_name in ["summarization", "qa", "hallucination_detection"]:

                # Idempotence : vérification si le résultat existe
                result_file = self.results_dir / f"{model_id}_{task_name}.json"
                if skip_existing and result_file.exists():
                    print(f"  ⏭️  {task_name} déjà calculé, skip")
                    # On charge le résultat existant
                    data       = json.loads(result_file.read_text(encoding="utf-8"))
                    result     = EvalResult(**data)
                    model_results.append(result)
                    continue

                # Évaluation selon la tâche
                if task_name == "summarization":
                    result = evaluate_summarization(model_id, self.client)
                elif task_name == "qa":
                    result = evaluate_qa(model_id, self.client)
                else:
                    result = evaluate_hallucination_detection(model_id, self.client)

                result.save(self.results_dir)
                model_results.append(result)
                time.sleep(INTER_MODEL_DELAY)

            all_results.extend(model_results)

        return all_results

    def print_summary(self, results: list[EvalResult]) -> None:
        """
        Affiche un tableau récapitulatif des résultats.

        Args:
            results : liste de tous les EvalResult
        """

        print("\n" + "=" * 70)
        print("RÉSULTATS FINAUX")
        print("=" * 70)

        # Résumé par modèle
        for model_id in MODELS.keys():

            model_results = [r for r in results if r.model_id == model_id]
            if not model_results:
                continue

            print(f"\n{model_id}")

            for result in model_results:

                if result.task == "summarization":
                    r1  = result.scores.get("rouge1", 0)
                    bs  = result.scores.get("bertscore_f1", 0)
                    print(f"  Résumé     : ROUGE-1={r1:.3f} | BERTScore={bs:.3f} | {result.latency_mean:.2f}s/req")

                elif result.task == "qa":
                    bs   = result.scores.get("bertscore_f1", 0)
                    hall = result.scores.get("hallucination_rate", 0)
                    print(f"  QA         : BERTScore={bs:.3f} | Hallucination={hall:.2f} | {result.latency_mean:.2f}s/req")

                elif result.task == "hallucination_detection":
                    f1 = result.scores.get("detection_f1", 0)
                    print(f"  Détection  : F1={f1:.3f} | {result.latency_mean:.2f}s/req")

        print("\n" + "=" * 70)