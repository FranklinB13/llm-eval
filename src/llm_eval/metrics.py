"""
Module : metrics.py
Auteur : Franklin
Date   : 2026-04

Description :
    Calcul des métriques d'évaluation pour comparer les 5 LLMs.

    Métriques implémentées :

        ROUGE (Recall-Oriented Understudy for Gisting Evaluation) :
            Mesure le chevauchement de n-grammes entre la réponse
            générée et la référence.
            - ROUGE-1 : unigrammes (mots individuels)
            - ROUGE-2 : bigrammes (paires de mots consécutifs)
            - ROUGE-L : plus longue sous-séquence commune
            Avantage : rapide, interprétable, standard pour summarisation
            Limite : sensible aux synonymes (ne comprend pas le sens)

        BERTScore :
            Mesure la similarité sémantique via des embeddings BERT.
            Compare chaque token de la prédiction avec chaque token
            de la référence et calcule la F1 sur les meilleures correspondances.
            Avantage : comprend les synonymes et paraphrases
            Limite : plus lent que ROUGE, moins interprétable

        Hallucination Rate (LLM-as-a-judge) :
            Utilise un LLM juge (gpt-oss-120b) pour évaluer si la réponse
            contient des affirmations non supportées par le contexte.
            Score entre 0 (aucune hallucination) et 1 (tout est inventé).
            Avantage : évaluation qualitative proche du jugement humain
            Limite : dépend de la qualité du modèle juge, coûte des tokens

        Hallucination Detection Score :
            Pour la tâche 3, mesure si le modèle détecte correctement
            les hallucinations listées. Calcule précision et rappel
            en vérifiant si les hallucinations connues sont mentionnées.

Utilisation :
    from llm_eval.metrics import compute_rouge, compute_bertscore
    from llm_eval.metrics import compute_hallucination_rate
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import re                           # expressions régulières pour le parsing
import time                         # délais pour le rate limit
from rouge_score import rouge_scorer  # calcul ROUGE
from bert_score import score as bert_score_fn  # calcul BERTScore
from llm_eval.models import LLMClient  # pour le LLM juge


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Modèle utilisé comme juge pour évaluer les hallucinations
# On choisit le plus puissant disponible sur Groq pour avoir
# le jugement le plus fiable possible
JUDGE_MODEL = "gpt-oss-120b"

# Modèle BERT utilisé pour BERTScore
# "distilbert-base-uncased" : version légère, rapide, suffisante
# Alternative : "roberta-large" (meilleure qualité mais plus lente)
BERTSCORE_MODEL = "distilbert-base-uncased"

# Délai entre appels au LLM juge (rate limit Groq)
JUDGE_DELAY = 5.0


# ==============================================================================
# MÉTRIQUE 1 : ROUGE
# ==============================================================================

def compute_rouge(
    prediction : str,
    reference  : str,
) -> dict[str, float]:
    """
    Calcule les scores ROUGE entre une prédiction et une référence.

    ROUGE mesure le chevauchement de n-grammes :
        ROUGE-1 : compare les mots individuels
                  Exemple : "The cat sat" vs "The dog sat"
                  Commun : "The", "sat" → 2/3 = 0.67
        ROUGE-2 : compare les paires de mots consécutifs
                  Exemple : "The cat sat" → bigrammes : ("The cat"), ("cat sat")
                  Plus sélectif que ROUGE-1, meilleur signal de qualité
        ROUGE-L : compare la plus longue sous-séquence commune
                  Capture la structure de la phrase mieux que ROUGE-1/2

    La F-mesure (F1) est le score principal car elle équilibre
    précision (est-ce que les mots générés sont dans la référence ?)
    et rappel (est-ce que les mots de la référence sont générés ?).

    Args:
        prediction : texte généré par le modèle
        reference  : texte de référence (ground truth)

    Returns:
        Dictionnaire avec rouge1, rouge2, rougeL (F1 scores entre 0 et 1)
    """

    # Initialisation du scorer ROUGE
    # use_stemmer=True : réduit les mots à leur racine
    # ("running" et "runs" deviennent "run")
    # Améliore le score pour les variations morphologiques
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    # Calcul des scores
    # scorer.score() retourne un dict avec (precision, recall, fmeasure)
    # pour chaque métrique
    scores = scorer.score(reference, prediction)

    # On retourne uniquement la F-mesure (fmeasure)
    # C'est le score standard reporté dans les papers
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def compute_rouge_batch(
    predictions : list[str],
    references  : list[str],
) -> dict[str, float]:
    """
    Calcule les scores ROUGE moyens sur plusieurs exemples.

    On traite une liste de paires (prédiction, référence) et on
    retourne la moyenne des scores — c'est ce qu'on reporte
    dans les tableaux de résultats.

    Args:
        predictions : liste de textes générés par le modèle
        references  : liste de références correspondantes

    Returns:
        Scores ROUGE moyens sur tous les exemples
    """

    if len(predictions) != len(references):
        raise ValueError(
            f"Nombre de prédictions ({len(predictions)}) "
            f"différent du nombre de références ({len(references)})"
        )

    # Calcul des scores pour chaque paire
    all_rouge1 = []
    all_rouge2 = []
    all_rougeL = []

    for pred, ref in zip(predictions, references):
        scores = compute_rouge(pred, ref)
        all_rouge1.append(scores["rouge1"])
        all_rouge2.append(scores["rouge2"])
        all_rougeL.append(scores["rougeL"])

    # Calcul des moyennes
    return {
        "rouge1": round(sum(all_rouge1) / len(all_rouge1), 4),
        "rouge2": round(sum(all_rouge2) / len(all_rouge2), 4),
        "rougeL": round(sum(all_rougeL) / len(all_rougeL), 4),
    }


# ==============================================================================
# MÉTRIQUE 2 : BERTScore
# ==============================================================================

def compute_bertscore(
    predictions : list[str],
    references  : list[str],
    model_type  : str = BERTSCORE_MODEL,
) -> dict[str, float]:
    """
    Calcule le BERTScore moyen sur une liste de paires.

    BERTScore compare les embeddings des tokens entre prédiction
    et référence. Pour chaque token de la prédiction, on trouve
    le token de la référence le plus similaire (cosinus).

    Exemple :
        Prédiction : "The automobile is red"
        Référence  : "The car is red"
        ROUGE-1 score : 3/4 = 0.75 ("automobile" ≠ "car")
        BERTScore : ~0.95 (BERT sait que "automobile" ≈ "car")

    Pourquoi distilbert-base-uncased ?
        C'est un bon équilibre entre qualité et vitesse.
        Il tourne sur CPU en quelques secondes.
        Pour un usage de recherche, roberta-large est meilleur
        mais 10x plus lent.

    Args:
        predictions : textes générés par le modèle
        references  : textes de référence
        model_type  : modèle BERT à utiliser pour les embeddings

    Returns:
        Dictionnaire avec precision, recall, f1 moyens
    """

    # bert_score_fn calcule les scores pour toute la liste d'un coup
    # C'est plus efficace que de calculer un par un (batch processing)
    # verbose=False : supprime les barres de progression
    precision, recall, f1 = bert_score_fn(
        cands      = predictions,
        refs       = references,
        model_type = model_type,
        verbose    = False,
    )

    # Les scores sont des tenseurs PyTorch → on les convertit en float
    return {
        "bertscore_precision": round(precision.mean().item(), 4),
        "bertscore_recall"   : round(recall.mean().item(), 4),
        "bertscore_f1"       : round(f1.mean().item(), 4),
    }


# ==============================================================================
# MÉTRIQUE 3 : HALLUCINATION RATE (LLM-as-a-judge)
# ==============================================================================

# Prompt du LLM juge pour évaluer le taux d'hallucination
# On demande un score numérique simple (0-10) pour faciliter le parsing
# "ONLY output a single integer" : on veut éviter les réponses verboses
HALLUCINATION_JUDGE_PROMPT = """You are an expert fact-checker evaluating AI-generated responses.

Given the context and the AI response, rate how many factual claims in the response are NOT supported by or contradict the context.

Context:
{context}

AI Response:
{response}

Rate the hallucination level from 0 to 10:
- 0: No hallucinations, all claims are supported by the context
- 5: About half the claims are hallucinated or unsupported
- 10: All claims are hallucinated or contradict the context

ONLY output a single integer from 0 to 10. Nothing else."""


def compute_hallucination_rate(
    context   : str,
    response  : str,
    client    : LLMClient,
) -> float:
    """
    Évalue le taux d'hallucination d'une réponse via un LLM juge.

    Approche LLM-as-a-judge :
        On utilise un modèle puissant (gpt-oss-120b) comme juge
        pour évaluer si la réponse générée par un autre modèle
        contient des hallucinations par rapport au contexte.

        Cette approche est très utilisée en recherche et en production
        car elle corrèle bien avec le jugement humain.
        Elle est beaucoup plus scalable que l'annotation humaine.

    Limitation :
        Le juge peut lui-même se tromper. On utilise le modèle le plus
        fort disponible (gpt-oss-120b) pour minimiser ce risque.

    Args:
        context  : texte de contexte de référence
        response : réponse générée par le modèle évalué
        client   : client LLM pour appeler le modèle juge

    Returns:
        Score entre 0.0 (pas d'hallucination) et 1.0 (tout inventé)
    """

    # Construction du prompt pour le juge
    prompt = HALLUCINATION_JUDGE_PROMPT.format(
        context  = context,
        response = response,
    )

    # Appel au modèle juge
    judge_response = client.call(JUDGE_MODEL, prompt)

    if not judge_response.success:
        print(f"    Erreur juge : {judge_response.error}")
        return 0.5   # valeur neutre en cas d'erreur

    # Parsing de la réponse : on cherche un entier entre 0 et 10
    # re.findall extrait tous les nombres du texte
    # On prend le premier trouvé
    text    = judge_response.text.strip()
    numbers = re.findall(r"\b(\d+)\b", text)

    if not numbers:
        return 0.5   # pas de nombre trouvé

    score = int(numbers[0])
    score = max(0, min(10, score))   # borné entre 0 et 10

    # Normalisation : 0-10 → 0.0-1.0
    return round(score / 10, 2)


def compute_hallucination_rate_batch(
    contexts  : list[str],
    responses : list[str],
    client    : LLMClient,
    delay     : float = JUDGE_DELAY,
) -> float:
    """
    Calcule le taux d'hallucination moyen sur plusieurs exemples.

    On traite les exemples un par un avec un délai entre chaque
    pour respecter le rate limit de l'API Groq.

    Args:
        contexts  : liste de contextes de référence
        responses : liste de réponses à évaluer
        client    : client LLM
        delay     : délai entre chaque appel au juge (secondes)

    Returns:
        Taux d'hallucination moyen entre 0.0 et 1.0
    """

    scores = []

    for i, (ctx, resp) in enumerate(zip(contexts, responses)):
        score = compute_hallucination_rate(ctx, resp, client)
        scores.append(score)
        if i < len(contexts) - 1:
            time.sleep(delay)

    return round(sum(scores) / len(scores), 4) if scores else 0.0


# ==============================================================================
# MÉTRIQUE 4 : HALLUCINATION DETECTION SCORE
# ==============================================================================

def compute_hallucination_detection_score(
    predicted_text    : str,
    known_hallucinations : list[str],
) -> dict[str, float]:
    """
    Évalue si le modèle a détecté les hallucinations connues.

    Pour la tâche 3, on sait exactement quelles hallucinations
    sont présentes dans le texte. On vérifie si la réponse du modèle
    mentionne chacune d'elles.

    Approche :
        Pour chaque hallucination connue, on vérifie si un mot-clé
        distinctif apparaît dans la réponse du modèle.
        C'est une approximation : une détection parfaite nécessiterait
        du NLI (Natural Language Inference).

    Métriques calculées :
        Précision : parmi les erreurs signalées, combien sont réelles ?
        Rappel    : parmi les vraies erreurs, combien sont détectées ?
        F1        : moyenne harmonique de précision et rappel

    Args:
        predicted_text       : réponse du modèle (liste des erreurs détectées)
        known_hallucinations : liste des vraies hallucinations

    Returns:
        Dictionnaire avec précision, rappel, f1
    """

    predicted_lower = predicted_text.lower()

    # Pour chaque hallucination connue, on extrait un mot-clé distinctif
    # et on vérifie s'il apparaît dans la réponse du modèle
    detected = []
    for hall in known_hallucinations:
        # On prend les mots significatifs (> 4 chars) de la description
        keywords = [w for w in hall.lower().split() if len(w) > 4]
        # Si au moins 2 mots-clés sont présents, on considère que
        # la hallucination a été détectée
        matches = sum(1 for kw in keywords if kw in predicted_lower)
        detected.append(matches >= 2)

    # Nombre de vraies hallucinations détectées (vrais positifs)
    true_positives = sum(detected)
    total_known    = len(known_hallucinations)

    # Rappel : combien de vraies hallucinations ont été détectées ?
    recall = true_positives / total_known if total_known > 0 else 0.0

    # Précision : approximée par le rappel car on ne parse pas
    # le nombre exact d'erreurs signalées par le modèle
    # (cela nécessiterait du parsing complexe de la réponse)
    precision = recall   # simplification raisonnable pour ce projet

    # F1 : moyenne harmonique
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall"   : round(recall, 4),
        "f1"       : round(f1, 4),
    }