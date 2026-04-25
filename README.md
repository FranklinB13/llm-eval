# LLM Evaluation Framework

Framework d'évaluation automatique comparant 5 LLMs open source sur 3 tâches NLP, avec métriques chiffrées et dashboard interactif.

Construit from scratch en Python — pas de framework d'évaluation tiers.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Le problème que ça résout

Avec des dizaines de LLMs disponibles en 2026, choisir le bon modèle pour une tâche précise est difficile. Les benchmarks publics (MMLU, HumanEval) mesurent des capacités générales mais pas les performances sur des cas d'usage spécifiques comme le résumé scientifique ou la détection d'hallucinations.

Ce framework permet d'évaluer n'importe quel ensemble de LLMs sur ses propres tâches et données, avec des métriques objectives et reproductibles.

---

## Résultats

Évaluation de 5 modèles sur 3 tâches NLP :

| Modèle | Résumé ROUGE-1 | Résumé BERTScore | QA BERTScore | QA Hallucination ↓ | Détection F1 |
|--------|---------------|-----------------|--------------|-------------------|--------------|
| Llama 3.1 8B | 0.413 | 0.847 | 0.878 | 0.00 | 0.933 |
| Llama 3.3 70B | 0.445 | 0.858 | 0.897 | 0.00 | 0.933 |
| Llama 4 Scout | 0.437 | 0.852 | **0.899** | **0.00** | **0.933** |
| Qwen3 32B | **0.490** | **0.863** | 0.838 | 0.12 | 0.767 |
| GPT-OSS 120B | 0.416 | 0.836 | 0.824 | 0.00 | 0.850 |

**Observations clés :**
- Llama 4 Scout surpasse GPT-OSS 120B sur le QA malgré une taille 7x inférieure
- Qwen3 32B domine sur la summarisation mais est le seul à halluciner sur le QA
- Llama 3.1 8B (le plus petit) égale les grands modèles sur la détection d'hallucinations

---

## Architecture

```
[5 LLMs via Groq API]
      ↓
  Tâche 1 : Résumé scientifique
      → 10 articles arXiv
      → Métriques : ROUGE-1, ROUGE-2, ROUGE-L, BERTScore F1

  Tâche 2 : Question-Answering avec contexte
      → 8 paires (contexte, question, référence)
      → Métriques : BERTScore F1, Hallucination Rate (LLM-as-a-judge)

  Tâche 3 : Détection d'hallucinations
      → 3 textes avec erreurs factuelles connues
      → Métriques : Précision, Rappel, F1
      ↓
  Dashboard Gradio avec visualisations comparatives
```

---

## Métriques implémentées

**ROUGE** — chevauchement de n-grammes entre réponse générée et référence. Standard pour la summarisation, rapide à calculer.

**BERTScore** — similarité sémantique via embeddings DistilBERT. Capture les synonymes et paraphrases que ROUGE rate. Deux textes qui disent la même chose avec des mots différents auront un bon BERTScore mais un ROUGE faible.

**Hallucination Rate (LLM-as-a-judge)** — un LLM juge (GPT-OSS 120B) évalue si la réponse contient des affirmations non supportées par le contexte. Score entre 0 (aucune hallucination) et 1 (tout inventé). Approche très utilisée en production car elle corrèle bien avec le jugement humain.

**Hallucination Detection F1** — précision et rappel de la capacité du modèle à identifier des erreurs factuelles dans un texte fourni.

---

## Modèles évalués

| Modèle | Paramètres | Organisation |
|--------|-----------|--------------|
| llama-3.1-8b-instant | 8B | Meta |
| llama-3.3-70b-versatile | 70B | Meta |
| llama-4-scout-17b | 17B | Meta |
| qwen3-32b | 32B | Alibaba |
| gpt-oss-120b | 120B | OpenAI |

Tous accessibles via l'API Groq (gratuite, ultra-rapide grâce aux puces LPU).

---

## Installation

**Prérequis :** Python 3.11+, Git

```bash
# 1. Cloner le repo
git clone https://github.com/FranklinB13/llm-eval.git
cd llm-eval

# 2. Installer les dépendances
pip install uv
uv sync

# 3. Configurer la clé API
# Créer un compte gratuit sur console.groq.com
echo "GROQ_API_KEY=ta_clé_ici" > .env

# 4. Lancer l'évaluation complète
uv run python scripts/run_eval.py
# (~30 minutes pour 5 modèles × 3 tâches)

# 5. Voir le dashboard
uv run python scripts/app.py
# Ouvrir http://127.0.0.1:7860
```

---

## Structure du projet

```
llm-eval/
├── src/
│   └── llm_eval/
│       ├── models.py       # client API pour les 5 LLMs (Groq)
│       ├── tasks.py        # 3 tâches + dataset (21 exemples annotés)
│       ├── metrics.py      # ROUGE, BERTScore, hallucination rate
│       ├── evaluator.py    # pipeline d'évaluation orchestré
│       └── visualizer.py   # graphiques matplotlib
├── scripts/
│   ├── run_eval.py         # lance l'évaluation complète
│   └── app.py              # dashboard Gradio interactif
├── results/                # résultats JSON par modèle et tâche
└── README.md
```

---

## Ce que j'ai appris

**La taille ne fait pas tout.** Llama 4 Scout (17B) surpasse GPT-OSS 120B sur le QA. Un modèle plus récent avec une meilleure architecture bat un modèle plus grand mais plus ancien.

**ROUGE et BERTScore sont complémentaires.** Qwen3 a le meilleur ROUGE (il choisit des mots proches de la référence) mais un BERTScore similaire aux autres (sens équivalent). Sans les deux métriques, on aurait une image incomplète.

**Le LLM-as-a-judge est puissant mais imparfait.** Sur le QA, 4 modèles obtiennent 0.00 d'hallucination. C'est probablement trop optimiste — nos contextes sont courts et clairs, ce qui aide les modèles à rester factuels. Sur des contextes plus ambigus, les scores seraient plus discriminants.

**L'idempotence est essentielle pour les pipelines longs.** L'évaluation complète prend 30 minutes. Sans sauvegarde intermédiaire et reprise automatique, la moindre interruption oblige à tout recommencer.

---

## Roadmap

- [ ] Ajout de nouveaux modèles (Claude, Gemini via API)
- [ ] Dataset d'évaluation plus large (50+ exemples par tâche)
- [ ] Export PDF du rapport d'évaluation
- [ ] Évaluation sur données utilisateur (upload de son propre dataset)

---

## Auteur

**Franklin** — ML Engineer · Mathématiques Appliquées, spécialité IA · CY Tech 2026
[GitHub](https://github.com/FranklinB13)