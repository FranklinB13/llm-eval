"""
Module : models.py
Auteur : Franklin
Date   : 2026-04

Description :
    Gestion des appels API vers les 5 LLMs évalués.

    Tous les modèles sont accessibles via l'API Groq (gratuite).
    L'API Groq est compatible OpenAI : même format de requête/réponse.
    On appelle l'API directement avec httpx, sans SDK supplémentaire.

    Les 5 modèles choisis couvrent différentes familles et tailles :
        - Llama 3.1 8B  : petit et rapide (Meta)
        - Llama 3.3 70B : grand et puissant (Meta)
        - Llama 4 Scout : tout nouveau, multi-modal (Meta)
        - Qwen3 32B     : excellent en raisonnement (Alibaba)
        - GPT-OSS 120B  : plus grand modèle open-weight sur Groq

    Pourquoi ces 5 là ?
        Ils représentent les grandes familles de LLMs open source
        disponibles en 2026. Comparer leurs performances montre
        qu'on connaît l'écosystème, pas juste un seul modèle.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import time
import httpx
from dotenv import load_dotenv
from dataclasses import dataclass


# ==============================================================================
# CONFIGURATION
# ==============================================================================

load_dotenv()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
TIMEOUT      = 60
TEMPERATURE  = 0.1
MAX_TOKENS   = 512

# Les 5 modèles qu'on évalue
# Clé courte = ce qu'on utilise dans notre code
# Valeur = identifiant exact attendu par l'API Groq
MODELS = {
    "llama-3.1-8b"  : "llama-3.1-8b-instant",
    "llama-3.3-70b" : "llama-3.3-70b-versatile",
    "llama-4-scout" : "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen3-32b"     : "qwen/qwen3-32b",
    "gpt-oss-120b"  : "openai/gpt-oss-120b",
}


# ==============================================================================
# STRUCTURE DE DONNÉES
# ==============================================================================

@dataclass
class ModelResponse:
    """
    Représente la réponse d'un modèle à une requête.

    Champs :
        model_id      : identifiant court (ex: "llama-3.1-8b")
        model_name    : nom complet API (ex: "llama-3.1-8b-instant")
        text          : texte généré
        latency_s     : temps de réponse en secondes
        input_tokens  : tokens dans le prompt
        output_tokens : tokens dans la réponse
        success       : True si l'appel a réussi
        error         : message d'erreur si success=False
    """
    model_id      : str
    model_name    : str
    text          : str
    latency_s     : float
    input_tokens  : int
    output_tokens : int
    success       : bool
    error         : str = ""

    @property
    def total_tokens(self) -> int:
        """Tokens totaux consommés."""
        return self.input_tokens + self.output_tokens

    @property
    def tokens_per_second(self) -> float:
        """
        Vitesse de génération.
        Permet de comparer la rapidité des modèles indépendamment
        de leur taille — un 70B peut être plus rapide qu'un 8B
        grâce aux optimisations de Groq.
        """
        if self.latency_s > 0:
            return round(self.output_tokens / self.latency_s, 1)
        return 0.0


# ==============================================================================
# CLIENT API
# ==============================================================================

class LLMClient:
    """
    Client pour appeler les 5 modèles via l'API Groq.

    Pourquoi une classe ?
        La clé API et les headers sont initialisés une fois
        et réutilisés pour tous les appels. Plus efficace que
        de recréer une connexion à chaque appel.

    Utilisation :
        client = LLMClient()
        response = client.call("llama-3.1-8b", "Résume ce texte...")
        responses = client.call_all("Question pour tous les modèles")
    """

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY non trouvée. "
                "Vérifie que ton .env contient : GROQ_API_KEY=ta_clé"
            )
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type" : "application/json",
        }

    def call(
        self,
        model_id : str,
        prompt   : str,
        system   : str = "",
    ) -> ModelResponse:
        """
        Appelle un modèle et retourne sa réponse avec métriques.

        On mesure la latence autour de l'appel HTTP pour avoir
        le vrai temps de réponse perçu par l'utilisateur.

        Args:
            model_id : clé courte du modèle (doit être dans MODELS)
            prompt   : texte de la requête
            system   : instruction système optionnelle

        Returns:
            ModelResponse avec texte + latence + tokens
        """

        if model_id not in MODELS:
            return ModelResponse(
                model_id=model_id, model_name="", text="",
                latency_s=0.0, input_tokens=0, output_tokens=0,
                success=False,
                error=f"Modèle inconnu : {model_id}",
            )

        model_name = MODELS[model_id]
        messages   = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        request_body = {
            "model"      : model_name,
            "messages"   : messages,
            "temperature": TEMPERATURE,
            "max_tokens" : MAX_TOKENS,
        }

        start_time = time.time()

        try:
            response = httpx.post(
                url     = GROQ_API_URL,
                json    = request_body,
                headers = self.headers,
                timeout = TIMEOUT,
            )
            response.raise_for_status()
            latency       = round(time.time() - start_time, 3)
            data          = response.json()
            text          = data["choices"][0]["message"]["content"]
            input_tokens  = data["usage"]["prompt_tokens"]
            output_tokens = data["usage"]["completion_tokens"]

            return ModelResponse(
                model_id=model_id, model_name=model_name, text=text,
                latency_s=latency, input_tokens=input_tokens,
                output_tokens=output_tokens, success=True,
            )

        except httpx.HTTPStatusError as e:
            latency = round(time.time() - start_time, 3)
            return ModelResponse(
                model_id=model_id, model_name=model_name, text="",
                latency_s=latency, input_tokens=0, output_tokens=0,
                success=False,
                error=f"HTTP {e.response.status_code}: {e.response.text[:200]}",
            )

        except httpx.TimeoutException:
            return ModelResponse(
                model_id=model_id, model_name=model_name, text="",
                latency_s=TIMEOUT, input_tokens=0, output_tokens=0,
                success=False, error=f"Timeout après {TIMEOUT}s",
            )

    def call_all(
        self,
        prompt : str,
        system : str = "",
        delay  : float = 3.0,
    ) -> list[ModelResponse]:
        """
        Appelle tous les modèles avec le même prompt.

        Pourquoi un délai entre les appels ?
            L'API Groq gratuite a un rate limit de tokens/minute.
            Sans délai, les appels successifs sur les grands modèles
            peuvent déclencher une erreur 429 (trop de requêtes).
            3 secondes suffit pour les prompts courts.

        Args:
            prompt : texte identique pour tous les modèles
            system : instruction système optionnelle
            delay  : secondes entre chaque appel

        Returns:
            Liste de ModelResponse dans l'ordre de MODELS
        """
        responses = []
        for i, model_id in enumerate(MODELS.keys()):
            print(f"  [{i+1}/{len(MODELS)}] {model_id}...")
            r = self.call(model_id, prompt, system)
            if r.success:
                print(f"  ✅ {r.latency_s}s | {r.tokens_per_second} tok/s")
            else:
                print(f"  ❌ {r.error[:80]}")
            responses.append(r)
            if i < len(MODELS) - 1:
                time.sleep(delay)
        return responses