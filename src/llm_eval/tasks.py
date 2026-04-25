"""
Module : tasks.py
Auteur : Franklin
Date   : 2026-04

Description :
    Définition des 3 tâches d'évaluation et du dataset associé.

    Tâche 1 — Résumé scientifique (Summarization) :
        On donne un extrait d'article arXiv et on demande un résumé.
        On compare avec le résumé de référence écrit par l'auteur.
        Métriques : ROUGE-1, ROUGE-2, ROUGE-L, BERTScore

    Tâche 2 — Question-Answering avec contexte (QA) :
        On donne un contexte documentaire et une question.
        Le modèle doit répondre en s'appuyant UNIQUEMENT sur le contexte.
        Métriques : BERTScore, hallucination rate

    Tâche 3 — Détection d'hallucinations (Hallucination) :
        On donne un contexte et une réponse qui contient des affirmations
        incorrectes. Le modèle doit identifier les hallucinations.
        Métriques : précision, rappel

    Pourquoi ces 3 tâches ?
        Elles couvrent les cas d'usage réels les plus fréquents en prod :
        - Summarization : assistant de veille, synthèse de documents
        - QA : chatbot d'entreprise, RAG
        - Hallucination : évaluation de fiabilité avant déploiement
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from dataclasses import dataclass, field


# ==============================================================================
# STRUCTURES DE DONNÉES
# ==============================================================================

@dataclass
class SummarizationSample:
    """
    Un exemple pour la tâche de résumé scientifique.

    Champs :
        id        : identifiant unique de l'exemple
        article   : texte de l'article (extrait arXiv)
        reference : résumé de référence écrit par l'auteur
        topic     : sujet de l'article (pour les statistiques par domaine)
    """
    id        : str
    article   : str
    reference : str
    topic     : str


@dataclass
class QASample:
    """
    Un exemple pour la tâche de question-answering.

    Champs :
        id        : identifiant unique
        context   : texte de contexte fourni au modèle
        question  : question posée
        reference : réponse de référence attendue
        topic     : sujet de la question
    """
    id        : str
    context   : str
    question  : str
    reference : str
    topic     : str


@dataclass
class HallucinationSample:
    """
    Un exemple pour la tâche de détection d'hallucinations.

    Champs :
        id               : identifiant unique
        context          : texte de contexte de référence
        hallucinated_text: texte qui contient des hallucinations
        hallucinations   : liste des affirmations incorrectes
    """
    id                : str
    context           : str
    hallucinated_text : str
    hallucinations    : list[str]


# ==============================================================================
# PROMPTS
# ==============================================================================

# Prompt pour la tâche de résumé
# On demande un résumé court et factuel, pas créatif
# "3-4 sentences" : assez pour capturer l'essentiel, pas trop pour
# éviter que le modèle paraphrase tout l'article
SUMMARIZATION_PROMPT = """You are a scientific assistant. Read the following research article excerpt and write a concise summary of 3-4 sentences.

Focus on: the main contribution, the method used, and the key results.
Do NOT add information that is not in the text.

Article:
{article}

Summary:"""


# Prompt pour la tâche de QA
# On insiste sur "ONLY based on the context" pour mesurer
# si le modèle respecte les limites du contexte donné
# C'est ce qui permet de mesurer le hallucination rate
QA_PROMPT = """You are a precise assistant. Answer the question based ONLY on the provided context.
If the answer is not in the context, say "I cannot answer based on the provided context."

Context:
{context}

Question: {question}

Answer:"""


# Prompt pour la détection d'hallucinations
# On demande au modèle d'identifier les affirmations incorrectes
# sous forme de liste numérotée pour faciliter le parsing
HALLUCINATION_PROMPT = """You are a fact-checker. Compare the following text with the reference context and identify all factual errors or hallucinations.

Reference context:
{context}

Text to check:
{hallucinated_text}

List each factual error found (one per line, numbered):
1."""


# ==============================================================================
# DATASET : TÂCHE 1 — RÉSUMÉ SCIENTIFIQUE
# ==============================================================================

# 10 exemples tirés de papers arXiv sur nos domaines d'intérêt
# Les articles sont des extraits réels, les résumés sont les abstracts originaux
# On couvre 5 domaines : RAG, LLM, hallucinations, fine-tuning, transformers

SUMMARIZATION_SAMPLES = [
    SummarizationSample(
        id        = "sum_001",
        topic     = "RAG",
        article   = """Retrieval-Augmented Generation (RAG) has emerged as a powerful approach for enhancing large language models with external knowledge. The core idea is to retrieve relevant documents from a knowledge base and incorporate them into the generation process. This approach addresses a fundamental limitation of parametric models: their knowledge is fixed at training time and cannot be updated without retraining. RAG systems typically consist of a retriever that finds relevant passages and a generator that produces the final output conditioned on both the query and the retrieved passages. Recent work has shown that hybrid retrieval, combining sparse BM25 and dense embeddings, significantly outperforms either approach alone. Additionally, reranking retrieved passages with a cross-encoder before generation further improves answer quality. Evaluation on knowledge-intensive tasks shows that RAG systems achieve state-of-the-art performance while remaining interpretable through source attribution.""",
        reference = "RAG enhances LLMs by retrieving external documents at inference time, addressing the static knowledge limitation of parametric models. Systems combining sparse and dense retrieval with cross-encoder reranking achieve state-of-the-art performance on knowledge-intensive tasks while providing source attribution for interpretability."
    ),
    SummarizationSample(
        id        = "sum_002",
        topic     = "Hallucinations",
        article   = """Hallucinations in large language models refer to the generation of plausible-sounding but factually incorrect or unsupported information. They can be categorized as intrinsic hallucinations, where the model contradicts the source context, or extrinsic hallucinations, where it adds unverifiable information. Several mitigation strategies have been proposed. Retrieval-augmented generation grounds model responses in retrieved documents, reducing reliance on parametric memory. Chain-of-thought prompting encourages step-by-step reasoning, reducing logical errors. Post-hoc verification methods use a separate model to check factual consistency. Fine-tuning on high-quality factual datasets also reduces hallucination rates. However, no single approach eliminates hallucinations entirely, and the problem remains an active research area. Evaluation typically uses metrics like FactScore, HaluEval, or human annotation.""",
        reference = "LLM hallucinations are categorized as intrinsic or extrinsic and are addressed through RAG, chain-of-thought prompting, post-hoc verification, and fine-tuning. No single method eliminates hallucinations, and evaluation relies on metrics such as FactScore and HaluEval."
    ),
    SummarizationSample(
        id        = "sum_003",
        topic     = "Fine-tuning",
        article   = """Parameter-efficient fine-tuning (PEFT) methods have become essential for adapting large language models to specific tasks without updating all model parameters. LoRA (Low-Rank Adaptation) is the most widely adopted PEFT method. It freezes the pretrained model weights and injects trainable low-rank decomposition matrices into the attention layers. Formally, for a weight matrix W, LoRA adds a low-rank update W + AB where A and B have rank r much smaller than the original matrix dimensions. This reduces the number of trainable parameters from billions to millions. QLoRA extends LoRA by quantizing the base model to 4-bit precision, further reducing memory requirements. Experiments show that LoRA achieves performance comparable to full fine-tuning while training only 0.1-1% of the parameters, making it accessible on consumer hardware.""",
        reference = "LoRA is the leading PEFT method that adds low-rank decomposition matrices to frozen pretrained weights, training only 0.1-1% of parameters while matching full fine-tuning performance. QLoRA extends this with 4-bit quantization, enabling fine-tuning on consumer hardware."
    ),
    SummarizationSample(
        id        = "sum_004",
        topic     = "Transformers",
        article   = """The transformer architecture introduced by Vaswani et al. revolutionized natural language processing through the self-attention mechanism. Self-attention allows each token in a sequence to attend to all other tokens, capturing long-range dependencies without the recurrence of RNNs. The scaled dot-product attention computes queries, keys, and values from the input, then weights the values by the softmax of the dot products between queries and keys, scaled by the square root of the key dimension. Multi-head attention applies this mechanism multiple times in parallel with different learned projections, allowing the model to capture different types of relationships simultaneously. The feed-forward layers process each position independently with a two-layer MLP. Positional encodings are added to preserve sequence order information since attention is permutation-invariant.""",
        reference = "Transformers use self-attention to capture long-range dependencies by computing weighted combinations of values based on query-key dot products. Multi-head attention enables learning diverse relationship types simultaneously, while positional encodings preserve sequence order in the permutation-invariant architecture."
    ),
    SummarizationSample(
        id        = "sum_005",
        topic     = "RLHF",
        article   = """Reinforcement Learning from Human Feedback (RLHF) has become the dominant approach for aligning large language models with human preferences. The process consists of three stages. First, the model is supervised fine-tuned on high-quality demonstrations. Second, a reward model is trained on human preference comparisons between pairs of model outputs. Third, the language model is optimized using PPO, a proximal policy optimization algorithm, with the reward model providing the training signal. A KL divergence penalty prevents the policy from deviating too far from the supervised fine-tuned model. RLHF has been shown to significantly improve helpfulness, harmlessness, and honesty compared to supervised fine-tuning alone. Recent alternatives include DPO (Direct Preference Optimization), which eliminates the need for a separate reward model by directly optimizing on preference data.""",
        reference = "RLHF aligns LLMs with human preferences through three stages: supervised fine-tuning, reward model training on preference comparisons, and PPO optimization with KL divergence regularization. DPO offers a simpler alternative by directly optimizing on preference data without a separate reward model."
    ),
    SummarizationSample(
        id        = "sum_006",
        topic     = "Embeddings",
        article   = """Sentence embeddings encode semantic meaning into dense vector representations, enabling efficient semantic search and retrieval. Modern embedding models like BGE (BAAI General Embedding) are trained using contrastive learning: positive pairs of semantically similar sentences are pulled together in the embedding space while negative pairs are pushed apart. The quality of embeddings is measured on the MTEB (Massive Text Embedding Benchmark), which evaluates across multiple tasks including retrieval, clustering, and classification. Bi-encoder models generate embeddings independently for queries and documents, enabling fast approximate nearest-neighbor search. Cross-encoders jointly process query-document pairs for more accurate relevance scoring but are too slow for retrieval over large corpora, making them ideal for reranking a small set of candidates.""",
        reference = "Sentence embedding models like BGE use contrastive learning to encode semantic similarity into dense vectors, evaluated on MTEB. Bi-encoders enable fast retrieval while cross-encoders provide more accurate reranking, combining both yields state-of-the-art RAG systems."
    ),
    SummarizationSample(
        id        = "sum_007",
        topic     = "RAG",
        article   = """Chunking strategy significantly impacts the performance of RAG systems. Naive fixed-size chunking splits documents at fixed character or token counts regardless of semantic boundaries, often breaking sentences or ideas mid-way. Semantic chunking respects paragraph boundaries and sentence structure, preserving coherence within each chunk. Sliding window chunking adds overlap between consecutive chunks to prevent information loss at boundaries, with typical overlap ratios of 10-20%. The chunk size presents a fundamental trade-off: smaller chunks yield more precise embeddings and better retrieval accuracy, while larger chunks provide more context for generation. Evaluation shows that chunk sizes between 200-500 words with 10-20% overlap typically perform best across retrieval benchmarks. Hierarchical chunking, which creates both sentence-level and paragraph-level representations, has shown promising results.""",
        reference = "Chunking strategy critically affects RAG performance, with semantic chunking and sliding windows outperforming naive fixed-size splitting. Optimal chunk sizes of 200-500 words with 10-20% overlap balance retrieval precision with generation context, while hierarchical chunking shows additional promise."
    ),
    SummarizationSample(
        id        = "sum_008",
        topic     = "LLM Evaluation",
        article   = """Evaluating large language models requires multi-dimensional benchmarks that capture different aspects of model capability. MMLU tests knowledge across 57 academic subjects. HumanEval measures code generation ability. TruthfulQA evaluates factual accuracy and resistance to generating falsehoods. MT-Bench and AlpacaEval assess instruction-following through GPT-4-based evaluation. However, static benchmarks suffer from data contamination as models are increasingly trained on internet data that overlaps with test sets. Dynamic evaluation frameworks that generate new questions on-the-fly have emerged as a solution. Human preference evaluation through platforms like Chatbot Arena provides complementary signal by collecting real-world user preferences, though it is expensive and slow to scale.""",
        reference = "LLM evaluation uses diverse benchmarks including MMLU for knowledge, HumanEval for coding, and TruthfulQA for factuality, but static benchmarks face data contamination risks. Dynamic evaluation and human preference platforms like Chatbot Arena complement automated metrics with real-world signal."
    ),
    SummarizationSample(
        id        = "sum_009",
        topic     = "Chain-of-thought",
        article   = """Chain-of-thought (CoT) prompting elicits complex reasoning from large language models by encouraging them to produce intermediate reasoning steps before the final answer. This contrasts with standard prompting where models directly output answers. Few-shot CoT provides examples of reasoning chains in the prompt, while zero-shot CoT simply adds "Let's think step by step" to the query. CoT significantly improves performance on arithmetic, commonsense, and symbolic reasoning tasks, with gains increasing with model scale. Self-consistency extends CoT by sampling multiple reasoning paths and selecting the most common final answer, improving robustness. Tree of Thoughts generalizes CoT by allowing exploration of multiple reasoning branches and backtracking, enabling more systematic problem solving for complex tasks.""",
        reference = "Chain-of-thought prompting improves LLM reasoning by generating intermediate steps, with few-shot and zero-shot variants both effective. Self-consistency and Tree of Thoughts extend CoT through multi-path sampling and systematic branch exploration, yielding further gains on complex reasoning tasks."
    ),
    SummarizationSample(
        id        = "sum_010",
        topic     = "LLM Efficiency",
        article   = """Efficient inference of large language models has become critical as model sizes grow. Quantization reduces model precision from 32-bit or 16-bit floats to 8-bit integers or lower, reducing memory footprint and accelerating computation with minimal quality loss. Knowledge distillation trains a smaller student model to mimic a larger teacher model, achieving better performance than training the small model from scratch. Speculative decoding uses a small draft model to generate candidate tokens that the large model verifies in parallel, significantly accelerating autoregressive generation. Flash Attention optimizes the attention computation to reduce memory bandwidth requirements, enabling longer context windows. These techniques are often combined: a quantized model with Flash Attention and speculative decoding can achieve 5-10x speedup over naive inference.""",
        reference = "LLM inference efficiency is improved through quantization, knowledge distillation, speculative decoding, and Flash Attention, with combined techniques achieving 5-10x speedups. These methods reduce memory requirements and accelerate generation while preserving model quality."
    ),
]


# ==============================================================================
# DATASET : TÂCHE 2 — QUESTION-ANSWERING
# ==============================================================================

QA_SAMPLES = [
    QASample(
        id        = "qa_001",
        topic     = "RAG",
        context   = """RAG systems retrieve relevant documents from a knowledge base to augment language model generation. The retrieval component typically uses either sparse methods like BM25, which rely on keyword matching, or dense methods using neural embeddings that capture semantic similarity. Hybrid retrieval combines both approaches. A reranker then scores the top retrieved candidates to select the most relevant passages for generation. The generator receives both the query and retrieved passages as context.""",
        question  = "What is the role of the reranker in a RAG system?",
        reference = "The reranker scores the top retrieved candidates to select the most relevant passages before generation."
    ),
    QASample(
        id        = "qa_002",
        topic     = "LoRA",
        context   = """LoRA adds low-rank matrices to the attention layers of a frozen pretrained model. For a weight matrix W of dimensions d×k, LoRA introduces two matrices A (d×r) and B (r×k) where r is the rank, typically set between 4 and 64. During training, only A and B are updated while W remains frozen. The effective weight becomes W + AB. The rank r controls the number of trainable parameters: lower rank means fewer parameters but less expressive power.""",
        question  = "How does LoRA reduce the number of trainable parameters?",
        reference = "LoRA freezes the original weight matrix W and only trains two small low-rank matrices A and B, where the rank r is much smaller than the original dimensions."
    ),
    QASample(
        id        = "qa_003",
        topic     = "RLHF",
        context   = """RLHF consists of three stages. In stage 1, supervised fine-tuning trains the model on high-quality human demonstrations. In stage 2, human annotators compare pairs of model outputs and express preferences. These comparisons train a reward model that predicts human preferences. In stage 3, the policy model is optimized with PPO using the reward model as the reward signal, subject to a KL divergence constraint against the SFT model.""",
        question  = "What prevents the model from deviating too much from the original in RLHF stage 3?",
        reference = "A KL divergence constraint against the supervised fine-tuned model prevents the policy from deviating too far during PPO optimization."
    ),
    QASample(
        id        = "qa_004",
        topic     = "Hallucinations",
        context   = """Intrinsic hallucinations occur when the model generates content that contradicts the provided source context. Extrinsic hallucinations occur when the model adds information that cannot be verified from the source, regardless of its factual accuracy. Both types are problematic in production systems. FactScore is a metric that decomposes generated text into atomic facts and verifies each against a knowledge source.""",
        question  = "What is the difference between intrinsic and extrinsic hallucinations?",
        reference = "Intrinsic hallucinations contradict the source context, while extrinsic hallucinations add unverifiable information not present in the source."
    ),
    QASample(
        id        = "qa_005",
        topic     = "Transformers",
        context   = """In the transformer architecture, self-attention computes three projections from the input: queries Q, keys K, and values V. The attention weights are computed as softmax(QK^T / sqrt(d_k)) where d_k is the key dimension. The output is the weighted sum of values. The square root scaling prevents the dot products from becoming too large in high dimensions, which would push the softmax into regions with very small gradients.""",
        question  = "Why is the scaling factor sqrt(d_k) used in attention?",
        reference = "The sqrt(d_k) scaling prevents dot products from becoming too large in high dimensions, which would push the softmax into regions with very small gradients."
    ),
    QASample(
        id        = "qa_006",
        topic     = "Embeddings",
        context   = """The MTEB benchmark evaluates embedding models across 56 tasks in 8 categories: retrieval, clustering, classification, reranking, semantic textual similarity, summarization, pair classification, and bitext mining. Models are ranked by average score across all tasks. BGE-large achieves top performance on many retrieval tasks. Embedding dimension varies from 384 for small models to 4096 for large ones, with larger dimensions generally providing better quality at higher computational cost.""",
        question  = "How many task categories does MTEB evaluate?",
        reference = "MTEB evaluates across 8 task categories: retrieval, clustering, classification, reranking, semantic textual similarity, summarization, pair classification, and bitext mining."
    ),
    QASample(
        id        = "qa_007",
        topic     = "Chain-of-thought",
        context   = """Zero-shot chain-of-thought prompting adds the phrase 'Let's think step by step' to the query without providing any examples. This simple modification significantly improves performance on reasoning tasks. Few-shot chain-of-thought provides several examples of (question, reasoning chain, answer) triples in the prompt. Self-consistency samples k different reasoning chains and takes the majority vote on the final answer, improving robustness compared to greedy decoding.""",
        question  = "What is self-consistency in chain-of-thought prompting?",
        reference = "Self-consistency samples multiple reasoning chains and selects the most common final answer through majority vote, improving robustness over greedy decoding."
    ),
    QASample(
        id        = "qa_008",
        topic     = "LLM Efficiency",
        context   = """Speculative decoding uses a small draft model to generate candidate sequences of k tokens. The large target model then verifies all k tokens in parallel in a single forward pass. If the draft model's token matches the target model's distribution, it is accepted. Otherwise, the sequence is truncated at the first mismatch and a corrected token is sampled. This achieves lossless acceleration: the output distribution is identical to the target model while requiring fewer serial forward passes.""",
        question  = "How does speculative decoding maintain the same output quality as the original model?",
        reference = "Speculative decoding achieves lossless acceleration by truncating and correcting sequences at mismatches, ensuring the output distribution is identical to the target model."
    ),
]


# ==============================================================================
# DATASET : TÂCHE 3 — DÉTECTION D'HALLUCINATIONS
# ==============================================================================

HALLUCINATION_SAMPLES = [
    HallucinationSample(
        id      = "hall_001",
        context = "LoRA adds low-rank matrices A and B to frozen pretrained weights. Only A and B are trained. The rank r is typically between 4 and 64. Training only 0.1-1% of parameters achieves performance comparable to full fine-tuning.",
        hallucinated_text = "LoRA was introduced by Google in 2021 and updates all model parameters. It uses rank r typically set to 128. LoRA achieves 50% of full fine-tuning performance by training 10% of parameters.",
        hallucinations = [
            "LoRA was introduced by Google (it was introduced by Microsoft researchers)",
            "LoRA updates all model parameters (it freezes original weights)",
            "Rank r is typically set to 128 (typical range is 4-64)",
            "LoRA achieves 50% of full fine-tuning (it achieves comparable performance)",
            "Training 10% of parameters (LoRA trains 0.1-1%)"
        ]
    ),
    HallucinationSample(
        id      = "hall_002",
        context = "RLHF has three stages: supervised fine-tuning, reward model training from human preferences, and PPO optimization with KL divergence constraint.",
        hallucinated_text = "RLHF consists of two stages. First, the reward model is trained. Second, the model is optimized with REINFORCE algorithm. No KL constraint is used.",
        hallucinations = [
            "RLHF consists of two stages (it has three stages)",
            "The reward model is trained first (SFT comes first)",
            "REINFORCE algorithm is used (PPO is used)",
            "No KL constraint is used (KL divergence constraint is a core component)"
        ]
    ),
    HallucinationSample(
        id      = "hall_003",
        context = "BM25 is a sparse retrieval method based on keyword matching. Dense retrieval uses neural embeddings. Hybrid retrieval combines both. Cross-encoders are used for reranking, not initial retrieval.",
        hallucinated_text = "BM25 uses neural networks for retrieval. Dense retrieval relies on exact keyword matching. Cross-encoders are used for initial retrieval over millions of documents.",
        hallucinations = [
            "BM25 uses neural networks (it uses keyword statistics)",
            "Dense retrieval uses exact keyword matching (it uses neural embeddings)",
            "Cross-encoders are used for initial retrieval (they are used for reranking only)"
        ]
    ),
]


# ==============================================================================
# FONCTIONS UTILITAIRES
# ==============================================================================

def get_summarization_prompt(sample: SummarizationSample) -> str:
    """
    Génère le prompt pour un exemple de résumé.

    On utilise le template SUMMARIZATION_PROMPT défini ci-dessus
    et on y insère le texte de l'article.

    Args:
        sample : exemple de résumé avec l'article et la référence

    Returns:
        Prompt formaté prêt à être envoyé au modèle
    """
    return SUMMARIZATION_PROMPT.format(article=sample.article)


def get_qa_prompt(sample: QASample) -> str:
    """
    Génère le prompt pour un exemple de QA.

    Args:
        sample : exemple QA avec contexte et question

    Returns:
        Prompt formaté
    """
    return QA_PROMPT.format(
        context  = sample.context,
        question = sample.question,
    )


def get_hallucination_prompt(sample: HallucinationSample) -> str:
    """
    Génère le prompt pour un exemple de détection d'hallucinations.

    Args:
        sample : exemple avec contexte et texte à vérifier

    Returns:
        Prompt formaté
    """
    return HALLUCINATION_PROMPT.format(
        context          = sample.context,
        hallucinated_text= sample.hallucinated_text,
    )