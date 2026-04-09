"""
PIPELINE DI VALUTAZIONE SISTEMA RAG
====================================

Valutazione quantitativa tramite ablation study sequenziale.
Varia una dimensione per volta rispetto alla BASELINE_CONFIG:
  Senza re-indicizzazione (eseguibili insieme, sessione 1):
  A) Strategia search   — multistage | hybrid | semantic | keyword
  B) Modello LLM        — gpt-4.1 | gpt-5
  C) Top-k              — 15 | 5

  Con re-indicizzazione (sessioni separate):
  D) Tipo chunking      — recursive_custom_baseline | fixed_size | recursive_standard
  E) Dimensione chunk   — 1024/overlap150 | 512/overlap100
  F) Modello embedding  — ada-002 | text-embedding-3-large

Uso da terminale:
  python evaluation_pipeline.py                        # ablation completo A-F
  python evaluation_pipeline.py -d A B C               # solo sessione 1 (no re-indicizzazione)
  python evaluation_pipeline.py -d D E                 # sessione 2 (fixed_size)
  python evaluation_pipeline.py --smoke-test            # smoke test multistage
  python evaluation_pipeline.py --smoke-test semantic   # smoke test strategia specifica

Uso da codice:
  from evaluation_pipeline import run_ablation_study, run_smoke_test
  run_ablation_study(dimensions=["A", "B", "C"])
  run_smoke_test(strategy=SearchStrategy.MULTISTAGE)

Note architetturali:
  - Le strategie semantic, keyword e multistage delegano l'intera pipeline a
    llm.run_pipeline_for_evaluation(), garantendo identità con il workflow reale.
  - La strategia hybrid è gestita inline come baseline di confronto: non fa
    parte del sistema reale, serve solo a verificare che la selezione adattiva
    di multistage non penalizzi eccessivamente il retrieval rispetto a hybrid fisso.
  - Per multistage, il retrieval è considerato un processo a due stadi
    (retriever + LLM selector via select_documents()). Le metriche di retrieval
    misurano l'output complessivo del processo (post-selezione).
  - Le metriche sono esclusivamente span-based (invarianti al chunking) più
    LLM-as-judge sulla rilevanza dei chunk selezionati.
"""

import os
import json
import time
import sys
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict, field
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np
from datetime import datetime
from openai import AzureOpenAI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import llm.search as search
import llm.llm as your_llm
from file_embedding.embedding import get_embedding


# ==================== CONFIGURAZIONE ====================

class SearchStrategy:
    SEMANTIC   = "semantic"    # Vector similarity — top_k doc diretti
    KEYWORD    = "keyword"     # BM25 — top_k doc diretti
    HYBRID     = "hybrid"      # Fusione pesata fissa (baseline di confronto)
    MULTISTAGE = "multistage"  # Pipeline multi-stage: decide_tools → retrieval → select_documents

class ExpectedBehavior:
    """Comportamenti attesi per le query negative."""
    NO_ANSWER     = "no_answer"     # Il sistema deve dichiarare assenza di informazioni
    CORRECTION    = "correction"    # Il sistema deve correggere un dato errato nella query
    CLARIFICATION = "clarification" # Il sistema deve disambiguare prima di rispondere
    POSITIVE      = None

# Path to the gold dataset used for evaluation.
# Defaults to a file called gold_dataset.json in the same directory as this script.
# Override by setting the GOLD_DATASET_PATH environment variable.
_DEFAULT_GOLD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold_dataset.json")
GOLD_DATASET_PATH = os.getenv("GOLD_DATASET_PATH", _DEFAULT_GOLD_PATH)

# ---- Configurazione baseline per l'ablation study ----

BASELINE_CONFIG = {
    "name":              "Baseline",
    "chunking":          "recursive_custom",
    "chunk_size":        1024,
    "chunk_overlap":     150,
    "embedding_model":   "text-embedding-ada-002",
    "search_strategy":   SearchStrategy.MULTISTAGE,
    "top_k":             15,    # per strategie NON multi-step
    "llm_model":         "gpt-4.1",
    "semantic_weight":   0.7,   # solo per strategia hybrid
}

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")

LLM_MODEL   = os.getenv("LLM_MODEL")
LLM_URL     = os.getenv("LLM_URL")
LLM_VERSION = os.getenv("LLM_VERSION")


# ==================== STRUTTURE DATI ====================

@dataclass
class EvaluationQuery:
    """Query del dataset GOLD con ground truth."""
    query_id:           str
    query_text:         str
    relevant_chunk_ids: List[str]
    relevant_doc_ids:   List[str]           = field(default_factory=list)
    relevant_spans:     List[str]           = field(default_factory=list)
    reference_answer:   Optional[str]       = None
    expected_behavior:  Optional[str]       = None
    negative_reason:    Optional[str]       = None

    @property
    def is_negative(self) -> bool:
        return self.expected_behavior is not None

    @property
    def has_spans(self) -> bool:
        return len(self.relevant_spans) > 0

@dataclass
class RetrievalResult:
    """
    Risultato del retrieval per una query.

    retrieved_docs: output grezzo del retriever, inclusi eventuali duplicati
                    quando entrambi i retriever sono attivi (multistage both=True).
                    Per semantic/keyword/hybrid coincide con selected_docs.

    selected_docs:  documenti effettivamente passati alla generation.
                    Per multistage: output di select_documents() — deduplicati,
                    filtrati sul template, riordinati per rilevanza crescente dall'LLM.
                    Per semantic/keyword/hybrid: uguale a retrieved_docs (top_k).

    n_docs_before_selection: numero di documenti in ingresso a select_documents()
                             (= len(retrieved_docs)). Diagnostico per multistage.
    n_docs_after_selection:  numero di documenti selezionati (= len(selected_docs)).
    selection_time:          tempo impiegato da select_documents() in secondi.
                             0.0 per strategie non-multistage.
    tool_decision:           decisione di decide_tools(). None per non-multistage.
    """
    query_id:                str
    retrieved_docs:          List[Dict]
    selected_docs:           List[Dict]
    retrieval_time:          float
    selection_time:          float
    n_docs_before_selection: int
    n_docs_after_selection:  int
    tool_decision:           Optional[Dict] = None

@dataclass
class GenerationResult:
    """Risultato della generation per una query."""
    query_id:         str
    generated_answer: str
    context_docs:     List[Dict]   # documenti effettivamente usati (= selected_docs)
    generation_time:  float
    selection_reason: str = ""     # reason da select_documents(), solo multistage

@dataclass
class NegativeEvaluationResult:
    """Risultato della valutazione per una query negativa."""
    query_id:          str
    expected_behavior: str
    behavior_score:    float
    behavior_label:    str
    llm_reasoning:     str
    generation_time:   float

@dataclass
class TestResult:
    """Risultato completo di un test con una specifica configurazione."""
    test_id:                str
    timestamp:              str
    configuration:          Dict
    retrieval_metrics:      Dict[str, float]
    generation_metrics:     Dict[str, float]
    negative_eval_metrics:  Dict[str, float]
    per_query_details:      List[Dict]
    total_evaluation_time:  float


# ==================== FUNZIONI DI UTILITÀ ====================

JUDGE_MODEL = os.getenv("JUDGE_MODEL", os.getenv("LLM_MODEL", "gpt-4o"))


def load_openai_client():
    """
    Build an OpenAI-compatible client for the judge model.

    Azure OpenAI  (set JUDGE_URL + JUDGE_VERSION in .env):
        Uses AzureOpenAI with the specified endpoint.

    Standard OpenAI  (leave JUDGE_URL unset):
        Uses the standard OpenAI client (api.openai.com or LLM_BASE_URL).
    """
    api_key   = os.getenv("OPENAI_API_KEY")
    judge_url = os.getenv("JUDGE_URL", os.getenv("LLM_URL", "")).strip()

    if judge_url:
        return AzureOpenAI(
            azure_endpoint=judge_url,
            api_key=api_key,
            api_version=os.getenv("JUDGE_VERSION", os.getenv("LLM_VERSION", "2024-02-01")),
        )
    else:
        from openai import OpenAI
        base_url = os.getenv("LLM_BASE_URL", None)
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def load_gold_dataset(filepath: str) -> List[EvaluationQuery]:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    queries = []
    for item in data:
        query = EvaluationQuery(
            query_id           = item['query_id'],
            query_text         = item['query_text'],
            relevant_chunk_ids = item.get('relevant_chunk_ids', []),
            relevant_doc_ids   = item.get('relevant_doc_ids', []),
            relevant_spans     = item.get('relevant_spans', []),
            reference_answer   = item.get('reference_answer'),
            expected_behavior  = item.get('expected_behavior'),
            negative_reason    = item.get('negative_reason')
        )
        queries.append(query)

    positive = sum(1 for q in queries if not q.is_negative)
    negative = sum(1 for q in queries if q.is_negative)
    print(f"  Dataset caricato: {len(queries)} query totali ({positive} positive, {negative} negative)")
    return queries


# ==================== METRICHE DI RETRIEVAL ====================

def _normalize_whitespace(text: str) -> str:
    """
    Normalizza whitespace in forma canonica per il confronto span-chunk.

    Collassa qualsiasi sequenza di caratteri whitespace (spazi, tab, newline,
    carriage return, spazi unificatori Unicode) in un singolo spazio.
    """
    import re
    return re.sub(r'\s+', ' ', text).strip()

def chunk_covers_span_partial(
    chunk_text: str,
    span:       str,
    min_coverage: float = 0.5
) -> float:
    """
    Ritorna la frazione dello span coperta dal chunk.
    Un chunk copre lo span se contiene almeno min_coverage
    della sua lunghezza come sottosequenza comune.
    """
    span_norm  = _normalize_whitespace(span)
    chunk_norm = _normalize_whitespace(chunk_text)

    if not span_norm:
        return 0.0

    # Sottostringa esatta — copertura totale
    if span_norm in chunk_norm:
        return 1.0

    # Stima della copertura tramite longest common substring
    matcher = SequenceMatcher(None, chunk_norm, span_norm)
    lcs_len = sum(block.size for block in matcher.get_matching_blocks())
    return lcs_len / len(span_norm)

def calculate_span_precision_at_k(
    retrieved_texts: List[str],
    relevant_spans:  List[str],
    k:               int,
    min_coverage:    float = 0.5
) -> float:
    """
    P@k span-based con copertura parziale.
    Un chunk è "rilevante" se copre >= min_coverage di almeno uno span.
    """
    if k == 0 or not retrieved_texts or not relevant_spans:
        return 0.0
    top_k = retrieved_texts[:k]
    relevant_count = sum(
        1 for chunk in top_k
        if max(
            (chunk_covers_span_partial(chunk, span, min_coverage)
             for span in relevant_spans),
            default=0.0
        ) >= min_coverage
    )
    return relevant_count / k

def calculate_span_recall_at_k(
    retrieved_texts: List[str],
    relevant_spans:  List[str],
    k:               int,
    min_coverage:    float = 0.5
) -> float:
    top_k = retrieved_texts[:k]
    if not relevant_spans:
        return 0.0
    covered = 0
    for span in relevant_spans:
        # Lo span è coperto se almeno un chunk copre >= min_coverage
        max_cov = max(
            (chunk_covers_span_partial(chunk, span, min_coverage)
             for chunk in top_k),
            default=0.0
        )
        if max_cov >= min_coverage:
            covered += 1
    return covered / len(relevant_spans)

def evaluate_chunk_relevance_with_llm(
    query:      str,
    chunk_text: str,
    client:     AzureOpenAI,
    model:      str = JUDGE_MODEL
) -> float:
    """
    Valuta la rilevanza di un chunk usando una rubrica discreta a 3 livelli
    (0/1/2) per ridurre la varianza rispetto a scale continue.
    """
    prompt = f"""Sei un valutatore esperto di sistemi di recupero documenti.

Query utente: "{query}"

Chunk di documento:
\"\"\"
{chunk_text}
\"\"\"

Valuta la rilevanza del chunk rispetto alla query seguendo questa rubrica:
- 0 = Il chunk non contiene informazioni utili per rispondere alla query
- 1 = Il chunk contiene informazioni parzialmente utili o solo indirettamente correlate
- 2 = Il chunk contiene informazioni direttamente utili per rispondere alla query

Ragiona brevemente (1-2 frasi), poi emetti il voto.

Formato risposta (rispetta esattamente):
Ragionamento: <testo>
Voto: <0, 1 o 2>"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=80
        )
        text = response.choices[0].message.content.strip()
        for line in reversed(text.splitlines()):
            if line.startswith("Voto:"):
                raw = line.split(":")[1].strip()
                if raw in ("0", "1", "2"):
                    return int(raw) / 2.0
        import re
        match = re.search(r'\b([012])\b', text)
        if match:
            return int(match.group(1)) / 2.0
        return 0.0
    except Exception as e:
        print(f"  Errore valutazione LLM chunk relevance: {e}")
        return 0.0

def calculate_average_chunk_relevance(
    query:            str,
    retrieved_chunks: List[str],
    client:           AzureOpenAI,
    model:            str = JUDGE_MODEL,
    top_k:            Optional[int] = None
) -> float:
    chunks_to_eval = retrieved_chunks[:top_k] if top_k else retrieved_chunks
    if not chunks_to_eval:
        return 0.0
    scores = [evaluate_chunk_relevance_with_llm(query, c, client, model) for c in chunks_to_eval]
    return float(np.mean(scores))

def evaluate_retrieval_for_query(
    query:            EvaluationQuery,
    retrieval_result: RetrievalResult,
    client:           AzureOpenAI,
    k_values:         List[int] = [5, 10, 15],
    llm_model:        str   = JUDGE_MODEL,
    span_threshold:   float = 0.7
) -> Dict[str, float]:
    """
    Calcola le metriche di retrieval per una singola query positiva.

    Tutte le metriche sono span-based (invarianti al chunking) e vengono
    calcolate su selected_docs, ovvero l'output finale del processo di
    retrieval (a due stadi per multistage, diretto per le altre strategie).

    Metriche calcolate per TUTTE le strategie:
      span_precision_set   P span-based sul set completo dei doc selezionati
                           = chunk selezionati che coprono ≥1 span / n_selezionati
      span_recall_set      R span-based sul set completo dei doc selezionati
                           = span coperti da ≥1 chunk selezionato / totale span
                           Metrica primaria per il confronto tra strategie.
      chunk_relevance_set  LLM-as-judge rilevanza media su tutti i doc selezionati
                           (rubrica discreta 0/1/2, normalizzata a [0,1])

    Metriche @k aggiuntive (solo per semantic/keyword/hybrid, dove l'output è
    una lista ordinata per score e il confronto a diversi k è significativo):
      span_precision_at_k  P@k span-based per k ∈ k_values
      span_recall_at_k     R@k span-based per k ∈ k_values

    Diagnostici (tutte le strategie):
      n_docs_before_selection  numero di doc in ingresso a select_documents()
                               (len(retrieved_docs), include eventuali duplicati)
      n_docs_after_selection   numero di doc selezionati (len(selected_docs))
    """
    metrics = {}

    if not query.has_spans:
        return metrics

    selected_texts = [d['content'] for d in retrieval_result.selected_docs]
    is_multistage  = retrieval_result.tool_decision is not None
    n_selected     = len(selected_texts)

    # ── Set-based span metrics (tutte le strategie) ───────────────────────
    metrics["span_precision_set"] = calculate_span_precision_at_k(
        selected_texts, query.relevant_spans, n_selected, span_threshold
    ) if selected_texts else 0.0

    metrics["span_recall_set"] = calculate_span_recall_at_k(
        selected_texts, query.relevant_spans, n_selected, span_threshold
    ) if selected_texts else 0.0

    # ── @k span metrics (solo non-multistage) ────────────────────────────
    if not is_multistage:
        for k in k_values:
            metrics[f"span_precision_at_{k}"] = calculate_span_precision_at_k(
                selected_texts, query.relevant_spans, k, span_threshold
            )
            metrics[f"span_recall_at_{k}"] = calculate_span_recall_at_k(
                selected_texts, query.relevant_spans, k, span_threshold
            )

    # ── LLM-as-judge rilevanza (set-based, tutte le strategie) ───────────
    metrics["chunk_relevance_set"] = calculate_average_chunk_relevance(
        query.query_text, selected_texts, client, llm_model, top_k=None
    )

    # ── Diagnostici ──────────────────────────────────────────────────────
    metrics["n_docs_before_selection"] = float(retrieval_result.n_docs_before_selection)
    metrics["n_docs_after_selection"]  = float(retrieval_result.n_docs_after_selection)

    return metrics


# ==================== METRICHE DI GENERATION (QUERY POSITIVE) ====================

def _strip_markdown(text: str) -> str:
    """
    Rimuove la formattazione markdown da un testo prima di passarlo al valutatore.

    Caso speciale: se l'intera risposta è wrappata in un fence ```markdown ... ```
    i delimitatori vengono rimossi preservando il contenuto interno.
    """
    import re

    text = re.sub(r'^```[a-zA-Z]*\n', '', text.strip())
    if text.endswith('```'):
        text = text[:-3].rstrip()

    text = re.sub(r'```[a-zA-Z]*\n?', '', text)
    text = re.sub(r'`[^`]+`', lambda m: m.group()[1:-1], text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\|[^\n]+\|', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def _extract_atomic_claims(
    generated_answer: str,
    client:           AzureOpenAI,
    model:            str
) -> List[str]:
    """
    Decompone la risposta generata in claim atomici verificabili.

    Il testo viene prima stripped del markdown. Se supera CHUNK_CHARS caratteri,
    viene spezzato in chunk sovrapposti ed elaborato separatamente — i claim
    vengono poi deduplicati per similarità stringa.
    """
    CHUNK_CHARS   = 2500
    OVERLAP_CHARS = 200

    plain_text = _strip_markdown(generated_answer)

    if len(plain_text) <= CHUNK_CHARS:
        return _extract_claims_from_chunk(plain_text, client, model)

    all_claims: List[str] = []
    start = 0
    while start < len(plain_text):
        chunk        = plain_text[start:start + CHUNK_CHARS]
        chunk_claims = _extract_claims_from_chunk(chunk, client, model)
        all_claims.extend(chunk_claims)
        start += CHUNK_CHARS - OVERLAP_CHARS

    seen: List[str] = []
    for claim in all_claims:
        normalized = claim.lower().strip().rstrip('.')
        if not any(normalized == s.lower().strip().rstrip('.') for s in seen):
            seen.append(claim)
    return seen

def _extract_claims_from_chunk(
    text:   str,
    client: AzureOpenAI,
    model:  str
) -> List[str]:
    """Estrae claim atomici da un singolo blocco di testo plain."""
    system_prompt = (
        "Sei un assistente specializzato nell'analisi di testi tecnici in italiano. "
        "Il tuo compito è identificare le affermazioni fattuali presenti in un testo, "
        "separando ogni fatto verificabile in un elemento distinto della lista."
    )
    user_prompt = f"""Analizza il testo seguente e produci una lista delle affermazioni fattuali in esso contenute.

Ogni elemento della lista deve essere una singola proposizione verificabile (un fatto specifico, un dato, un riferimento).
Le frasi introduttive, le congiunzioni e le valutazioni soggettive non sono affermazioni fattuali.

Testo da analizzare:
\"\"\"
{text}
\"\"\"

Formato di risposta: lista JSON di stringhe. Esempio:
["L'email del cliente è info@esempio.com", "Il referente è Mario Rossi"]

Se il testo non contiene affermazioni fattuali verificabili, scrivi: []"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=800
        )
        text_resp = response.choices[0].message.content.strip()
        text_resp = text_resp.replace("```json", "").replace("```", "").strip()

        if not text_resp:
            return []

        try:
            claims = json.loads(text_resp)
        except json.JSONDecodeError:
            last_quote = text_resp.rfind('"')
            if last_quote > 0:
                truncated = text_resp[:last_quote + 1].rstrip().rstrip(',')
                try:
                    claims = json.loads(truncated + ']')
                except json.JSONDecodeError:
                    return []
            else:
                return []

        if isinstance(claims, list):
            return [str(c) for c in claims]
        return []
    except Exception as e:
        print(f"  Errore estrazione claim atomici: {e}")
        return []

def _verify_single_claim(
    claim:        str,
    context_text: str,
    client:       AzureOpenAI,
    model:        str
) -> float:
    """
    Verifica se un singolo claim atomico è supportato dal contesto.
    Ritorna 1.0 se supportato, 0.5 se parzialmente, 0.0 se non supportato.
    """
    system_prompt = (
        "Sei un valutatore esperto di sistemi RAG. "
        "Il tuo compito è stabilire se una affermazione risulta confermata "
        "da un testo di riferimento."
    )
    user_prompt = f"""Testo di riferimento (estratto dai documenti recuperati):
\"\"\"
{context_text}
\"\"\"

Affermazione da valutare: "{claim}"

Quanto è supportata questa affermazione dal testo di riferimento?

Valori possibili:
- SUPPORTATO      il testo conferma esplicitamente l'affermazione
- PARZIALE        il testo è correlato ma non la conferma in modo diretto
- NON_SUPPORTATO  l'affermazione non è presente o è in contrasto con il testo

Risposta:"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=10
        )
        verdict = response.choices[0].message.content.strip().upper()
        if   "NON_SUPPORTATO" in verdict: return 0.0
        elif "PARZIALE"        in verdict: return 0.5
        elif "SUPPORTATO"      in verdict: return 1.0
        return 0.0
    except Exception as e:
        print(f"  Errore verifica claim: {e}")
        return 0.0

def _faithfulness_rubric_fallback(
    context_chunks:   List[str],
    generated_answer: str,
    client:           AzureOpenAI,
    model:            str
) -> float:
    """
    Valutazione di fallback con rubrica discreta 0/1/2 quando l'estrazione
    dei claim atomici non produce risultati (es. risposta troppo breve).
    """
    import re
    context_text = "\n\n".join(context_chunks)
    system_prompt = (
        "Sei un valutatore esperto di sistemi RAG. "
        "Il tuo compito è valutare quanto le informazioni contenute in una risposta "
        "siano supportate da un testo di riferimento."
    )
    user_prompt = f"""Testo di riferimento:
\"\"\"
{context_text}
\"\"\"

Risposta da valutare:
\"\"\"
{generated_answer}
\"\"\"

Quanto è fedele la risposta al testo di riferimento?

Rubrica:
- 0 = La risposta contiene informazioni assenti nel testo di riferimento
- 1 = La risposta è parzialmente fedele: alcune informazioni sono nel testo, altre no
- 2 = La risposta è completamente fedele: ogni informazione è supportata dal testo

Ragionamento: <1-2 frasi>
Voto: <0, 1 o 2>"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=80
        )
        text = response.choices[0].message.content.strip()
        for line in reversed(text.splitlines()):
            if line.startswith("Voto:"):
                raw = line.split(":")[1].strip()
                if raw in ("0", "1", "2"):
                    return int(raw) / 2.0
        import re as _re
        match = _re.search(r'\b([012])\b', text)
        if match:
            return int(match.group(1)) / 2.0
        return 0.0
    except Exception as e:
        print(f"  Errore fallback faithfulness: {e}")
        return 0.0

def evaluate_faithfulness_with_llm(
    context_chunks:   List[str],
    generated_answer: str,
    client:           AzureOpenAI,
    model:            str = JUDGE_MODEL
) -> Tuple[float, int, int]:
    """
    Valuta la fedeltà della risposta al contesto usando claim atomici.

    Approccio (ispirato a RAGAS, Es et al. 2023):
    1. Decompone la risposta in claim atomici verificabili individualmente.
    2. Per ogni claim verifica se è supportato dal contesto recuperato.
    3. Score = claims_supportati / claims_totali.

    Ritorna: (score, n_claims_totali, n_claims_supportati)
    """
    context_text = "\n\n".join(context_chunks)

    claims = _extract_atomic_claims(generated_answer, client, model)

    if not claims:
        fallback_score = _faithfulness_rubric_fallback(
            context_chunks, generated_answer, client, model
        )
        return fallback_score, 0, 0

    scores      = [_verify_single_claim(c, context_text, client, model) for c in claims]
    faith_score = float(np.mean(scores))
    n_supported = sum(1 for s in scores if s >= 1.0)

    print(f"    Faithfulness: {len(claims)} claim, "
          f"{n_supported} supportati, score={faith_score:.3f}")

    return faith_score, len(claims), n_supported

def _strip_answer_for_relevancy(text: str) -> str:
    """
    Prepara la risposta generata per la valutazione di answer_relevancy:
    rimuove la sezione 'Documenti di riferimento' e la formattazione markdown.
    """
    import re
    patterns = [
        r'\n---\n+#{1,4}\s*Documenti di riferimento.*',
        r'\n#{1,4}\s*Documenti di riferimento.*',
        r'\n---\n+\*\*Documenti[^*]*\*\*.*',
        r'\n\d+\. RI: \[.*',
    ]
    for p in patterns:
        m = re.search(p, text, re.DOTALL | re.IGNORECASE)
        if m:
            text = text[:m.start()]
            break
    return _strip_markdown(text).strip()

def evaluate_answer_relevancy_with_llm(
    query:            str,
    generated_answer: str,
    client:           AzureOpenAI,
    model:            str = JUDGE_MODEL
) -> float:
    """
    Valuta la pertinenza della risposta rispetto alla query con rubrica discreta.

    Rubrica:
    - 0 = La risposta non affronta la domanda (fuori tema o rifiuto)
    - 1 = La risposta affronta la domanda ma in modo incompleto
    - 2 = La risposta affronta direttamente tutti gli aspetti della domanda
    """
    import re
    clean_answer = _strip_answer_for_relevancy(generated_answer)

    system_prompt = (
        "Sei un valutatore esperto di sistemi RAG specializzato in documentazione "
        "tecnica ERP. Il tuo compito è valutare quanto una risposta copre gli "
        "aspetti richiesti dalla domanda dell'utente."
    )
    user_prompt = f"""Valuta la pertinenza della risposta rispetto alla query.

Rubrica:
- 0 = Il chunk è completamente fuori tema rispetto alla query (argomento diverso, nessuna sovrapposizione)
- 1 = Il chunk contiene informazioni correlate ma non direttamente utili per rispondere
- 2 = Il chunk contiene informazioni direttamente utili per rispondere alla query

Nota: la risposta può essere lunga e strutturata in sezioni — "
"questo non influenza il voto. Valuta esclusivamente se gli argomenti "
"trattati rispondono alla domanda, ignorando lunghezza e formato.

Esempi calibrati:

Query: "Qual è l'email di GMR Enlights e chi è il referente?"
Risposta: "L'email è info@gmrenlights.com."
Ragionamento: La risposta fornisce l'email ma non risponde alla domanda sul referente, che era esplicitamente richiesto.
Voto: 1

Query: "Come funziona il calcolo delle spese di trasporto per Logos SPA?"
Risposta: "Le spese di trasporto dipendono dal corriere scelto e dalle tariffe di mercato."
Ragionamento: La risposta parla di spese di trasporto in generale ma non affronta la logica implementata in SAM ERP2 né il caso specifico di Logos SPA. Non contiene nulla di utile per rispondere alla domanda.
Voto: 0

Query: "Come si configura la periodicità di liquidazione degli interessi per conto banca?"
Risposta: "La periodicità si configura per ogni conto banca con diverse opzioni: mensile (giorno e periodo di riferimento), trimestrale (quattro date), semestrale (due date con relativi periodi). Ogni configurazione determina quando il sistema calcola e registra gli interessi."
Ragionamento: La risposta copre direttamente la configurazione della periodicità con tutte le opzioni richieste.
Voto: 2

Query: "Qual è la logica di reperimento dello sconto sulle righe di ordine per MVM srl con classificatore 5?"
Risposta: "### Logica sconto per MVM srl\n\n**Classificatore 5 e sconto massimo**\n- Gli articoli sono raggruppati tramite classificatore generico 5 [1].\n- È presente un campo 'sconto massimo' nella tabella del classificatore.\n\n**Priorità listino personalizzato**\n- Il listino personalizzato del cliente ha sempre la precedenza sugli sconti standard [2].\n- Se i codici non hanno sconto sul listino, viene applicata la logica del classificatore 5 [1][2].\n\nDocumenti di riferimento:\n1. RI: [46443] ...\n2. RI: [46444] ..."
Ragionamento: La risposta è lunga e strutturata in sezioni, ma affronta direttamente tutti gli aspetti della query: il ruolo del classificatore 5, la logica dello sconto massimo e la priorità del listino personalizzato. La struttura markdown non riduce la pertinenza.
Voto: 2

---

Query: "{query}"

Risposta da valutare:
\"\"\"
{clean_answer}
\"\"\"

Ragionamento: <1-2 frasi che identificano quali aspetti della query sono coperti e quali mancano>
Voto: <0, 1 o 2>"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=120
        )
        text = response.choices[0].message.content.strip()
        for line in reversed(text.splitlines()):
            if line.startswith("Voto:"):
                raw = line.split(":")[1].strip()
                if raw in ("0", "1", "2"):
                    print(f"    chunk_relevance vote: {raw} | query: {query[:40]}")
                    return int(raw) / 2.0
        import re as _re
        match = _re.search(r'\b([012])\b', text)
        if match:
            return int(match.group(1)) / 2.0
        return 0.0
    except Exception as e:
        print(f"  Errore valutazione answer relevancy: {e}")
        return 0.0

def calculate_semantic_similarity_embeddings(
    reference_answer: str,
    generated_answer: str,
) -> float:
    """Similarità coseno tra embedding della reference e della risposta generata."""
    try:
        return cosine_similarity(
            get_embedding(reference_answer),
            get_embedding(generated_answer)
        )
    except Exception as e:
        print(f"  Errore calcolo semantic similarity: {e}")
        return 0.0

def evaluate_generation_for_query(
    query:             EvaluationQuery,
    generation_result: GenerationResult,
    client:            AzureOpenAI,
    llm_model:         str = JUDGE_MODEL,
    embedding_model:   str = "text-embedding-ada-002"
) -> Dict[str, float]:

    metrics = {}

    # I testi dei doc effettivamente usati per la generation
    context_texts = [d['content'] for d in generation_result.context_docs]

    faith_score, n_claims, n_supported = evaluate_faithfulness_with_llm(
        context_texts,
        generation_result.generated_answer,
        client, llm_model
    )
    metrics["faithfulness"]                  = faith_score
    metrics["faithfulness_claims_total"]     = float(n_claims)
    metrics["faithfulness_claims_supported"] = float(n_supported)

    metrics["answer_relevancy"] = evaluate_answer_relevancy_with_llm(
        query.query_text,
        generation_result.generated_answer,
        client, llm_model
    )

    if query.reference_answer:
        metrics["semantic_similarity"] = calculate_semantic_similarity_embeddings(
            query.reference_answer,
            generation_result.generated_answer
        )
    else:
        metrics["semantic_similarity"] = None

    return metrics


# ==================== VALUTAZIONE QUERY NEGATIVE ====================

def evaluate_no_answer_behavior(
    query:            EvaluationQuery,
    generated_answer: str,
    client:           AzureOpenAI,
    model:            str = JUDGE_MODEL
) -> Tuple[float, str, str]:
    """
    Verifica se il sistema ha correttamente dichiarato di non avere informazioni
    sufficienti per rispondere.

    Ritorna: (score 0.0/1.0, label, reasoning)
    """
    system_prompt = (
        "Sei un valutatore esperto di sistemi RAG specializzato in documentazione tecnica ERP. "
        "Il tuo compito è stabilire se un sistema RAG ha riconosciuto correttamente l'assenza "
        "di informazioni nel proprio database, invece di rispondere con contenuto non supportato."
    )
    user_prompt = f"""La seguente domanda riguarda un'informazione NON presente nel database documentale del sistema.
Query: "{query.query_text}"
Motivo per cui non dovrebbe rispondere: {query.negative_reason}

Risposta generata dal sistema:
\"\"\"\n{generated_answer}
\"\"\"\n
Rubrica di valutazione:
- 1 (CORRETTO) = Il sistema ha riconosciuto l'assenza di informazioni. Sono accettabili sia
  dichiarazioni dirette ("non ho informazioni", "non posso rispondere") sia indirette
  ("non viene menzionato nei documenti", "non risulta nei dati disponibili").
  Il sistema può aggiungere contesto generale purché segnali esplicitamente che
  l'informazione specifica richiesta non è disponibile o non trovata nei documenti.
- 0 (ERRATO) = Il sistema ha risposto come se disponesse dell'informazione richiesta,
  senza segnalare l'assenza. Sono ERRATE le risposte che forniscono dati specifici,
  procedure dettagliate o affermazioni fattuali sulla questione senza alcuna riserva.

Esempi calibrati:

Query: "È previsto un limite massimo mensile alle provvigioni per agente?"
Risposta: "Non viene menzionato alcun limite massimo mensile alle provvigioni nei documenti
disponibili, né come funzionalità standard né come personalizzazione."
Ragionamento: La risposta dichiara esplicitamente che l'informazione non è presente nei documenti.
Voto: 1

Query: "Come si configura il calcolo delle spese di trasporto per zona geografica?"
Risposta: "Il calcolo avviene tramite una personalizzazione specifica che considera il peso
e la destinazione della merce."
Ragionamento: La risposta fornisce dettagli come se l'informazione fosse disponibile.
Voto: 0

---

Ragionamento: <1-2 frasi>
Voto: <0 o 1>"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=150
        )
        text = response.choices[0].message.content.strip()
        reasoning = ""
        score = 0.0
        for line in text.splitlines():
            if line.startswith("Ragionamento:"):
                reasoning = line.split(":", 1)[1].strip()
            if line.startswith("Voto:"):
                score = float(line.split(":")[1].strip())
        label = "CORRETTO (ha dichiarato assenza info)" if score == 1.0 else "ERRATO (ha risposto senza basi)"
        return score, label, reasoning
    except Exception as e:
        print(f"  Errore valutazione no_answer: {e}")
        return 0.0, "ERRORE", str(e)

def evaluate_correction_behavior(
    query:            EvaluationQuery,
    generated_answer: str,
    client:           AzureOpenAI,
    model:            str = JUDGE_MODEL
) -> Tuple[float, str, str]:
    """
    Verifica se il sistema ha identificato e corretto il dato errato presente
    nella query.

    Ritorna: (score 0.0/1.0, label, reasoning)
    """
    prompt = f"""Sei un valutatore esperto di sistemi RAG.

La seguente domanda contiene un dato errato che il sistema dovrebbe correggere:
Query: "{query.query_text}"
Dato errato e correzione attesa: {query.negative_reason}

Risposta generata dal sistema:
\"\"\"
{generated_answer}
\"\"\"

Valuta se il sistema ha risposto correttamente seguendo questa rubrica:
- 0 = Il sistema ha confermato o ignorato il dato errato senza correggerlo
- 1 = Il sistema ha esplicitamente identificato il dato errato e fornito la correzione corretta

Ragiona brevemente (1-2 frasi), poi emetti il voto.

Formato risposta (rispetta esattamente):
Ragionamento: <testo>
Voto: <0 o 1>"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100
        )
        text = response.choices[0].message.content.strip()
        reasoning = ""
        score = 0.0
        for line in text.splitlines():
            if line.startswith("Ragionamento:"):
                reasoning = line.split(":", 1)[1].strip()
            if line.startswith("Voto:"):
                score = float(line.split(":")[1].strip())
        label = "CORRETTO (ha corretto il dato errato)" if score == 1.0 else "ERRATO (ha confermato o ignorato il dato)"
        return score, label, reasoning
    except Exception as e:
        print(f"  Errore valutazione correction: {e}")
        return 0.0, "ERRORE", str(e)

def evaluate_clarification_behavior(
    query:            EvaluationQuery,
    generated_answer: str,
    client:           AzureOpenAI,
    model:            str = JUDGE_MODEL
) -> Tuple[float, str, str]:
    """
    Verifica se il sistema ha gestito correttamente una query ambigua,
    chiedendo chiarimenti o coprendo esplicitamente le diverse interpretazioni.

    Ritorna: (score 0.0/1.0, label, reasoning)
    """
    prompt = f"""Sei un valutatore esperto di sistemi RAG.

La seguente domanda è ambigua e ammette interpretazioni multiple:
Query: "{query.query_text}"
Motivo dell'ambiguità: {query.negative_reason}

Risposta generata dal sistema:
\"\"\"
{generated_answer}
\"\"\"

Valuta se il sistema ha gestito l'ambiguità correttamente seguendo questa rubrica:
- 0 = Il sistema ha risposto a una sola interpretazione senza riconoscere l'ambiguità
- 1 = Il sistema ha riconosciuto l'ambiguità e ha chiesto chiarimenti OPPURE ha coperto esplicitamente le diverse interpretazioni possibili

Ragiona brevemente (1-2 frasi), poi emetti il voto.

Formato risposta (rispetta esattamente):
Ragionamento: <testo>
Voto: <0 o 1>"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100
        )
        text = response.choices[0].message.content.strip()
        reasoning = ""
        score = 0.0
        for line in text.splitlines():
            if line.startswith("Ragionamento:"):
                reasoning = line.split(":", 1)[1].strip()
            if line.startswith("Voto:"):
                score = float(line.split(":")[1].strip())
        label = "CORRETTO (ha gestito l'ambiguità)" if score == 1.0 else "ERRATO (ha ignorato l'ambiguità)"
        return score, label, reasoning
    except Exception as e:
        print(f"  Errore valutazione clarification: {e}")
        return 0.0, "ERRORE", str(e)

def evaluate_negative_query(
    query:             EvaluationQuery,
    generation_result: GenerationResult,
    client:            AzureOpenAI,
    model:             str = JUDGE_MODEL
) -> NegativeEvaluationResult:
    """
    Router principale per la valutazione delle query negative.
    Seleziona la funzione di valutazione corretta in base a expected_behavior.
    """
    start = time.time()

    if query.expected_behavior == ExpectedBehavior.NO_ANSWER:
        score, label, reasoning = evaluate_no_answer_behavior(
            query, generation_result.generated_answer, client, model
        )
    elif query.expected_behavior == ExpectedBehavior.CORRECTION:
        score, label, reasoning = evaluate_correction_behavior(
            query, generation_result.generated_answer, client, model
        )
    elif query.expected_behavior == ExpectedBehavior.CLARIFICATION:
        score, label, reasoning = evaluate_clarification_behavior(
            query, generation_result.generated_answer, client, model
        )
    else:
        score, label, reasoning = 0.0, "TIPO NON GESTITO", f"expected_behavior sconosciuto: {query.expected_behavior}"

    return NegativeEvaluationResult(
        query_id          = query.query_id,
        expected_behavior = query.expected_behavior,
        behavior_score    = score,
        behavior_label    = label,
        llm_reasoning     = reasoning,
        generation_time   = time.time() - start
    )

def aggregate_negative_metrics(results: List[NegativeEvaluationResult]) -> Dict[str, float]:
    """Calcola metriche aggregate per le query negative, suddivise per tipo."""
    if not results:
        return {}

    by_type: Dict[str, List[float]] = {}
    for r in results:
        by_type.setdefault(r.expected_behavior, []).append(r.behavior_score)

    metrics   = {}
    all_scores = []
    for behavior, scores in by_type.items():
        metrics[f"robustness_{behavior}"] = float(np.mean(scores))
        metrics[f"count_{behavior}"]      = len(scores)
        all_scores.extend(scores)

    metrics["robustness_overall"] = float(np.mean(all_scores))
    metrics["count_total"]        = len(all_scores)
    return metrics


# ==================== AGGREGAZIONE METRICHE POSITIVE ====================

def aggregate_retrieval_metrics(
    per_query_metrics: List[Dict[str, float]],
    retrieval_results: List[RetrievalResult]
) -> Dict[str, float]:
    """
    Aggrega le metriche di retrieval su tutte le query positive.
    retrieval_time e selection_time vengono calcolati come media
    su tutte le query (incluse le negative).
    """
    if not per_query_metrics:
        return {}

    metric_keys = set(k for m in per_query_metrics for k in m.keys())
    aggregated  = {}
    for key in metric_keys:
        values = [m[key] for m in per_query_metrics if key in m and m[key] is not None]
        if values:
            aggregated[f"avg_{key}"] = float(np.mean(values))

    if retrieval_results:
        aggregated["avg_retrieval_time"] = float(
            np.mean([r.retrieval_time for r in retrieval_results])
        )
        aggregated["avg_selection_time"] = float(
            np.mean([r.selection_time for r in retrieval_results])
        )

    return aggregated

def aggregate_generation_metrics(
    per_query_metrics:  List[Dict[str, float]],
    generation_results: List[GenerationResult]
) -> Dict[str, float]:
    """
    Aggrega le metriche di generation su tutte le query positive.
    """
    if not per_query_metrics:
        return {}

    metric_keys = set(k for m in per_query_metrics for k in m.keys())
    aggregated  = {}
    for key in metric_keys:
        values = [m[key] for m in per_query_metrics if key in m and m[key] is not None]
        if values:
            aggregated[f"avg_{key}"] = float(np.mean(values))

    if generation_results:
        aggregated["avg_generation_time"] = float(
            np.mean([g.generation_time for g in generation_results])
        )

    return aggregated


# ==================== INTEGRAZIONE CON IL SISTEMA REALE ====================

def run_pipeline_and_collect(
    queries:         List[EvaluationQuery],
    strategy:        str,
    top_k:           int,
    semantic_weight: float = 0.7
) -> Tuple[List[RetrievalResult], List[GenerationResult]]:

    retrieval_results  = []
    generation_results = []

    print(f"\n{'='*70}")
    print(f"PIPELINE ({strategy.upper()}): {len(queries)} query  |  top_k={top_k}")
    print(f"{'='*70}\n")

    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {query.query_text[:60]}...")

        try:
            result = your_llm.run_pipeline_for_evaluation(
                user_prompt     = query.query_text,
                strategy        = strategy,
                top_k           = top_k,
                semantic_weight = semantic_weight
            )

            docs_after       = result["docs_after_selection"]
            tool_decision    = result["tool_decision"]
            selection_reason = result.get("selection_reason", "")
            n_before         = result["n_docs_before"]
            n_after          = result["n_docs_after"]
            timings          = result["timings"]

            if tool_decision:
                use_sem = tool_decision.get("use_semantic", False)
                use_kw  = tool_decision.get("use_keyword",  False)
                mode    = ("HYBRID"   if use_sem and use_kw else
                           "SEMANTIC" if use_sem else
                           "KEYWORD"  if use_kw  else "NESSUNA")
                print(f"  → Tool selection: {mode}  |  {tool_decision.get('reason','')[:70]}")

            print(f"  ✓ {n_before} doc recuperati → {n_after} selezionati")

            ret = RetrievalResult(
                query_id                = query.query_id,
                retrieved_docs          = docs_after,
                selected_docs           = docs_after,
                retrieval_time          = timings["retrieval_s"],
                selection_time          = timings["selection_s"],
                n_docs_before_selection = n_before,
                n_docs_after_selection  = n_after,
                tool_decision           = tool_decision
            )
            gen = GenerationResult(
                query_id         = query.query_id,
                generated_answer = result["generated_answer"],
                context_docs     = docs_after,
                generation_time  = timings["generation_s"],
                selection_reason = selection_reason
            )

        except Exception as e:
            print(f"  ✗ Errore: {e}")
            import traceback; traceback.print_exc()
            ret = RetrievalResult(
                query_id=query.query_id, retrieved_docs=[], selected_docs=[],
                retrieval_time=0.0, selection_time=0.0,
                n_docs_before_selection=0, n_docs_after_selection=0, tool_decision=None
            )
            gen = GenerationResult(
                query_id=query.query_id, generated_answer=f"[ERRORE] {e}",
                context_docs=[], generation_time=0.0
            )

        retrieval_results.append(ret)
        generation_results.append(gen)

    print(f"\n{'='*70}\n")
    return retrieval_results, generation_results

# ==================== STATISTICHE TOOL SELECTION ====================

def aggregate_tool_selection_stats(retrieval_results: List[RetrievalResult]) -> Dict:
    decisions = [r.tool_decision for r in retrieval_results if r.tool_decision is not None]
    if not decisions:
        return {}

    total         = len(decisions)
    semantic_only = sum(1 for d in decisions if     d.get("use_semantic") and not d.get("use_keyword"))
    keyword_only  = sum(1 for d in decisions if     d.get("use_keyword")  and not d.get("use_semantic"))
    both          = sum(1 for d in decisions if     d.get("use_semantic") and     d.get("use_keyword"))
    none_selected = sum(1 for d in decisions if not d.get("use_semantic") and not d.get("use_keyword"))

    return {
        "total_queries":       total,
        "semantic_only_count": semantic_only,
        "keyword_only_count":  keyword_only,
        "both_count":          both,
        "none_count":          none_selected,
        "pct_semantic_only":   round(semantic_only / total * 100, 1),
        "pct_keyword_only":    round(keyword_only  / total * 100, 1),
        "pct_both":            round(both          / total * 100, 1),
        "pct_none":            round(none_selected / total * 100, 1),
        "pct_avoided_hybrid":  round((semantic_only + keyword_only + none_selected) / total * 100, 1),
        "per_query_decisions": [
            {
                "query_id":    r.query_id,
                "use_semantic": r.tool_decision.get("use_semantic"),
                "use_keyword":  r.tool_decision.get("use_keyword"),
                "mode": (
                    "both"          if r.tool_decision.get("use_semantic") and r.tool_decision.get("use_keyword")
                    else "semantic_only" if r.tool_decision.get("use_semantic")
                    else "keyword_only"  if r.tool_decision.get("use_keyword")
                    else "none"
                ),
                "reason": r.tool_decision.get("reason", "")
            }
            for r in retrieval_results if r.tool_decision is not None
        ]
    }


# ==================== VALUTAZIONE COMPLETA ====================

def run_full_evaluation(
    gold_queries:       List[EvaluationQuery],
    retrieval_results:  List[RetrievalResult],
    generation_results: List[GenerationResult],
    configuration:      Dict,
    client:             AzureOpenAI,
    k_values:           List[int] = [5, 10, 15],
    llm_model:          str = JUDGE_MODEL,
    embedding_model:    str = "text-embedding-3-large"
) -> TestResult:

    start_time = time.time()
    print("\n" + "="*70)
    print("VALUTAZIONE METRICHE")
    print("="*70)

    per_query_retrieval_metrics  = []
    per_query_generation_metrics = []
    negative_eval_results        = []
    per_query_details            = []

    for i, (query, ret, gen) in enumerate(
        zip(gold_queries, retrieval_results, generation_results), 1
    ):
        print(f"\n[{i}/{len(gold_queries)}] {query.query_id}  {'[NEGATIVA]' if query.is_negative else ''}")

        if query.is_negative:
            print(f"  - Comportamento atteso: {query.expected_behavior}")
            neg_result = evaluate_negative_query(query, gen, client, llm_model)
            negative_eval_results.append(neg_result)
            print(f"  → {neg_result.behavior_label}")

            per_query_details.append({
                "query_id":          query.query_id,
                "query_text":        query.query_text,
                "is_negative":       True,
                "expected_behavior": query.expected_behavior,
                "behavior_score":    neg_result.behavior_score,
                "behavior_label":    neg_result.behavior_label,
                "llm_reasoning":     neg_result.llm_reasoning,
                "generated_answer":  gen.generated_answer,
                "n_docs_selected":   ret.n_docs_after_selection,
            })

        else:
            print("  - Metriche retrieval...")
            ret_metrics = evaluate_retrieval_for_query(
                query, ret, client, k_values, llm_model
            )
            per_query_retrieval_metrics.append(ret_metrics)

            print("  - Metriche generation...")
            gen_metrics = evaluate_generation_for_query(
                query, gen, client, llm_model, embedding_model
            )
            per_query_generation_metrics.append(gen_metrics)

            # Metadati leggeri per il JSON (no contenuti completi)
            selected_ids = [
                f"{d['numero']}_{d['progressivo']}"
                for d in ret.selected_docs
            ]
            per_query_details.append({
                "query_id":               query.query_id,
                "query_text":             query.query_text,
                "is_negative":            False,
                "retrieval_metrics":      ret_metrics,
                "generation_metrics":     gen_metrics,
                "generated_answer":       gen.generated_answer,
                "selected_doc_ids":       selected_ids,
                "n_docs_before_selection": ret.n_docs_before_selection,
                "n_docs_after_selection":  ret.n_docs_after_selection,
                "retrieval_time":         ret.retrieval_time,
                "selection_time":         ret.selection_time,
                "generation_time":        gen.generation_time,
                "selection_reason":       gen.selection_reason,
            })

        print(f"  ✓ {query.query_id} completata")

    print("\n" + "="*70)
    print("AGGREGAZIONE METRICHE")
    print("="*70)

    aggregated_retrieval  = aggregate_retrieval_metrics(
        per_query_retrieval_metrics, retrieval_results
    )
    aggregated_generation = aggregate_generation_metrics(
        per_query_generation_metrics, generation_results
    )
    aggregated_negative   = aggregate_negative_metrics(negative_eval_results)

    return TestResult(
        test_id               = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        timestamp             = datetime.now().isoformat(),
        configuration         = configuration,
        retrieval_metrics     = aggregated_retrieval,
        generation_metrics    = aggregated_generation,
        negative_eval_metrics = aggregated_negative,
        per_query_details     = per_query_details,
        total_evaluation_time = time.time() - start_time
    )


# ==================== GESTIONE RISULTATI ====================

def save_test_result(test_result: TestResult, output_dir: str = "evaluation_results"):
    Path(output_dir).mkdir(exist_ok=True)
    filepath = Path(output_dir) / f"{test_result.test_id}.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(asdict(test_result), f, indent=2, ensure_ascii=False)
    print(f"\n✓ Risultati salvati: {filepath}")

def print_test_result(test_result: TestResult):
    print("\n" + "="*70)
    print(f"TEST REPORT: {test_result.test_id}")
    print("="*70)
    print(f"\n📅 Timestamp: {test_result.timestamp}")
    print(f"⏱️  Tempo totale: {test_result.total_evaluation_time:.2f}s")

    print("\n📋 CONFIGURAZIONE:")
    for k, v in test_result.configuration.items():
        if k != "tool_selection_stats":
            print(f"  • {k}: {v}")

    print("\n📊 METRICHE RETRIEVAL (query positive):")
    for metric, value in sorted(test_result.retrieval_metrics.items()):
        if isinstance(value, float):
            print(f"  • {metric}: {value:.4f}")
        else:
            print(f"  • {metric}: {value}")

    print("\n📝 METRICHE GENERATION (query positive):")
    for metric, value in sorted(test_result.generation_metrics.items()):
        if value is not None and isinstance(value, float):
            print(f"  • {metric}: {value:.4f}")

    print("\n🛡️  ROBUSTEZZA (query negative):")
    for metric, value in sorted(test_result.negative_eval_metrics.items()):
        if isinstance(value, float):
            print(f"  • {metric}: {value:.4f}")
        else:
            print(f"  • {metric}: {value}")

    print("\n" + "="*70)

def compare_test_results(test_results: List[TestResult]):
    if not test_results:
        return

    print("\n" + "="*140)
    print("CONFRONTO RISULTATI TEST")
    print("="*140)

    col = 38
    print(
        f"\n{'Configurazione':<{col}} "
        f"{'SpP_set':>8} {'SpR_set':>8} {'LLM-Rel':>8} "
        f"{'Faith':>7} {'Relev':>7} {'SemSim':>7} {'Robust':>8} "
        f"{'N_bef':>6} {'N_aft':>6} "
        f"{'RetT':>7} {'SelT':>7} {'GenT':>7}"
    )
    print("-"*140)

    for result in test_results:
        name   = result.configuration.get('name', result.test_id)[:col - 2]
        spp    = result.retrieval_metrics.get('avg_span_precision_set', 0.0)
        spr    = result.retrieval_metrics.get('avg_span_recall_set', 0.0)
        llmrel = result.retrieval_metrics.get('avg_chunk_relevance_set', 0.0)
        faith  = result.generation_metrics.get('avg_faithfulness', 0.0)
        relev  = result.generation_metrics.get('avg_answer_relevancy', 0.0)
        semsim = result.generation_metrics.get('avg_semantic_similarity') or 0.0
        robust = result.negative_eval_metrics.get('robustness_overall', 0.0)
        n_bef  = result.retrieval_metrics.get('avg_n_docs_before_selection', 0.0)
        n_aft  = result.retrieval_metrics.get('avg_n_docs_after_selection', 0.0)
        ret_t  = result.retrieval_metrics.get('avg_retrieval_time', 0.0)
        sel_t  = result.retrieval_metrics.get('avg_selection_time', 0.0)
        gen_t  = result.generation_metrics.get('avg_generation_time', 0.0)

        print(
            f"{name:<{col}} "
            f"{spp:>8.4f} {spr:>8.4f} {llmrel:>8.4f} "
            f"{faith:>7.4f} {relev:>7.4f} {semsim:>7.4f} {robust:>8.4f} "
            f"{n_bef:>6.1f} {n_aft:>6.1f} "
            f"{ret_t:>6.2f}s {sel_t:>6.2f}s {gen_t:>6.2f}s"
        )

    print("="*140)
    print("\nLegenda:")
    print("  SpP_set  = Span Precision set-based — chunk selezionati che coprono ≥1 span / n_selezionati")
    print("  SpR_set  = Span Recall set-based — span coperti / totale span  ← metrica primaria di confronto")
    print("             Per multistage: calcolata sui doc selezionati da select_documents() (N ≤ 15)")
    print("             Per semantic/keyword/hybrid: calcolata sui top_k doc (retriever puro)")
    print("  LLM-Rel  = LLM-as-judge rilevanza media dei doc selezionati (rubrica 0/1/2, normalizzata)")
    print("  Faith    = Faithfulness con claim atomici (LLM-as-judge, gpt-4.1)")
    print("  Relev    = Answer Relevancy (LLM-as-judge, rubrica 0/1/2)")
    print("  SemSim   = Semantic Similarity (cosine su embeddings tra risposta e reference)")
    print("  Robust   = Robustezza su query negative (0=comportamento errato, 1=corretto)")
    print("  N_bef    = Media doc in ingresso a select_documents() (= top_k per non-multistage)")
    print("  N_aft    = Media doc selezionati (= N_bef per non-multistage, variabile per multistage)")
    print("  RetT     = Tempo medio retrieval per query (secondi)")
    print("  SelT     = Tempo medio select_documents() per query (0 per non-multistage)")
    print("  GenT     = Tempo medio generation per query (secondi)")
    print()

def print_tool_selection_stats(stats: Dict):
    if not stats:
        return
    print("\n" + "="*70)
    print("📊 STATISTICHE TOOL SELECTION (Multi-stage)")
    print("="*70)
    print(f"\n  Totale query analizzate : {stats['total_queries']}")
    print(f"\n  Semantic only           : {stats['semantic_only_count']:>3}  ({stats['pct_semantic_only']:>5.1f}%)")
    print(f"  Keyword only            : {stats['keyword_only_count']:>3}  ({stats['pct_keyword_only']:>5.1f}%)")
    print(f"  Entrambi (≈ hybrid)     : {stats['both_count']:>3}  ({stats['pct_both']:>5.1f}%)")
    print(f"  Nessuna ricerca         : {stats['none_count']:>3}  ({stats['pct_none']:>5.1f}%)")
    print(f"\n  → Risparmio vs hybrid fisso: {stats['pct_avoided_hybrid']:.1f}% delle query")
    print(f"    ha evitato di eseguire entrambi i retriever\n")
    print("  Dettaglio per query:")
    print(f"  {'Query ID':<12} {'Mode':<16} {'Reason'}")
    print("  " + "-"*66)
    for d in stats.get("per_query_decisions", []):
        reason_short = d['reason'][:45] + "..." if len(d['reason']) > 45 else d['reason']
        print(f"  {d['query_id']:<12} {d['mode']:<16} {reason_short}")
    print("="*70)


# ==================== TEST PRINCIPALE ====================

def run_test_with_your_pipeline(
    gold_dataset_path: str,
    configuration:     dict,
    search_strategy:   str = SearchStrategy.SEMANTIC,
    results_dir:       str = "evaluation_results"
):
    print("\n" + "="*70)
    print(f"TEST: {configuration.get('name', 'Unnamed')}")
    print(f"Search Strategy: {search_strategy}")
    print("="*70)

    print("\n[1/3] Caricamento dataset GOLD...")
    gold_queries = load_gold_dataset(gold_dataset_path)

    print(f"\n[2/3] Esecuzione pipeline ({search_strategy})...")
    top_k = configuration.get('top_k', 10)
    retrieval_results, generation_results = run_pipeline_and_collect(
        queries         = gold_queries,
        strategy        = search_strategy,
        top_k           = top_k,
        semantic_weight = configuration.get('semantic_weight', 0.7)
    )

    print("\n[3/3] Valutazione metriche...")
    client = load_openai_client()

    test_result = run_full_evaluation(
        gold_queries       = gold_queries,
        retrieval_results  = retrieval_results,
        generation_results = generation_results,
        configuration      = configuration,
        client             = client,
        k_values           = [5, 10, 15],
        llm_model          = JUDGE_MODEL,
        embedding_model    = "text-embedding-3-large"
    )

    print_test_result(test_result)

    if search_strategy == SearchStrategy.MULTISTAGE:
        tool_stats = aggregate_tool_selection_stats(retrieval_results)
        print_tool_selection_stats(tool_stats)
        test_result.configuration["tool_selection_stats"] = tool_stats

    save_test_result(test_result, results_dir)
    return test_result


# ==================== ABLATION STUDY ====================

def run_ablation_study(
    gold_dataset_path: str       = GOLD_DATASET_PATH,
    results_dir:       str       = "evaluation_results/ablation",
    dimensions:        List[str] = None,
    variant:           str       = None
) -> Dict[str, "TestResult"]:
    """
    Ablation study sequenziale: parte dalla BASELINE_CONFIG e varia
    una dimensione per volta, mantenendo tutto il resto fisso.

    Dimensioni disponibili:
      — Senza re-indicizzazione del DB (eseguibili in una sola sessione) —
      A) Strategia di search   : multistage (baseline) | hybrid | semantic | keyword
      B) Modello LLM generativo: gpt-4.1 (baseline) | gpt-5
      C) Top-k documenti       : 15 (baseline) | 5

      — Con re-indicizzazione del DB (una variante per sessione) —
      D) Tipo di chunking: D1_recursive_custom_baseline | D2_fixed_size | D3_recursive_standard
            Tutte con size=1024, overlap=150 — varia solo la strategia di chunking.
      E) Dimensione chunk      : E1_chunk1024_overlap150_baseline | E2_chunk512_overlap100
      F) Modello embedding     : F1_ada002_baseline | F2_embedding3large

    Per le dimensioni D, E, F è OBBLIGATORIO specificare --variant, poiché ogni
    variante richiede un DB indicizzato con parametri diversi.

    Workflow corretto per D/E/F:
      1. Re-indicizza il DB con la configurazione target
      2. python evaluation_pipeline.py -d D --variant D2_fixed_size
      3. Re-indicizza con la configurazione successiva
      4. python evaluation_pipeline.py -d D --variant D3_recursive_standard
      ... e così via per E e F

    Per D ed E la metrica primaria di confronto è span_recall_set, invariante al
    chunking perché lavora sul testo originale invece che sugli ID dei chunk.
    """
    active = set(d.upper() for d in dimensions) if dimensions else None

    # Validazione: D/E/F richiedono --variant esplicito
    db_dependent = {"D", "E", "F"}
    if active and (active & db_dependent):
        if not variant:
            dims_str = ", ".join(sorted(active & db_dependent))
            print(
                f"\nERRORE: le dimensioni {dims_str} richiedono re-indicizzazione del DB.\n"
                f"Specifica la variante da eseguire con --variant.\n"
                f"Varianti disponibili:\n"
                f"  D: D1_recursive_custom_baseline | D2_fixed_size | D3_recursive_standard\n"
                f"  E: E1_chunk1024_overlap150_baseline | E2_chunk512_overlap100\n"
                f"  F: F1_ada002_baseline | F2_embedding3large\n"
                f"Esempio: python evaluation_pipeline.py -d D --variant D2_fixed_size"
            )
            return {}

    results: Dict[str, TestResult] = {}
    all_configs = []

    # ── A: Strategia di search (no re-indicizzazione) ────────────────────────
    if active is None or "A" in active:
        for strategy, name in [
            (SearchStrategy.MULTISTAGE, "A1_multistage_baseline"),
            (SearchStrategy.HYBRID,     "A2_hybrid_fixed"),
            (SearchStrategy.SEMANTIC,   "A3_semantic_only"),
            (SearchStrategy.KEYWORD,    "A4_keyword_only"),
        ]:
            cfg = {**BASELINE_CONFIG, "name": name, "search_strategy": strategy}
            all_configs.append(("A", cfg, strategy))

    # ── B: Modello LLM generativo (no re-indicizzazione) ─────────────────────
    if active is None or "B" in active:
        for llm_model, name in [
            #("gpt-4.1", "B1_gpt41_baseline"),
            #("gpt-5",   "B2_gpt5"),
        ]:
            cfg = {**BASELINE_CONFIG, "name": name, "llm_model": llm_model}
            all_configs.append(("B", cfg, BASELINE_CONFIG["search_strategy"]))

    # ── C: Top-k (no re-indicizzazione) ──────────────────────────────────────
    if active is None or "C" in active:
        for top_k, name in [
            #(15, "C1_topk15_baseline"),
            #(5,  "C2_topk5"),
            (10, "C3_topk10"),
        ]:
            cfg = {**BASELINE_CONFIG, "name": name, "top_k": top_k}
            all_configs.append(("C", cfg, BASELINE_CONFIG["search_strategy"]))

    # ── D: Tipo di chunking (richiede re-indicizzazione) ─────────────────────
    if active is None or "D" in active:
        d_variants = {
            "D1_recursive_custom_baseline": ("recursive_custom",   1024, 150),
            "D2_fixed_size":                ("fixed_size",         1024, 150),
            "D3_recursive_standard":        ("recursive_standard", 1024, 150),
        }
        for name, (chunking, chunk_size, overlap) in d_variants.items():
            if variant and name != variant:
                continue
            cfg = {**BASELINE_CONFIG, "name": name,
                "chunking": chunking, "chunk_size": chunk_size, "chunk_overlap": overlap}
            all_configs.append(("D", cfg, BASELINE_CONFIG["search_strategy"]))

    # ── E: Dimensione chunk (richiede re-indicizzazione) ──────────────────────
    if active is None or "E" in active:
        e_variants = {
            "E1_chunk1024_overlap150_baseline": (1024, 150),
            "E2_chunk512_overlap100":           (512,  100),
        }
        for name, (chunk_size, overlap) in e_variants.items():
            if variant and name != variant:
                continue
            cfg = {**BASELINE_CONFIG, "name": name,
                   "chunk_size": chunk_size, "chunk_overlap": overlap}
            all_configs.append(("E", cfg, BASELINE_CONFIG["search_strategy"]))

    # ── F: Modello embedding (richiede re-indicizzazione) ─────────────────────
    if active is None or "F" in active:
        f_variants = {
            "F1_ada002_baseline": "text-embedding-ada-002",
            "F2_embedding3large": "text-embedding-3-large",
        }
        for name, emb_model in f_variants.items():
            if variant and name != variant:
                continue
            cfg = {**BASELINE_CONFIG, "name": name, "embedding_model": emb_model}
            all_configs.append(("F", cfg, BASELINE_CONFIG["search_strategy"]))

    if not all_configs:
        if variant:
            print(f"Variante '{variant}' non trovata nelle dimensioni specificate.")
        else:
            print(f"Nessuna dimensione valida tra quelle specificate: {dimensions}")
        return results

    # ── Esecuzione ────────────────────────────────────────────────────────
    total      = len(all_configs)
    dims_label = ", ".join(sorted(active)) if active else "A-F"
    print(f"\n{'='*70}")
    print(f"ABLATION STUDY — dimensioni: {dims_label} ({total} esperimenti)")
    print(f"{'='*70}\n")

    for i, (dimension, cfg, strategy) in enumerate(all_configs, 1):
        print(f"\n[{i}/{total}] [{dimension}]  {cfg['name']}")
        result = run_test_with_your_pipeline(
            gold_dataset_path = gold_dataset_path,
            configuration     = cfg,
            search_strategy   = strategy,
            results_dir       = results_dir
        )
        results[cfg["name"]] = result

    print(f"\n{'='*70}")
    print(f"ABLATION STUDY COMPLETATO — {len(results)}/{total} esperimenti")
    print(f"{'='*70}")
    compare_test_results(list(results.values()))
    return results


def run_smoke_test(strategy: str = SearchStrategy.MULTISTAGE) -> TestResult:
    """
    Esegue un test rapido su una singola strategia con la configurazione baseline.
    Non produce dati della tesi — serve a verificare che la pipeline funzioni
    end-to-end prima di lanciare l'ablation study completo.

    Uso da codice:
        from evaluation_pipeline import run_smoke_test, SearchStrategy
        result = run_smoke_test(strategy=SearchStrategy.MULTISTAGE)

    Uso da terminale:
        python evaluation_pipeline.py --smoke-test [semantic|keyword|hybrid|multistage]
    """
    config = dict(BASELINE_CONFIG)
    config["name"] = f"Smoke Test — {strategy}"
    return run_test_with_your_pipeline(
        gold_dataset_path = GOLD_DATASET_PATH,
        configuration     = config,
        search_strategy   = strategy,
        results_dir       = "evaluation_results/smoke"
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Pipeline di valutazione sistema RAG — Tesi magistrale"
    )
    parser.add_argument(
        "--dimensions", "-d",
        nargs="+",
        metavar="DIM",
        help="Dimensioni ablation da eseguire (es. A E F). Default: tutte (A-F)."
    )
    parser.add_argument(
        "--smoke-test", "-s",
        metavar="STRATEGY",
        nargs="?",
        const="multistage",
        help="Smoke test rapido invece dell'ablation. "
             "Strategie: semantic | keyword | hybrid | multistage (default)."
    )
    parser.add_argument(
        "--variant", "-v",
        metavar="VARIANT",
        default=None,
        help=(
            "Variante specifica da eseguire per dimensioni D/E/F (obbligatorio per queste). "
            "Es: D2_fixed_size | D3_recursive_standard | "
            "E2_chunk512_overlap100 | F2_embedding3large. "
            "Non ha effetto sulle dimensioni A/B/C."
        )
    )
    parser.add_argument(
        "--results-dir", "-o",
        default="evaluation_results/ablation",
        help="Directory di output per i risultati JSON."
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PIPELINE DI VALUTAZIONE SISTEMA RAG")
    print("=" * 70)

    if args.smoke_test is not None:
        strategy_map = {
            "semantic":   SearchStrategy.SEMANTIC,
            "keyword":    SearchStrategy.KEYWORD,
            "hybrid":     SearchStrategy.HYBRID,
            "multistage": SearchStrategy.MULTISTAGE,
        }
        strategy = strategy_map.get(args.smoke_test.lower())
        if strategy is None:
            print(f"Strategia non valida: {args.smoke_test!r}. "
                  f"Usa: {', '.join(strategy_map)}")
            return
        print(f"\nModalità: smoke test — {args.smoke_test}")
        run_smoke_test(strategy)
    else:
        dims  = args.dimensions if args.dimensions else None
        label = ", ".join(dims) if dims else "A-F (complete)"
        print(f"\nModalità: ablation study — dimensioni {label}")
        run_ablation_study(
            gold_dataset_path = GOLD_DATASET_PATH,
            results_dir       = args.results_dir,
            dimensions        = dims,
            variant           = args.variant
        )


if __name__ == "__main__":
    main()