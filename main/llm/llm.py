"""
llm.py
======
Core pipeline for the Adaptive Multi-Stage RAG system.

Pipeline stages
---------------
1. decide_tools      — LLM selects which retrievers to activate (semantic / keyword / both / none)
2. select_documents  — LLM filters and re-ranks retrieved chunks by relevance
3. generate_final_answer — LLM produces a streamed, cited answer
4. gpt_request       — Orchestrates the full chatbot flow
5. summarize_*       — Manages long chat-history compression
6. run_pipeline_for_evaluation — Thin wrapper used by the evaluation harness
"""

from dotenv import load_dotenv
import json
import os
import time
import sys

try:
    from llm.search import semantic_search, keyword_search
except ImportError:
    from search import semantic_search, keyword_search

import tiktoken
from difflib import SequenceMatcher

# Config loader — resolves config.yaml from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config_loader import cfg

# ---------------------------------------------------------------------------
# 0. CLIENT SETUP
# ---------------------------------------------------------------------------

load_dotenv()

def _build_client():
    """
    Build an OpenAI-compatible client from environment variables.

    Azure OpenAI  (set LLM_URL + LLM_VERSION):
        Requires: OPENAI_API_KEY, LLM_URL, LLM_VERSION, LLM_MODEL

    Standard OpenAI  (leave LLM_URL unset or empty):
        Requires: OPENAI_API_KEY, LLM_MODEL
        Optional: LLM_BASE_URL (to use a compatible third-party endpoint)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    llm_url = os.getenv("LLM_URL", "").strip()

    if llm_url:
        # Azure OpenAI
        from openai import AzureOpenAI
        return AzureOpenAI(
            azure_endpoint=llm_url,
            api_key=api_key,
            api_version=os.getenv("LLM_VERSION", "2024-02-01"),
        )
    else:
        # Standard OpenAI (or compatible endpoint)
        from openai import OpenAI
        base_url = os.getenv("LLM_BASE_URL", None)
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)


client = _build_client()
MODEL_NAME = os.getenv("LLM_MODEL", "gpt-4o")


# ---------------------------------------------------------------------------
# Helper: build a clickable document link from its numeric ID
# ---------------------------------------------------------------------------

def _doc_link(doc_id: str) -> str:
    """
    Return a Markdown hyperlink for the given document ID, using the
    document_url_template from config.yaml.
    If no template is configured, returns the bare ID.
    """
    tmpl = (cfg.domain.get("document_url_template") or "").strip()
    if tmpl:
        url = tmpl.replace("{id}", str(doc_id))
        return f"[{doc_id}]({url})"
    return str(doc_id)


# ---------------------------------------------------------------------------
# 1. TOOL SELECTION
# ---------------------------------------------------------------------------

def decide_tools(prompt: str) -> dict:
    """
    Step 1 — decide which retrievers to use for the given user prompt.

    Returns a dict with keys:
        use_semantic (bool), use_keyword (bool), reason (str)
    """
    domain_name   = cfg.domain.name
    document_type = cfg.domain.document_type
    out_of_scope  = cfg.domain.get("out_of_scope_description") or f"topics unrelated to {domain_name}"

    # Override prompt from config.yaml if provided
    custom_prompt = cfg.prompts.get("decide_tools")
    if custom_prompt:
        system_message = custom_prompt.format(
            domain_name=domain_name,
            document_type=document_type,
            out_of_scope=out_of_scope,
        )
    else:
        examples = [
            {
                "prompt": f"How does the order approval workflow work in {domain_name}?",
                "decision": {
                    "use_semantic": True,
                    "use_keyword": False,
                    "reason": f"General workflow question about {domain_name}. No precise terms to search — semantic retrieval finds conceptually related documents.",
                },
            },
            {
                "prompt": f"What is the contact email for customer Acme Corp?",
                "decision": {
                    "use_semantic": False,
                    "use_keyword": True,
                    "reason": "Specific data tied to a named entity. Keyword search finds the exact document mentioning 'Acme Corp' more effectively.",
                },
            },
            {
                "prompt": f"Custom payroll module with a bespoke export format for a manufacturing company",
                "decision": {
                    "use_semantic": True,
                    "use_keyword": True,
                    "reason": f"Complex customisation request: semantic search finds similar cases, keyword search finds explicit references to the specific format.",
                },
            },
            {
                "prompt": "What are the best agile project management techniques?",
                "decision": {
                    "use_semantic": False,
                    "use_keyword": False,
                    "reason": f"The request is not related to {domain_name} or its customisations. Documents do not contain this information.",
                },
            },
        ]

        few_shot_text = ""
        for ex in examples:
            few_shot_text += f"User prompt: {ex['prompt']}\nDecision: {json.dumps(ex['decision'])}\n\n"

        system_message = f"""
You are a decision-making assistant for the knowledge base of {domain_name}.
Your task is to decide which search tools to use to find relevant {document_type}s.

Decision rules (in priority order):

1. If the request is OUT OF SCOPE ({out_of_scope}):
   → use_semantic: false, use_keyword: false

2. If the request contains PRECISE, IDENTIFYING TERMS (customer name, specific module name,
   exact technical term, document number):
   → use_semantic: false, use_keyword: true

3. If the request is about GENERAL CONCEPTS, WORKFLOWS or FEATURES described
   generically without a precise technical term to search for:
   → use_semantic: true, use_keyword: false

4. Only if the request combines COMPLEX CUSTOMISATIONS where both conceptual
   similarity AND precise references are needed, or if it is genuinely ambiguous
   between cases 2 and 3:
   → use_semantic: true, use_keyword: true

The "both" choice (case 4) is the most expensive — use it only when necessary.

Always reply in JSON format:
{{
    "use_semantic": true/false,
    "use_keyword": true/false,
    "reason": "<which rule applies and why>"
}}

Examples:
{few_shot_text}
"""

    user_message = f"User prompt: {prompt}\nDecision:"

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
    )

    decision_text = response.choices[0].message.content

    try:
        decision = json.loads(decision_text)
    except json.JSONDecodeError:
        decision = {
            "use_semantic": False,
            "use_keyword": False,
            "reason": f"Parsing error: {decision_text}",
        }

    return decision


# ---------------------------------------------------------------------------
# 2. DOCUMENT SELECTION
# ---------------------------------------------------------------------------

def select_documents(user_prompt: str, chunks: list) -> dict:
    """
    Step 2 — filter and re-rank retrieved chunks by relevance.

    Returns a dict with keys:
        relevant_docs (list), irrelevant_docs (list), reason (str)
    """
    domain_name   = cfg.domain.name
    document_type = cfg.domain.document_type

    # ── Deduplication: keep the most similar chunk per (numero, progressivo) ──
    seen = {}
    for doc in chunks:
        key = (doc["numero"], doc["progressivo"])
        if key not in seen:
            seen[key] = doc
        else:
            existing  = seen[key].get("retrieval_sources", [])
            incoming  = doc.get("retrieval_sources", [])
            seen[key]["retrieval_sources"] = list(set(existing + incoming))
            if doc.get("similarity", 0) > seen[key].get("similarity", 0):
                seen[key]["similarity"] = doc["similarity"]
    chunks = list(seen.values())

    # ── Template filtering: discard chunks that are near-copies of the blank template ──
    documents = chunks
    try:
        template_path = os.path.join(
            os.path.dirname(__file__), "..", "evaluation", "sample_template.docx"
        )
        if os.path.exists(template_path):
            from docx import Document as DocxDocument
            tmpl_doc = DocxDocument(template_path)
            template_text = "\n".join([p.text for p in tmpl_doc.paragraphs if p.text.strip()])
            threshold = cfg.retrieval.get("template_similarity_threshold") or 0.9
            documents = []
            for chunk in chunks:
                sim = SequenceMatcher(None, chunk["content"], template_text).ratio()
                if sim < threshold:
                    documents.append(chunk)
    except Exception:
        pass  # If template file is missing or unreadable, skip this step

    if not documents:
        return {
            "relevant_docs": [],
            "irrelevant_docs": list(range(len(chunks))),
            "reason": "All chunks are similar to the blank template and therefore not useful.",
        }

    # ── LLM selection ──
    custom_prompt = cfg.prompts.get("select_documents")
    if custom_prompt:
        system_message = custom_prompt.format(
            domain_name=domain_name,
            document_type=document_type,
        )
    else:
        examples = [
            {
                "prompt": "How can I generate a warehouse movement report?",
                "documents": [
                    {"titolo": "Monthly sales report",     "content": "Sales data per customer and product.",               "autore": "Alice", "cliente": "ClientA", "similarity": 0.42},
                    {"titolo": "Warehouse movements",      "content": "Detailed list of warehouse movements with date/product filters.", "autore": "Bob", "cliente": "ClientB", "similarity": 0.91},
                    {"titolo": "Warehouse configuration",  "content": "Instructions to configure the warehouse module and optional reports.", "autore": "Carol", "cliente": "ClientC", "similarity": 0.75},
                ],
                "decision": {
                    "relevant_docs": [1, 2],
                    "irrelevant_docs": [0],
                    "reason": "Doc 1 directly covers warehouse movements. Doc 2 is useful as technical support for configuration. Doc 0 is not relevant.",
                },
            },
            {
                "prompt": "How can I set unique keys for customers?",
                "documents": [
                    {"titolo": "Unique customer keys",  "content": "How to configure unique keys to avoid duplicate customer records.", "autore": "Dave", "cliente": "ClientX", "similarity": 0.95},
                    {"titolo": "Warehouse management",  "content": "Instructions for creating new warehouses and assigning codes.",      "autore": "Eve",  "cliente": "ClientY", "similarity": 0.55},
                    {"titolo": "General parameters",    "content": "Company general settings — does not cover unique keys.",             "autore": "Frank","cliente": "ClientZ", "similarity": 0.40},
                ],
                "decision": {
                    "relevant_docs": [0],
                    "irrelevant_docs": [1, 2],
                    "reason": "Only Doc 0 directly covers unique key configuration. Others are off-topic.",
                },
            },
        ]

        few_shot_text = ""
        for ex in examples:
            docs_text = "\n".join([
                f"{i}: {d['titolo']} - {d['content']} (author: {d['autore']}, client: {d['cliente']})"
                for i, d in enumerate(ex["documents"])
            ])
            few_shot_text += (
                f"User prompt: {ex['prompt']}\nDocuments:\n{docs_text}\n"
                f"Decision: {json.dumps(ex['decision'])}\n\n"
            )

        system_message = f"""
You are a relevance filter for a RAG system built on the knowledge base of {domain_name}.
Your task is to identify which of the retrieved {document_type}s are useful or potentially useful
for answering the user's request.

Notes:
- The retrieval pipeline always returns the most relevant documents, but not all are genuinely useful.
- Select only documents coherent with the request — up to a maximum of {cfg.retrieval.get("max_selected_docs", 15)}.
- Each document includes its content and cosine similarity score with the user prompt.
  Use both as signals, but consider the actual content and coherence with the question.
- If a document has retrieval_sources: ["semantic", "keyword"], it was retrieved by BOTH techniques —
  this is a strong relevance signal and should be prioritised.

Relevance criteria:
1. A document is relevant if it contains concrete information that helps answer the request
   (standard features AND custom plugins/implementations are both valid).
2. A document is acceptable if it covers the topic only partially or from a related angle —
   exclude it ONLY if it is clearly off-topic.
3. Reorder relevant document indices from LEAST to MOST relevant (ascending relevance).

Always reply in JSON format:
{{
    "relevant_docs": [indices of useful documents in ascending relevance order],
    "irrelevant_docs": [indices of non-useful documents],
    "reason": "<brief explanation>"
}}

Examples:
{few_shot_text}
"""

    docs_text = "\n".join([
        f"{i}: {d['titolo']} - {d['content']} "
        f"(author: {d['autore']}, client: {d['cliente']}, "
        f"sources: {'+'.join(d.get('retrieval_sources', ['unknown']))})"
        for i, d in enumerate(documents)
    ])
    user_message = f"{few_shot_text}User prompt: {user_prompt}\nDocuments:\n{docs_text}\nDecision:"

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
    )

    decision_text = response.choices[0].message.content

    try:
        decision_filtered = json.loads(decision_text)
        relevant_docs   = [documents[i] for i in decision_filtered.get("relevant_docs", [])]
        irrelevant_docs = [documents[i] for i in decision_filtered.get("irrelevant_docs", [])]
        decision = {
            "relevant_docs":   relevant_docs,
            "irrelevant_docs": irrelevant_docs,
            "reason":          decision_filtered.get("reason", "LLM-generated decision"),
        }
    except json.JSONDecodeError:
        decision = {
            "relevant_docs":   [],
            "irrelevant_docs": list(documents),
            "reason":          f"Parsing error: {decision_text}",
        }

    return decision


# ---------------------------------------------------------------------------
# 3. ANSWER GENERATION
# ---------------------------------------------------------------------------

def generate_final_answer(user_prompt: str, selected_docs: list, chat_history: list = None):
    """
    Step 3 — stream the final answer, with inline citations and a reference list.

    Yields successive string tokens (streaming response).
    """
    domain_name   = cfg.domain.name
    document_type = cfg.domain.document_type

    context_text = "\n\n".join([
        f"{document_type} ID: {d['numero']} - Title: {d['titolo']}, Chunk: {d['progressivo']}, "
        f"Author: {d['autore']}, Client: {d['cliente']}, "
        f"Retrieval sources: {'+'.join(d.get('retrieval_sources', ['unknown']))}\n"
        f" - Content: {d['content']}"
        for d in selected_docs
    ]) if selected_docs else f"No relevant {document_type}s found."

    # ── Few-shot examples (generic, no company-specific data) ──
    examples = [
        {
            "user_prompt": f"How can I create a new customer order in {domain_name}?",
            "documents": [
                {
                    "numero": "1001",
                    "titolo": f"Creating a Customer Order",
                    "progressivo": "2",
                    "content": "To create a new customer order, go to Sales > Orders > New. Fill in the required fields: Customer, Date, Warehouse and Payment Terms. Save to confirm.",
                    "autore": "Smith",
                    "cliente": "Sample Corp",
                },
                {
                    "numero": "1002",
                    "titolo": "Customer Master Data Management",
                    "progressivo": "0",
                    "content": "Customer master records must be created before inserting orders or sales documents.",
                    "autore": "Jones",
                    "cliente": "Sample Corp",
                },
            ],
            "answer": (
                f"To create a new customer order in **{domain_name}**, go to **Sales → Orders → New**. "
                "Fill in the required fields (Customer, Date, Warehouse, Payment Terms) and save to confirm [1]. "
                "Make sure the customer already exists in master data [2].\n\n"
                f"Reference documents:\n"
                f"1. {document_type}: {_doc_link('1001')} - Creating a Customer Order, Chunk: 3, Author: Smith, Client: Sample Corp\n"
                f"2. {document_type}: {_doc_link('1002')} - Customer Master Data Management, Chunk: 1, Author: Jones, Client: Sample Corp"
            ),
        },
        {
            "user_prompt": "What does the error 'warehouse not found' mean when saving an order?",
            "documents": [
                {
                    "numero": "2001",
                    "titolo": "Common Order Errors",
                    "progressivo": "6",
                    "content": "The 'warehouse not found' error occurs when the warehouse in the order is inactive or not linked to the selected company.",
                    "autore": "Brown",
                    "cliente": "Sample Corp",
                },
                {
                    "numero": "2002",
                    "titolo": "Warehouse Configuration",
                    "progressivo": "1",
                    "content": "To check active warehouses, go to Warehouses > Master Data. Enable the 'Active' flag to make them available in sales documents.",
                    "autore": "Green",
                    "cliente": "Sample Corp",
                },
            ],
            "answer": (
                "The **'warehouse not found'** error indicates that the selected warehouse is inactive or not linked to the current company [1]. "
                "To resolve, go to **Warehouses → Master Data**, verify the warehouse is active and linked to the correct company [2].\n\n"
                f"Reference documents:\n"
                f"1. {document_type}: {_doc_link('2001')} - Common Order Errors, Chunk: 7, Author: Brown, Client: Sample Corp\n"
                f"2. {document_type}: {_doc_link('2002')} - Warehouse Configuration, Chunk: 2, Author: Green, Client: Sample Corp"
            ),
        },
        {
            "user_prompt": "What is the company HR policy for employee leave?",
            "documents": [],
            "answer": (
                f"I'm sorry, but I could not find any information about HR leave policies in the available {document_type}s. "
                "Please refer to your HR department for this type of information."
            ),
        },
        {
            "user_prompt": "Is contact@example-customer.com the right email for Example Customer Ltd?",
            "documents": [
                {
                    "numero": "3001",
                    "titolo": "Example Customer Ltd — Master Data",
                    "progressivo": "0",
                    "content": "Client: Example Customer Ltd — Email: info@example-customer.com — Contact: John Doe.",
                    "autore": "Taylor",
                    "cliente": "Example Customer Ltd",
                },
            ],
            "answer": (
                "The email address you provided is **not correct**. According to the available documents, "
                "the email for **Example Customer Ltd** is **info@example-customer.com** [1].\n\n"
                f"Reference documents:\n"
                f"1. {document_type}: {_doc_link('3001')} - Example Customer Ltd Master Data, Chunk: 1, Author: Taylor, Client: Example Customer Ltd"
            ),
        },
    ]

    few_shot_text = ""
    for ex in examples:
        docs_text_example = "\n".join([
            f"{i+1}: {document_type} ID: {d['numero']} - Title: {d['titolo']}, Author: {d['autore']}, Client: {d['cliente']}\n - Content: {d['content']}"
            for i, d in enumerate(ex["documents"])
        ])
        few_shot_text += (
            f"User prompt: {ex['user_prompt']}\nDocuments:\n{docs_text_example}\n"
            f"Expected answer:\n{ex['answer']}\n\n"
        )

    # ── System prompt ──
    custom_prompt = cfg.prompts.get("generate_answer")
    if custom_prompt:
        system_message = custom_prompt.format(
            domain_name=domain_name,
            document_type=document_type,
        )
    else:
        system_message = f"""
You are an expert assistant for the knowledge base of {domain_name}.
You answer user questions using ONLY the information present in the provided {document_type}s.
You may use your pre-existing knowledge only to clarify the difference between standard features and customisations.

Rules:
1. If the provided documents do not contain specific, direct information that answers the question,
   state this explicitly. Do not infer, generalise, or answer based on knowledge external to the documents.
2. If a feature is standard and known from your training, you may clarify this, but always highlight
   when a feature is a customisation or plugin based on the provided {document_type}s.
3. Annotate every reference to a document with a number in square brackets [1], [2], etc.
4. At the end of the answer, provide the list of referenced documents in this format:
   1. {document_type}: [ID](url) - Title, Chunk: chunk_number, Author: author, Client: client
5. Keep the answer clear, structured and grounded in the provided documents for customisations.
6. If no documents are provided, answer naturally and professionally based on known standard features only.
7. Output in Markdown format (suitable for a Streamlit interface).
8. If information in the user's query contradicts the documents (e.g. a wrong email or parameter),
   explicitly flag the discrepancy and report the correct value from the documents.
9. Start the answer directly with content — never wrap it in a code fence (```markdown etc.).
10. If the query is ambiguous and admits multiple plausible interpretations within the domain,
    acknowledge the ambiguity, list the possible interpretations, and ask for clarification or
    cover all of them clearly in the response.

Expected answer examples:
{few_shot_text}
"""

    summarized_chat = summarize_chat_history(chat_history)

    user_message = f"""
User prompt:
{user_prompt}

Recent conversation context:
{summarized_chat}

Available documents:
{context_text}

Expected answer:
"""

    response_stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0.5,
        stream=True,
    )

    for event in response_stream:
        if not event.choices:
            continue
        choice = event.choices[0]
        delta = getattr(choice, "delta", None)
        if delta and getattr(delta, "content", None):
            yield delta.content


# ---------------------------------------------------------------------------
# CHATBOT MAIN FLOW
# ---------------------------------------------------------------------------

def gpt_request(messages):
    """Orchestrate the full chatbot pipeline and return a streaming generator."""
    top_k = int(cfg.retrieval.get("default_top_k") or 15)

    user_prompt = [m["content"] for m in messages if m["role"] == "user"][-1]

    tools = decide_tools(user_prompt)

    all_documents = []
    if tools["use_semantic"]:
        sem_docs = semantic_search(user_prompt, top_k)
        for d in sem_docs:
            d["retrieval_sources"] = ["semantic"]
        all_documents += sem_docs

    if tools["use_keyword"]:
        kw_docs = keyword_search(user_prompt, top_k)
        for d in kw_docs:
            d["retrieval_sources"] = ["keyword"]
        all_documents += kw_docs

    selected_docs = []
    if (tools["use_semantic"] or tools["use_keyword"]) and all_documents:
        document_selection = select_documents(user_prompt, all_documents)
        selected_docs = document_selection["relevant_docs"]

    return generate_final_answer(user_prompt, selected_docs, messages)


# ---------------------------------------------------------------------------
# CHAT HISTORY MANAGEMENT
# ---------------------------------------------------------------------------

def summarize_chat_history(chat_history, model_name=MODEL_NAME):
    """Compress chat history to stay within the context window."""
    if not chat_history:
        return ""

    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    max_tokens = 10000
    reserved_tokens_for_summary = 2000

    token_counts = [
        len(encoding.encode(f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}\n"))
        for m in chat_history
    ]
    total_tokens = sum(token_counts)

    if total_tokens <= max_tokens:
        return "".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}\n"
            for m in chat_history
        )

    recent_lines, old_lines = [], []
    recent_tokens, old_tokens = 0, 0
    half = (max_tokens - reserved_tokens_for_summary) // 2

    for m, t in zip(chat_history, token_counts):
        if old_tokens + t <= half:
            role = "User" if m["role"] == "user" else "Assistant"
            old_lines.append(f"{role}: {m['content']}\n")
            old_tokens += t
        else:
            break

    for m, t in zip(reversed(chat_history), reversed(token_counts)):
        if recent_tokens + t <= half:
            role = "User" if m["role"] == "user" else "Assistant"
            recent_lines.insert(0, f"{role}: {m['content']}\n")
            recent_tokens += t
        else:
            break

    start_idx = len(old_lines)
    end_idx = len(chat_history) - len(recent_lines)
    central_messages = chat_history[start_idx:end_idx]
    summary_text = ""
    if central_messages:
        summary_text = summarize_old_messages(central_messages, reserved_tokens_for_summary)
        summary_text = f"Summary of central messages: {summary_text}\n"

    return "".join(old_lines) + summary_text + "".join(recent_lines)


def summarize_old_messages(messages, max_tokens):
    """Ask the LLM to summarise an older portion of the chat history."""
    if not messages:
        return ""

    domain_name   = cfg.domain.name
    document_type = cfg.domain.document_type

    custom_prompt = cfg.prompts.get("summarize_history")
    if custom_prompt:
        system_prompt = custom_prompt.format(domain_name=domain_name, document_type=document_type)
    else:
        system_prompt = f"""
You are an assistant that summarises chat messages related to the knowledge base of {domain_name}.
The summary must be clear, concise and useful as context for future questions.

Criteria:
1. Keep information relevant for understanding the context of requests, noting whether they
   concern standard features or custom plugins / {document_type}s.
2. Synthesise concepts, removing redundant or non-essential details.
3. The summary should provide enough context for the LLM to answer subsequent questions correctly.
"""

    text_to_summarise = "".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}\n"
        for m in messages
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text_to_summarise},
        ],
        temperature=0.3,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# EVALUATION PIPELINE WRAPPER
# ---------------------------------------------------------------------------

def run_pipeline_for_evaluation(
    user_prompt: str,
    strategy: str,
    top_k: int = 15,
    chat_history: list = None,
    semantic_weight: float = 0.7,
) -> dict:
    """
    Thin wrapper that runs the RAG pipeline and returns a structured result
    dict consumed by the evaluation harness.

    strategy options: "multistage" | "hybrid" | "semantic" | "keyword"
    """
    chat_history = chat_history or []

    tool_decision    = None
    docs_after       = []
    selection_reason = ""
    t_selection      = 0.0

    # ── Retrieval ────────────────────────────────────────────────────────────
    t0 = time.time()

    if strategy == "multistage":
        tool_decision = decide_tools(user_prompt)
        use_semantic  = tool_decision["use_semantic"]
        use_keyword   = tool_decision["use_keyword"]

        all_docs = []
        if use_semantic:
            sem_docs = semantic_search(user_prompt, top_k)
            for d in sem_docs:
                d["retrieval_sources"] = ["semantic"]
            all_docs += sem_docs
        if use_keyword:
            kw_docs = keyword_search(user_prompt, top_k)
            for d in kw_docs:
                d["retrieval_sources"] = ["keyword"]
            all_docs += kw_docs

        n_before    = len(all_docs)
        t_retrieval = time.time() - t0

        # ── LLM selection ────────────────────────────────────────────────────
        t1 = time.time()
        if all_docs:
            selection_result = select_documents(user_prompt, all_docs)
            docs_after       = selection_result["relevant_docs"]
            selection_reason = selection_result["reason"]
        else:
            docs_after       = []
            selection_reason = "No documents retrieved."
        t_selection = time.time() - t1
        n_after = len(docs_after)

    elif strategy == "hybrid":
        keyword_weight = 1.0 - semantic_weight
        sem_docs = semantic_search(user_prompt, top_n=top_k * 2)
        for d in sem_docs:
            d["retrieval_sources"] = ["semantic"]
        kw_docs = keyword_search(user_prompt, top_n=top_k * 2)
        for d in kw_docs:
            d["retrieval_sources"] = ["keyword"]

        combined = {}
        for doc in sem_docs:
            cid = f"{doc['numero']}_{doc['progressivo']}"
            combined[cid] = {"doc": doc, "score": doc["similarity"] * semantic_weight}
        for doc in kw_docs:
            cid = f"{doc['numero']}_{doc['progressivo']}"
            norm_score = doc["score"] / (doc["score"] + 1)
            if cid in combined:
                combined[cid]["score"] += norm_score * keyword_weight
            else:
                combined[cid] = {"doc": doc, "score": norm_score * keyword_weight}

        sorted_results = sorted(combined.items(), key=lambda x: x[1]["score"], reverse=True)[:top_k]
        docs_after  = [data["doc"] for _, data in sorted_results]
        n_before    = len(docs_after)
        n_after     = len(docs_after)
        t_retrieval = time.time() - t0

    elif strategy == "semantic":
        docs_after  = semantic_search(user_prompt, top_k)
        n_before    = len(docs_after)
        n_after     = len(docs_after)
        t_retrieval = time.time() - t0

    elif strategy == "keyword":
        docs_after  = keyword_search(user_prompt, top_k)
        n_before    = len(docs_after)
        n_after     = len(docs_after)
        t_retrieval = time.time() - t0

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # ── Answer generation ────────────────────────────────────────────────────
    t2 = time.time()
    final_text = "".join(generate_final_answer(user_prompt, docs_after, chat_history))
    t_answer   = time.time() - t2

    # ── Return structured result ─────────────────────────────────────────────
    return {
        "answer":           final_text,
        "selected_docs":    docs_after,
        "tool_decision":    tool_decision,
        "selection_reason": selection_reason,
        "n_docs_before":    n_before if strategy in ("multistage",) else n_after,
        "n_docs_after":     n_after,
        "t_retrieval":      t_retrieval,
        "t_selection":      t_selection,
        "t_answer":         t_answer,
    }
