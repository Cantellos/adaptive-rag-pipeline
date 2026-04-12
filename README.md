# Adaptive Multi-Stage RAG System

A **domain-agnostic, multi-stage Retrieval-Augmented Generation (RAG) pipeline** for intelligent document Q&A and decision support.

The system combines semantic vector search, BM25 keyword search, and LLM-based filtering to adaptively retrieve and present information from a custom document knowledge base. An interactive Streamlit chat interface lets users query the knowledge base in natural language with cited, source-grounded answers.

Developed as a Master's thesis project at the University of Ferrara (Università degli Studi di Ferrara).

---

## Architecture

The pipeline consists of four sequential stages, all configurable via `config.yaml`:

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  Stage 1 — Tool Selection (LLM)     │  decide_tools()
│  Decides: semantic / keyword / both │
│  or "out of scope" (no search)      │
└────────────────┬────────────────────┘
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
 Semantic     Keyword      (Both)
 Search       Search
 (FAISS-      (BM25 +
  like via    inverted
  SQL Server  index)
  VECTOR)
    └────────────┼────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Stage 2 — Document Selection (LLM) │  select_documents()
│  Filters and re-ranks retrieved     │
│  chunks by relevance; deduplicates  │
│  and removes template-like content  │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Stage 3 — Answer Generation (LLM)  │  generate_final_answer()
│  Streams a cited, markdown-formatted│
│  answer with a reference list       │
└─────────────────────────────────────┘
```

### Component map

| File | Role |
|---|---|
| `config.yaml` | All domain/app configuration (no secrets) |
| `.env` | Secrets: API keys, DB credentials |
| `main/config_loader.py` | Loads `config.yaml` as a Python object |
| `main/llm/llm.py` | Pipeline stages 1–3, chatbot flow, history management |
| `main/llm/search.py` | Semantic search (SQL Server VECTOR) + BM25 keyword search |
| `main/llm/app.py` | Streamlit chat application |
| `main/llm/ui.py` | Streamlit UI helpers and user auth |
| `main/file_embedding/files_processing.py` | Ingestion pipeline (folder or DB → chunks → embeddings → BM25 index) |
| `main/file_embedding/extract_text.py` | Text extraction + OCR from PDF/DOCX |
| `main/file_embedding/embedding.py` | Calls the embedding API |
| `main/file_embedding/db_connection.py` | SQL Server connection |
| `main/evaluation/evaluation_pipeline.py` | Ablation study + retrieval/generation metrics |

---

## Requirements

- **Python** 3.9+
- **SQL Server 2022+** with the native `VECTOR` type (used for cosine similarity search)
- **OpenAI API key** (standard OpenAI *or* an Azure OpenAI deployment)
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

> **Note on SQL Server:** The semantic search relies on `VECTOR_DISTANCE('cosine', ...)` which requires SQL Server 2022 (16.x) or later with the VECTOR preview feature enabled.  
> If you cannot use SQL Server, the BM25 keyword search works standalone via the `bm25s` library and a local index file.

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-username/adaptive-rag-system.git
cd adaptive-rag-system

# With uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### 2. Configure secrets

```bash
cp .env.example .env
# Edit .env and fill in your API key, DB credentials, etc.
```

### 3. Configure your domain

Open `config.yaml` and set the values for your knowledge base:

```yaml
domain:
  name: "My Knowledge Base"          # Name of the system/domain
  document_type: "Document"          # What individual documents are called
  document_url_template: ""          # Optional link template: "https://my-intranet/docs/{id}"
  language: "english"                # Language for BM25 stemming

input:
  mode: "folder"                     # "folder" or "database"

folder_input:
  path: "./documents"                # Put your .pdf and .docx files here
```

### 4. Ingest documents

Place your `.pdf` and `.docx` files in the `documents/` folder (or configure a database source), then run:

```bash
python main/file_embedding/files_processing.py
```

This will:
1. Extract text (+ OCR for embedded images) from each file
2. Split each document into chunks
3. Compute and store embeddings in SQL Server
4. Build the BM25 reverse index

### 5. Launch the chat interface

```bash
streamlit run main/llm/app.py
```

Open your browser at `http://localhost:8501`, register an account and start querying your knowledge base.

---

## Input modes

### Folder mode (`input.mode: "folder"`)

Documents are read from a local directory. This is the easiest way to get started.

**Supported formats:** `.pdf`, `.docx` (configurable in `config.yaml`).

**Optional metadata manifest:** If you want to associate rich metadata (title, author, client, URL) with each file, create a JSON manifest:

```json
{
  "contract_2024.pdf": {
    "id": "2024-001",
    "title": "Service Contract 2024",
    "author": "Legal Team",
    "client": "Acme Corp",
    "url_doc": "https://my-intranet/docs/2024-001"
  }
}
```

Set the path in `config.yaml`:

```yaml
folder_input:
  metadata_manifest: "./documents/metadata.json"
```

Without a manifest, filenames are used as titles and IDs.

### Database mode (`input.mode: "database"`)

Documents are fetched from SQL Server source tables. This mode is intended for systems where documents are already managed inside a workflow engine.

Configure the table names in `config.yaml`:

```yaml
database:
  source_variables_table: "VAR_DOCUMENTS"   # table with document metadata
  source_files_table:     "DocumentFiles"   # table with binary file data
  chunks_table:           "DocumentChunks"  # output table (always required)
```

See `docs/database_schema.md` for the expected table structure.

---

## Adapting to a new domain

The system is fully domain-agnostic. To adapt it to any knowledge base:

**Step 1 — Edit `config.yaml`:**

```yaml
app:
  title: "Legal Document Assistant"
  icon: "⚖️"
  welcome_message: "Ask me anything about our legal documents."

domain:
  name: "Legal Knowledge Base"
  description: "Repository of contracts, NDAs, and regulatory filings."
  document_type: "Contract"
  document_url_template: "https://legal.mycompany.com/contracts/{id}"
  language: "english"
  out_of_scope_description: "topics unrelated to contracts or legal documents"
```

**Step 2 — Optionally override LLM prompts:**

For fine-grained control, override individual pipeline prompts in `config.yaml`:

```yaml
prompts:
  decide_tools: |
    You are a routing assistant for a legal knowledge base.
    Decide which search tool to use for this query about {domain_name}.
    ...
  generate_answer: null   # null = use the built-in default
```

Available tokens in prompt templates: `{domain_name}`, `{document_type}`, `{domain_description}`, `{out_of_scope}`.

**Step 3 — Ingest your documents** (see Quick Start above).

---

## Evaluation

The evaluation pipeline (`main/evaluation/evaluation_pipeline.py`) implements a sequential ablation study comparing:

- Search strategies: `multistage` (adaptive), `hybrid` (fixed weight), `semantic`, `keyword`
- Chunking strategies: recursive custom, fixed-size, recursive standard
- Chunk sizes and embedding models

**Metrics:**
- Retrieval: span-based Precision@k, Recall@k, LLM-as-judge chunk relevance
- Generation: faithfulness (atomic claim verification), answer relevancy

**To run the evaluation:**

1. Create a gold dataset in the format of `main/evaluation/gold_dataset_example.json`
2. Save it as `main/evaluation/gold_dataset.json`
3. Run:

```bash
# Full ablation study
python main/evaluation/evaluation_pipeline.py

# Only non-re-indexing dimensions (A, B, C)
python main/evaluation/evaluation_pipeline.py -d A B C

# Quick smoke test
python main/evaluation/evaluation_pipeline.py --smoke-test
```

Results are saved as JSON files in `evaluation_results/`.

---

## Project structure

```
.
├── config.yaml                         # Domain & app configuration
├── .env.example                        # Environment variables template
├── pyproject.toml                      # Python project metadata & dependencies
├── documents/                          # Put your source documents here (folder mode)
├── reverse_index/
│   └── bm25_index/                     # Auto-generated BM25 index (git-ignored)
└── main/
    ├── config_loader.py                # Loads config.yaml
    ├── llm/
    │   ├── app.py                      # Streamlit entry point
    │   ├── ui.py                       # UI helpers + auth
    │   ├── llm.py                      # RAG pipeline (stages 1–3)
    │   └── search.py                   # Semantic + keyword retrieval
    ├── file_embedding/
    │   ├── files_processing.py         # Ingestion pipeline
    │   ├── extract_text.py             # Text + OCR extraction
    │   ├── embedding.py                # Embedding API client
    │   └── db_connection.py            # SQL Server connection
    └── evaluation/
        ├── evaluation_pipeline.py      # Ablation study + metrics
        └── gold_dataset_example.json   # Example evaluation dataset
```

---

## OpenAI provider configuration

The system supports both **standard OpenAI** and **Azure OpenAI** (and any compatible endpoint).

**Standard OpenAI** (default):

```env
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o
# Leave LLM_URL and LLM_VERSION empty
```

**Azure OpenAI:**

```env
OPENAI_API_KEY=your_azure_key
LLM_MODEL=gpt-4o          # Your deployment name
LLM_URL=https://your-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview
LLM_VERSION=2025-01-01-preview
```

**Local / compatible endpoint** (e.g. LM Studio, Ollama with OpenAI-compatible API):

```env
OPENAI_API_KEY=not-used
LLM_MODEL=mistral
LLM_BASE_URL=http://localhost:1234/v1
```

---

## Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{cantelli2026adaptive,
  author  = {Cantelli, Andrea},
  title   = {Adaptive Multi-Stage RAG Architecture for Decision Support},
  school  = {Università degli Studi di Ferrara},
  year    = {2026}
}
```

---

## License

This project is released under the [GPL-3.0 License](LICENSE).
