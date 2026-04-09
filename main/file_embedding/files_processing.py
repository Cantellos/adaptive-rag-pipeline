"""
files_processing.py
===================
Ingestion pipeline: read source documents → extract text (+ OCR) → chunk
→ embed → store in SQL Server DocumentChunks table → rebuild BM25 index.

Two input modes (set in config.yaml → input.mode):

  "database"  (default)
      Documents are fetched from a SQL Server table. The connection and table
      names are read from .env / config.yaml. Suitable when documents are
      managed inside a workflow / BPM system.

  "folder"
      Documents are read from a local directory (config.yaml → folder_input.path).
      Supported extensions: .pdf, .docx (configurable in config.yaml).
      An optional JSON metadata manifest can map filenames to metadata fields
      (see config.yaml for the expected format). Metadata defaults are used
      for any file not listed in the manifest.
"""

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
import extract_text as extract_text_module
import db_connection
import embedding
import json
import easyocr
import os
import sys
from pathlib import Path
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import bm25s
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config_loader import cfg


# ---------------------------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------------------------

class ChunkingStrategy(Enum):
    FIXED_SIZE = "fixed"
    RECURSIVE  = "recursive"


@dataclass
class ChunkingConfig:
    strategy:             ChunkingStrategy
    chunk_size:           int
    chunk_overlap:        int
    recursive_separators: Optional[List[str]] = None

    def __post_init__(self):
        if self.recursive_separators is None:
            self.recursive_separators = [
                "\nTechnical Analysis\n",   # common section header in structured docs
                "\n[OCR embedded image]:",  # OCR caption injected by extract_text
                "\n\n\n",
                "\n\n",
                "\n",
                ". ",
                ", ",
                " ",
                "",
            ]


def create_text_splitter(config: ChunkingConfig):
    if config.strategy == ChunkingStrategy.FIXED_SIZE:
        print(f"Using Fixed-size chunking: size={config.chunk_size}, overlap={config.chunk_overlap}")
        return TokenTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
    elif config.strategy == ChunkingStrategy.RECURSIVE:
        print(f"Using Recursive chunking: size={config.chunk_size}, overlap={config.chunk_overlap}")
        return RecursiveCharacterTextSplitter(
            separators=config.recursive_separators,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
    else:
        raise ValueError(f"Unsupported chunking strategy: {config.strategy}")


# ---------------------------------------------------------------------------
# Helper: build a ChunkingConfig from config.yaml
# ---------------------------------------------------------------------------

def config_from_yaml() -> ChunkingConfig:
    """Build a ChunkingConfig from the values in config.yaml."""
    strategy_str = (cfg.chunking.get("strategy") or "recursive").lower()
    strategy = ChunkingStrategy.FIXED_SIZE if strategy_str == "fixed" else ChunkingStrategy.RECURSIVE
    return ChunkingConfig(
        strategy=strategy,
        chunk_size=int(cfg.chunking.get("chunk_size") or 1024),
        chunk_overlap=int(cfg.chunking.get("chunk_overlap") or 150),
    )


# ---------------------------------------------------------------------------
# Metadata helpers for folder mode
# ---------------------------------------------------------------------------

def _load_metadata_manifest(manifest_path: str) -> dict:
    """Load a JSON metadata manifest mapping filename → metadata dict."""
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _filename_to_id(filename: str) -> str:
    """Derive a numeric-ish document ID from a filename (stem, digits only)."""
    import re
    stem = Path(filename).stem
    digits = re.sub(r"\D", "", stem)
    return digits if digits else stem[:20]


def _default_metadata(filename: str) -> dict:
    """Return sensible metadata defaults for a file with no manifest entry."""
    return {
        "numero":    _filename_to_id(filename),
        "cliente":   "N/A",
        "titolo":    Path(filename).stem,
        "autore":    "N/A",
        "documento": filename,
        "url_doc":   "",
    }


# ---------------------------------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------------------------------

def process_files(config: ChunkingConfig, limit: Optional[int] = None):
    """
    Run the full ingestion pipeline.

    The input mode (database / folder) is read from config.yaml.
    If limit is given, at most that many documents are processed.
    """
    input_mode = (cfg.input.get("mode") or "database").lower()

    if input_mode == "folder":
        _process_from_folder(config, limit)
    else:
        _process_from_database(config, limit)


# ---------------------------------------------------------------------------
# Input mode: folder
# ---------------------------------------------------------------------------

def _process_from_folder(config: ChunkingConfig, limit: Optional[int] = None):
    """Ingest documents from a local directory."""
    folder_path = cfg.folder_input.get("path") or "./documents"
    manifest_path = cfg.folder_input.get("metadata_manifest") or ""
    supported_ext = cfg.folder_input.get("supported_extensions") or [".pdf", ".docx"]

    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(
            f"Document folder not found: {folder_path}\n"
            "Set folder_input.path in config.yaml to the correct path."
        )

    # Collect files
    all_files = [
        f for f in sorted(folder.iterdir())
        if f.is_file() and f.suffix.lower() in supported_ext
    ]
    if limit:
        all_files = all_files[:limit]

    manifest = _load_metadata_manifest(manifest_path)
    reader = easyocr.Reader(["it", "en"], gpu=False)
    conn = db_connection.get_connection()
    cursor = conn.cursor()
    splitter = create_text_splitter(config)

    corpus = []
    total = len(all_files)
    print(f"\n{'='*60}")
    print(f"Folder mode: processing {total} files from '{folder_path}'")
    print(f"Strategy: {config.strategy.value}")
    print(f"{'='*60}\n")

    for idx, filepath in enumerate(all_files, 1):
        filename = filepath.name
        meta = manifest.get(filename) or _default_metadata(filename)
        numero    = str(meta.get("numero",    _filename_to_id(filename)))
        cliente   = str(meta.get("cliente",   "N/A"))
        titolo    = str(meta.get("titolo",    Path(filename).stem))
        autore    = str(meta.get("autore",    "N/A"))
        documento = str(meta.get("documento", filename))
        url_doc   = str(meta.get("url_doc",   ""))
        extension = filepath.suffix.lower()

        print(f"[{idx}/{total}] Processing: {filename}")

        # Read file bytes
        with open(filepath, "rb") as fh:
            file_data = fh.read()

        # Extract text
        text = extract_text_module.extract_text_from_varbinary(file_data, extension, numero, reader)
        if not text.strip():
            print(f"  ⚠ No text extracted, skipping\n")
            continue

        # Chunk
        try:
            chunks = splitter.split_text(text)
            print(f"  ✓ {len(chunks)} chunks generated")
        except Exception as exc:
            print(f"  ✗ Chunking error: {exc}\n")
            continue

        # Embed and store
        chunk_records = []
        for i, chunk in enumerate(chunks):
            emb = embedding.get_embedding(chunk)
            emb_json = json.dumps(emb)
            chunk_records.append((numero, i, cliente, titolo, autore, documento, url_doc, chunk, emb_json))
            corpus.append(chunk)

        if chunk_records:
            chunks_table = cfg.database.get("chunks_table") or "DocumentChunks"
            cursor.executemany(
                f"INSERT INTO {chunks_table} "
                "(NumRI, Progressivo, Cliente, Titolo, Autore, Documento, Url_doc, Content, Embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, CAST(CAST(? AS VARCHAR(MAX)) AS VECTOR(1536)))",
                chunk_records,
            )

        conn.commit()
        print(f"  ✓ Embeddings saved for: {documento}\n")

    print(f"{'='*60}")
    print(f"Done: {total} files processed")
    print(f"{'='*60}\n")

    build_bm25_index(conn, cursor)
    cursor.close()
    conn.close()


# ---------------------------------------------------------------------------
# Input mode: database
# ---------------------------------------------------------------------------

def _process_from_database(config: ChunkingConfig, limit: Optional[int] = None):
    """Ingest documents from SQL Server source tables."""
    reader = easyocr.Reader(["it", "en"], gpu=False)
    conn = db_connection.get_connection()
    cursor = conn.cursor()
    splitter = create_text_splitter(config)

    source_vars_table  = cfg.database.get("source_variables_table") or "VAR_RICSW"
    source_files_table = cfg.database.get("source_files_table")      or "DocumentFiles"
    chunks_table       = cfg.database.get("chunks_table")            or "DocumentChunks"

    limit_clause = f"TOP {limit}" if limit else ""
    query = f"""
    SELECT {limit_clause}
        v.InstanceID,
        MAX(CASE WHEN v.VariableName = 'NUMERO'  THEN v.StringValue END) AS numero,
        MAX(CASE WHEN v.VariableName = 'CLIENTE' THEN v.StringValue END) AS cliente,
        MAX(CASE WHEN v.VariableName = 'TITOLO'  THEN v.StringValue END) AS titolo,
        MAX(CASE WHEN v.VariableName = 'AUTORE'  THEN v.StringValue END) AS autore,
        MAX(CASE WHEN v.VariableName = 'DOC'     THEN v.StringValue END) AS documento,
        MAX(CASE WHEN v.VariableName = 'URL_DOC' THEN v.StringValue END) AS url_doc,
        MAX(f.FileData)  AS FileData,
        MAX(f.Extension) AS Extension
    FROM {source_vars_table} v
    JOIN {source_files_table} f ON v.InstanceID = f.InstanceID
    WHERE v.InstanceID IN (
        SELECT v2.InstanceID
        FROM {source_vars_table} v2
        WHERE v2.VariableName = 'ELABORATO'
          AND v2.BooleanValue = 0
    )
    GROUP BY v.InstanceID
    ORDER BY numero DESC;
    """

    cursor.execute(query)
    rows = cursor.fetchall()
    corpus = []
    total_files = len(rows)

    print(f"\n{'='*60}")
    print(f"Database mode: processing {total_files} files")
    print(f"Strategy: {config.strategy.value}")
    print(f"{'='*60}\n")

    for idx, row in enumerate(rows, 1):
        chunk_records = []
        instance_id, numero, cliente, titolo, autore, documento, url_doc, file_data, extension = row

        print(f"[{idx}/{total_files}] Processing: {numero}{extension}")

        text = extract_text_module.extract_text_from_varbinary(file_data, extension, numero, reader)
        if not text.strip():
            print(f"  ⚠ No text extracted, skipping\n")
            continue

        try:
            chunks = splitter.split_text(text)
            print(f"  ✓ {len(chunks)} chunks generated")
        except Exception as exc:
            print(f"  ✗ Chunking error: {exc}\n")
            continue

        for i, chunk in enumerate(chunks):
            emb = embedding.get_embedding(chunk)
            emb_json = json.dumps(emb)
            chunk_records.append((numero, i, cliente, titolo, autore, documento, url_doc, chunk, emb_json))
            corpus.append(chunk)

        # Mark document as processed
        cursor.execute(
            f"UPDATE {source_vars_table} SET BooleanValue = 1 "
            "WHERE VariableName = 'ELABORATO' AND InstanceID = ?",
            (instance_id,),
        )

        if chunk_records:
            cursor.executemany(
                f"INSERT INTO {chunks_table} "
                "(NumRI, Progressivo, Cliente, Titolo, Autore, Documento, Url_doc, Content, Embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, CAST(CAST(? AS VARCHAR(MAX)) AS VECTOR(1536)))",
                chunk_records,
            )

        conn.commit()
        print(f"  ✓ Embeddings saved for: {documento}\n")

    print(f"{'='*60}")
    print(f"Done: {total_files} files processed")
    print(f"{'='*60}\n")

    build_bm25_index(conn, cursor)
    cursor.close()
    conn.close()


# ---------------------------------------------------------------------------
# BM25 index rebuild
# ---------------------------------------------------------------------------

def build_bm25_index(conn, cursor):
    """Rebuild the BM25 keyword-search index from all chunks in DocumentChunks."""
    chunks_table = cfg.database.get("chunks_table") or "DocumentChunks"
    language     = cfg.domain.get("language") or "english"

    print(f"\n{'='*60}")
    print("Rebuilding BM25 index...")
    print(f"{'='*60}\n")

    cursor.execute(f"SELECT id, Content FROM {chunks_table} ORDER BY id ASC")
    rows = cursor.fetchall()
    db_ids     = [row[0] for row in rows]
    all_chunks = [row[1] for row in rows]

    stemmer     = SnowballStemmer(language)
    stop_words  = stopwords.words(language)
    print("Tokenising...")
    all_tokens = bm25s.tokenize(
        all_chunks,
        stopwords=stop_words,
        stemmer=lambda tokens: [stemmer.stem(t.lower()) for t in tokens],
    )

    print("Building BM25 index...")
    retriever = bm25s.BM25()
    if all_chunks:
        retriever.index(all_tokens)
        index_path = os.path.join("reverse_index", "bm25_index")
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        retriever.save(index_path)
        print(f"✓ Index saved to: {index_path}")

    print("Updating index mapping in database...")
    for pos, db_id in enumerate(db_ids):
        cursor.execute(
            f"UPDATE {chunks_table} SET Bm25_index = ? WHERE id = ?",
            (pos, db_id),
        )

    conn.commit()
    print(f"✓ BM25 index built for {len(all_chunks)} chunks\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run the ingestion pipeline using settings from config.yaml."""
    config = config_from_yaml()
    limit = 350  # adjust as needed
    process_files(config, limit)


if __name__ == "__main__":
    main()
