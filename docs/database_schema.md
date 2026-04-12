# Database Schema (SQL Server 2022)

This document describes the SQL Server tables used by the pipeline in `database` input mode.

---

## DocumentChunks

Stores processed text chunks with embeddings. **Required in both `folder` and `database` input modes.**

```sql
CREATE TABLE DocumentChunks (
    Id              INT IDENTITY(1,1) PRIMARY KEY,
    DocumentId      NVARCHAR(64)   NOT NULL,   -- unique document identifier
    Title           NVARCHAR(512),             -- document title
    Author          NVARCHAR(256),             -- author name
    Client          NVARCHAR(256),             -- client / category tag
    ChunkIndex      INT            NOT NULL,   -- zero-based chunk position within document
    ChunkText       NVARCHAR(MAX)  NOT NULL,   -- raw chunk text
    Embedding1      VECTOR(1536),              -- primary embedding (e.g. text-embedding-ada-002)
    Embedding2      VECTOR(3072)               -- secondary embedding (optional, for ablation)
);
```

> **Note:** `VECTOR` is a preview type introduced in SQL Server 2022 (16.x). Enable it with:
> ```sql
> EXEC sp_configure 'show advanced options', 1; RECONFIGURE;
> EXEC sp_configure 'vector search', 1; RECONFIGURE;
> ```

---

## VAR_DOCUMENTS *(database mode only)*

Source table containing document metadata. Read by the ingestion pipeline to enumerate documents.

```sql
CREATE TABLE VAR_DOCUMENTS (
    NumeroRI        INT            PRIMARY KEY,  -- numeric document ID
    Titolo          NVARCHAR(512),               -- document title
    Autore          NVARCHAR(256),               -- author
    Cliente         NVARCHAR(256),               -- client / category
    DataModifica    DATETIME                     -- last modified date
);
```

---

## DocumentFiles *(database mode only)*

Stores the raw binary content of each document. Joined with `VAR_DOCUMENTS` during ingestion.

```sql
CREATE TABLE DocumentFiles (
    NumeroRI        INT            PRIMARY KEY REFERENCES VAR_DOCUMENTS(NumeroRI),
    Estensione      NVARCHAR(16),               -- file extension: ".pdf", ".docx", ...
    Documento       VARBINARY(MAX)              -- raw file bytes
);
```

---

## Configuration

Table names are configurable in `config.yaml`:

```yaml
database:
  chunks_table:           "DocumentChunks"
  source_variables_table: "VAR_DOCUMENTS"
  source_files_table:     "DocumentFiles"
```
