-- ============================================================
-- create_tables.sql
-- Creates the SQL Server tables required by the RAG pipeline.
-- Run this once against your target database before ingestion.
--
-- Requirements: SQL Server 2022 (16.x) with VECTOR preview enabled.
-- ============================================================

-- Enable VECTOR support (run once per server, requires sysadmin)
-- EXEC sp_configure 'show advanced options', 1; RECONFIGURE;
-- EXEC sp_configure 'vector search', 1; RECONFIGURE;

-- ── DocumentChunks ──────────────────────────────────────────
-- Stores processed text chunks with embeddings.
-- Required in BOTH folder and database input modes.
CREATE TABLE DocumentChunks (
    Id              INT IDENTITY(1,1) PRIMARY KEY,
    DocumentId      NVARCHAR(64)   NOT NULL,
    Title           NVARCHAR(512)  NULL,
    Author          NVARCHAR(256)  NULL,
    Client          NVARCHAR(256)  NULL,
    ChunkIndex      INT            NOT NULL,
    ChunkText       NVARCHAR(MAX)  NOT NULL,
    Embedding1      VECTOR(1536)   NULL,   -- primary embedding (e.g. text-embedding-ada-002)
    Embedding2      VECTOR(3072)   NULL    -- secondary embedding (optional, for ablation)
);

-- ── VAR_DOCUMENTS ────────────────────────────────────────────
-- Source table with document metadata.
-- Only required when input.mode = "database".
CREATE TABLE VAR_DOCUMENTS (
    NumeroRI        INT            NOT NULL PRIMARY KEY,
    Titolo          NVARCHAR(512)  NULL,
    Autore          NVARCHAR(256)  NULL,
    Cliente         NVARCHAR(256)  NULL,
    DataModifica    DATETIME       NULL
);

-- ── DocumentFiles ────────────────────────────────────────────
-- Stores raw binary file content.
-- Only required when input.mode = "database".
CREATE TABLE DocumentFiles (
    NumeroRI        INT            NOT NULL
                        PRIMARY KEY
                        REFERENCES VAR_DOCUMENTS(NumeroRI),
    Estensione      NVARCHAR(16)   NULL,   -- e.g. ".pdf", ".docx"
    Documento       VARBINARY(MAX) NULL
);
