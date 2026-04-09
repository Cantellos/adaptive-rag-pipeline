import json
import os
import sys
import bm25s
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import nltk
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from file_embedding.db_connection import get_connection
from file_embedding.embedding import get_embedding
nltk.download('stopwords')

def semantic_search(prompt, top_n=10):
    prompt_embedding = get_embedding(prompt)
    prompt_embedding_json = json.dumps(prompt_embedding)

    # Query per calcolo similarità coseno direttamente nel DB
    query = f"""
            DECLARE @prompt VECTOR(1536) = CAST('{prompt_embedding_json}' AS VECTOR(1536));

            SELECT TOP (?)
                ID, NumRI, Progressivo, Cliente, Titolo, 
                Autore, Documento, Url_doc, Content, Embedding,
                1 - VECTOR_DISTANCE('cosine', Embedding, @prompt) AS Similarity
            FROM DocumentChunks
            ORDER BY Similarity DESC;
        """

    cursor = get_connection().cursor()
    cursor.execute(query, (top_n,))
    rows = cursor.fetchall()

    docs = []
    for row in rows:
        doc_id, numero, progressivo, cliente, titolo, autore, documento, url_doc, content, embedding, similarity = row

        docs.append({
            "id": doc_id,
            "numero": numero,
            "progressivo": progressivo,
            "cliente": cliente,
            "titolo": titolo,
            "autore": autore,
            "documento": documento,
            "url_doc": url_doc,
            "content": content,
            "embedding": embedding,
            "similarity": similarity
        })
    return docs


def keyword_search(prompt, top_n=10, language='italian'):

    # Setup stemmer e stopwords
    stemmer = SnowballStemmer(language)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Stopwords non trovate, scarico il pacchetto...")
        nltk.download('stopwords')
    stop_words = stopwords.words(language)

    # Tokenizzazione query
    query_tokens = bm25s.tokenize(
        prompt,
        stopwords=stop_words,
        stemmer=lambda tokens: [stemmer.stem(t.lower()) for t in tokens]
    )

    # Carica indice BM25
    index_folder = "reverse_index"
    index_path = os.path.join(index_folder, "bm25_index")
    retriever = bm25s.BM25.load(index_path, load_corpus=True)

    # Numero di documenti 
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM DocumentChunks")
    num_docs = cursor.fetchone()[0]
    cursor.close()

    k = min(top_n, num_docs)

    # Recupera risultati BM25
    results, scores = retriever.retrieve(query_tokens, k=k)

    # Estrai chunk e metadati dal DB
    docs = []
    cursor = conn.cursor()
    for idx, score in zip(results.flatten(), scores.flatten()):
        if score <= 0:
            continue

        cursor.execute("""
            SELECT 
                id, NumRI, progressivo, cliente, titolo, autore,
                documento, url_doc, content, embedding
            FROM DocumentChunks 
            WHERE bm25_index = ?
        """, (int(idx),))
        row = cursor.fetchone()
        if not row:
            continue

        (doc_id, numero, progressivo, cliente, titolo, autore,
            documento, url_doc, content, embedding) = row

        docs.append({
            "id": doc_id,
            "numero": numero,
            "progressivo": progressivo,
            "cliente": cliente,
            "titolo": titolo,
            "autore": autore,
            "documento": documento,
            "url_doc": url_doc,
            "content": content,
            "embedding": embedding,
            "score": float(score)
        })        
        
    cursor.close()
    print(f"\nRicerca BM25 completata: {len(docs)} risultati trovati")
    return docs



