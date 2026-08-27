"""
FALLBACK retriever using TF-IDF (scikit-learn) instead of HF embeddings.

Use this ONLY if you don't have internet access to huggingface.co.
On your own machine, prefer kb_index.py (sentence-transformers + FAISS) —
dense embeddings handle paraphrasing/semantic similarity much better than
TF-IDF's exact-term matching, which matters a lot for support tickets where
customers rarely use the same wording as the docs.

This module mirrors the same chunk_all_kb_files() output shape so you can
swap between the two without changing the rest of the pipeline.
"""
import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from kb_index import chunk_all_kb_files, Chunk, INDEX_DIR

INDEX_DIR.mkdir(exist_ok=True)


def build_tfidf_index(chunks: list[Chunk]):
    texts = [c.text for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def save_tfidf_index(vectorizer, matrix, chunks: list[Chunk]):
    with open(INDEX_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(INDEX_DIR / "tfidf_matrix.pkl", "wb") as f:
        pickle.dump(matrix, f)
    with open(INDEX_DIR / "kb_chunks.json", "w", encoding="utf-8") as f:
        json.dump([c.__dict__ for c in chunks], f, indent=2)


def load_tfidf_index():
    with open(INDEX_DIR / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(INDEX_DIR / "tfidf_matrix.pkl", "rb") as f:
        matrix = pickle.load(f)
    with open(INDEX_DIR / "kb_chunks.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    chunks = [Chunk(**c) for c in raw]
    return vectorizer, matrix, chunks


def search_tfidf(query: str, vectorizer, matrix, chunks: list[Chunk], top_k: int = 3):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    return [(chunks[i], float(sims[i])) for i in top_idx]


if __name__ == "__main__":
    print("Chunking KB files...")
    chunks = chunk_all_kb_files()
    print(f"Produced {len(chunks)} chunks")

    print("Building TF-IDF index...")
    vectorizer, matrix = build_tfidf_index(chunks)
    save_tfidf_index(vectorizer, matrix, chunks)
    print(f"Saved TF-IDF index with {matrix.shape[0]} chunks")

    # quick smoke test
    query = "connection timeout error connecting to data source"
    results = search_tfidf(query, vectorizer, matrix, chunks, top_k=2)
    print(f"\nTest query: {query!r}")
    for chunk, score in results:
        print(f"  score={score:.3f}  [{chunk.heading}]  {chunk.text[:100]}...")
