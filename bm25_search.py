from rank_bm25 import BM25Okapi
import re

# Global BM25 index and chunks
_bm25_index = None
_bm25_chunks = []

def tokenize(text: str) -> list:
    return re.findall(r'[a-zA-Z0-9]+', text.lower())

def build_bm25_index(chunks: list):
    global _bm25_index, _bm25_chunks
    _bm25_chunks = chunks
    if not chunks:
        _bm25_index = None
        print("BM25 index cleared.")
        return

    tokenized = [tokenize(c["text"]) for c in chunks]
    _bm25_index = BM25Okapi(tokenized)
    print(f"BM25 index built with {len(chunks)} chunks!")

def keyword_search(question: str, top_k: int = 3) -> list:
    if _bm25_index is None:
        print("BM25 index not built yet!")
        return []

    tokens = tokenize(question)
    scores = _bm25_index.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for idx, score in ranked:
        if score > 0:
            results.append({
                "text": _bm25_chunks[idx]["text"],
                "source": _bm25_chunks[idx]["source"],
                "page": _bm25_chunks[idx]["page"],
                "score": round(score, 4),
                "chunk_id": _bm25_chunks[idx]["chunk_id"]
            })

    print(f"BM25 found {len(results)} results for: {question}")
    return results

def get_bm25_chunks():
    return _bm25_chunks
