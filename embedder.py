import hashlib
import math
import re

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

VECTOR_SIZE = 384
model = SentenceTransformer('all-MiniLM-L6-v2') if SentenceTransformer else None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _hash_embedding(text: str) -> list[float]:
    vector = [0.0] * VECTOR_SIZE
    for token in _tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % VECTOR_SIZE
        sign = 1 if digest[4] % 2 == 0 else -1
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]

def embed_texts(texts: list) -> list:
    if model is None:
        print("sentence-transformers unavailable; using hashed fallback embeddings.")
        return [_hash_embedding(text) for text in texts]

    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()

def embed_query(question: str) -> list:
    if model is None:
        return _hash_embedding(question)

    embedding = model.encode(question)
    return embedding.tolist()
