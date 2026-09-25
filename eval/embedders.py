import numpy as np
from rank_bm25 import BM25Okapi

class Bm25Embedder:
#offline retrieval using bm25 instead of vector, its queired with score() insetead of producing fixed sized vectors.
    name = "bm25"
    def __init__(self):
        self._bm25 = None

    def fit(self, documents: list[str]) -> None:
        tokenized_docs = []
        for document in documents:
            tokenized_docs.append(document.lower().split())
        self._bm25 = BM25Okapi(tokenized_docs)

    def score (self, question: str) -> np.ndarray:
        if self._bm25 is None:
            raise RuntimeError("BM25Embedder must be fit before use")

        tokenized_question = question.lower().split()
        return np.array(self._bm25.get_scores(tokenized_question))


class DenseEmbedder:
#wraps real sentence transformer model
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, _documents: list[str]) -> None:
        pass

    def encode_document(self, texts: list[str], **kwargs) -> np.ndarray:
        kwargs.setdefault("normalize_embeddings", True)
        kwargs.setdefault("batch_size", 32)
        return self.model.encode_document(texts, **kwargs)

    def encode_query(self, text: str, **kwargs) -> np.ndarray:
        kwargs.setdefault("normalize_embeddings", True)
        return self.model.encode_query(text, **kwargs)

#short names used from the config dicts in the cli

Dense_model_short = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}

def get_embedder(name: str):
#name = tfidf, bm25, or key in dense model short (or can use full model name)

    if name == "bm25":
        return Bm25Embedder()

    model_name = Dense_model_short.get(name, name)
    return DenseEmbedder(model_name)