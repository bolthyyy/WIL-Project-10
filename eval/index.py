import json
import time
from pathlib import Path
import numpy as np

from src.pdf_utils import split_into_chunks
from src.settings import MANUAL_METADATA_FILE
from eval.embedders import Bm25Embedder, get_embedder
from eval.reranker import rerank as rerank_passages

eval_dir = Path(__file__).resolve().parent
cache_dir = eval_dir / "cache"

def load_manuals() -> list[dict]:
    return json.loads(MANUAL_METADATA_FILE.read_text(encoding="utf-8"))

def load_cached_pages(vehicle_id: str) -> list[dict]:
    cache_path = cache_dir / f"{vehicle_id}.json"

    if not cache_path.exists():
        raise FileNotFoundError(f"No cached text for {vehicle_id}")
    return json.loads(cache_path.read_text(encoding="utf-8"))

class EvalIndex:
    def __init__(self, embedder_name: str, chunk_size_words: int, overlap_words: int):
        self.embedder_name = embedder_name
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words
        self.embedder = get_embedder(embedder_name)
        self.chunks = []
        self.vectors = None

    def build(self, manuals: list[dict] | None = None) -> None:
        if manuals is None:
            manuals = load_manuals()

        texts = []

        for manual in manuals:
            pages = load_cached_pages(manual["vehicle_id"])

            for page in pages:
                if not page["text"]:
                    continue

                page_chunks = split_into_chunks(page["text"],
                                                chunk_size_words=self.chunk_size_words,
                                                overlap_words=self.overlap_words)

                for chunk_number, chunk_text in enumerate(page_chunks):
                    chunk = {
                        "text": chunk_text,
                        "metadata": {
                            "vehicle_id": manual["vehicle_id"],
                            "make": manual["make"],
                            "model": manual["model"],
                            "year": manual["year"],
                            "manual_title": manual["manual_title"],
                            "pdf_page": page["page"],
                            "chunk_number": chunk_number,
                        },
                    }
                    self.chunks.append(chunk)
                    texts.append(chunk_text)

        self.embedder.fit(texts)

        if isinstance(self.embedder, Bm25Embedder):
            return

        vectors = self.embedder.encode_document(texts)
        self.vectors = np.asarray(vectors, dtype=np.float32)


    def search(self, question: str, vehicle_id: str | None = None,
               top_k: int = 5, rerank: bool = False) -> tuple[list[dict], float, float | None]:
        #returns passages, latency and top1_score. top1_score is bm25 score of best ranked candidate befor reranking reorders the list.
        start = time.perf_counter()

        candidate_indexes = []
        for i, chunk in enumerate(self.chunks):
            if vehicle_id is None or chunk["metadata"]["vehicle_id"] == vehicle_id:
                candidate_indexes.append(i)

        if not candidate_indexes:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return [], elapsed_ms, None

        if isinstance(self.embedder, Bm25Embedder):
            scores = self.embedder.score(question)
            ranked = sorted(candidate_indexes, key= lambda i: scores[i], reverse=True)
            top1_score = float(scores[ranked[0]])
        else:
            query_vector = np.asarray(self.embedder.encode_query(question), dtype=np.float32)
            candidate_vectors = self.vectors[candidate_indexes]
            local_scores = cos_similarity(query_vector, candidate_vectors)

            #sort from lowest sim to highest then flip
            order = np.argsort(local_scores)[::-1]
            ranked = [candidate_indexes[i] for i in order]
            top1_score = float(local_scores[order[0]])

        #pull wider pool from first stage of ranking so reranking has more top_k to re-sort
        size = top_k * 4 if rerank else top_k
        top_indexes = ranked[:size]

        passages = []
        for i in top_indexes:
            passages.append(dict(self.chunks[i]))

        if rerank:
            passages = rerank_passages(question, passages, top_k=top_k)
        else: passages = passages[:top_k]

        latency_ms = (time.perf_counter() - start) * 1000

        return passages, latency_ms, top1_score

        

def cos_similarity(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vector) + 1e-8
    matrix_norms = np.linalg.norm(matrix, axis=1) + 1e-8
    return (matrix @ query_vector) / (matrix_norms * query_norm)