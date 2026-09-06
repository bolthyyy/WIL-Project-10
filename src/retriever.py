import chromadb
from sentence_transformers import SentenceTransformer

from .settings import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
)


_embedding_model = None


def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    return _embedding_model


def get_collection():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    return client.get_collection(
        name=COLLECTION_NAME
    )


def retrieve(
    question: str,
    vehicle_id: str | None = None,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Retrieve the most relevant manual chunks.

    If vehicle_id is supplied, retrieval is restricted to that
    vehicle's manual.

    If vehicle_id is None, all manuals are searched.
    """
    embedding_model = get_embedding_model()
    collection = get_collection()

    query_embedding = embedding_model.encode_query(
        question,
        normalize_embeddings=True,
    ).tolist()

    query_arguments = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    if vehicle_id:
        query_arguments["where"] = {
            "vehicle_id": vehicle_id
        }

    results = collection.query(**query_arguments)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    passages = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        passages.append(
            {
                "text": document,
                "metadata": metadata,
                "distance": distance,
            }
        )

    return passages