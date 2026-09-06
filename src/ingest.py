import argparse
import json

import chromadb
from sentence_transformers import SentenceTransformer

from .pdf_utils import extract_pdf_chunks
from .settings import (
    CHROMA_DIR,
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    MANUAL_DIR,
    MANUAL_METADATA_FILE,
)


def load_manual_metadata() -> list[dict]:
    with open(MANUAL_METADATA_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def get_chroma_client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def reset_collection(client):
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted old collection: {COLLECTION_NAME}")
    except Exception:
        pass


def main(reset: bool = False):
    manuals = load_manual_metadata()

    client = get_chroma_client()

    if reset:
        reset_collection(client)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    total_chunks = 0

    for manual in manuals:
        pdf_path = MANUAL_DIR / manual["file"]

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"Missing PDF: {pdf_path}"
            )

        print(
            f"\nProcessing "
            f"{manual['make']} {manual['model']}..."
        )

        chunks = extract_pdf_chunks(
            pdf_path=pdf_path,
            chunk_size_words=CHUNK_SIZE_WORDS,
            overlap_words=CHUNK_OVERLAP_WORDS,
        )

        if not chunks:
            print(
                f"WARNING: No text extracted from "
                f"{manual['file']}"
            )
            continue

        documents = [chunk["text"] for chunk in chunks]

        print(f"Creating embeddings for {len(documents)} chunks...")

        embeddings = embedding_model.encode_document(
            documents,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        ids = []
        metadatas = []

        for chunk in chunks:
            chunk_id = (
                f"{manual['vehicle_id']}"
                f"_page_{chunk['pdf_page']}"
                f"_chunk_{chunk['chunk_number']}"
            )

            ids.append(chunk_id)

            metadatas.append(
                {
                    "vehicle_id": manual["vehicle_id"],
                    "make": manual["make"],
                    "model": manual["model"],
                    "year": manual["year"],
                    "manual_title": manual["manual_title"],
                    "source_file": manual["file"],
                    "source_url": manual["source_url"],
                    "pdf_page": chunk["pdf_page"],
                    "chunk_number": chunk["chunk_number"],
                }
            )

        # Add to Chroma in manageable batches.
        batch_size = 100

        for start in range(0, len(documents), batch_size):
            end = start + batch_size

            collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                embeddings=embeddings[start:end].tolist(),
                metadatas=metadatas[start:end],
            )

        total_chunks += len(documents)

        print(
            f"Indexed {len(documents)} chunks from "
            f"{manual['manual_title']}."
        )

    print("\nIndexing complete.")
    print(f"Total chunks indexed: {total_chunks}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing vector database collection first.",
    )

    arguments = parser.parse_args()

    main(reset=arguments.reset)