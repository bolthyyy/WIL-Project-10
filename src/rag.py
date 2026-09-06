from .generator import generate_answer
from .retriever import retrieve


def ask(
    question: str,
    vehicle_id: str | None = None,
    vehicle_name: str | None = None,
    top_k: int = 5,
) -> dict:
    passages = retrieve(
        question=question,
        vehicle_id=vehicle_id,
        top_k=top_k,
    )

    if not passages:
        return {
            "answer": (
                "No relevant owner's-manual information "
                "was retrieved."
            ),
            "sources": [],
        }

    answer = generate_answer(
        question=question,
        passages=passages,
        vehicle_name=vehicle_name,
    )

    sources = []

    for source_number, passage in enumerate(
        passages,
        start=1,
    ):
        metadata = passage["metadata"]

        sources.append(
            {
                "source_number": source_number,
                "vehicle_id": metadata["vehicle_id"],
                "make": metadata["make"],
                "model": metadata["model"],
                "year": metadata["year"],
                "manual_title": metadata["manual_title"],
                "pdf_page": metadata["pdf_page"],
                "source_file": metadata["source_file"],
                "distance": passage["distance"],
                "text": passage["text"],
            }
        )

    return {
        "answer": answer,
        "sources": sources,
    }