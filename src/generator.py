import requests

from .settings import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)


SYSTEM_PROMPT = """
You are an internal information assistant for rental-car
roadside support staff.

Your job is to answer questions using only the owner's-manual
passages supplied to you.

Rules:

1. Use only information supported by the supplied sources.
2. Do not rely on general automotive knowledge.
3. If the sources do not contain enough information to answer
   the question, state that the available owner's-manual
   information is insufficient.
4. Do not invent procedures, specifications, warnings,
   diagnoses, or repair advice.
5. Do not present yourself as a mechanic or emergency service.
6. If the manual indicates that professional assistance,
   roadside assistance, or emergency action is required,
   preserve that recommendation.
7. Cite supporting passages using [Source 1], [Source 2], etc.
8. Keep the response concise and practical for a rental support
   employee.
""".strip()


def build_context(passages: list[dict]) -> str:
    context_blocks = []

    for source_number, passage in enumerate(
        passages,
        start=1,
    ):
        metadata = passage["metadata"]

        source_header = (
            f"[Source {source_number}] "
            f"{metadata['make']} "
            f"{metadata['model']} "
            f"({metadata['year']}), "
            f"{metadata['manual_title']}, "
            f"PDF page {metadata['pdf_page']}"
        )

        context_blocks.append(
            f"{source_header}\n"
            f"{passage['text']}"
        )

    return "\n\n".join(context_blocks)


def generate_answer(
    question: str,
    passages: list[dict],
    vehicle_name: str | None = None,
) -> str:
    context = build_context(passages)

    vehicle_text = (
        vehicle_name
        if vehicle_name
        else "Vehicle not explicitly specified"
    )

    user_prompt = f"""
Vehicle:
{vehicle_text}

Retrieved owner's-manual passages:

{context}

Customer/support question:
{question}

Answer the question using only the retrieved passages.
""".strip()

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "stream": False,
        "options": {
            "temperature": 0.1
        },
    }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=300,
        )

        response.raise_for_status()

    except requests.RequestException as error:
        raise RuntimeError(
            "Could not communicate with Ollama. "
            "Check that Ollama is installed and running."
        ) from error

    data = response.json()

    return data["message"]["content"]