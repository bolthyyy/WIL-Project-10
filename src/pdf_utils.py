import re
from pathlib import Path

from pypdf import PdfReader


def clean_text(text: str) -> str:
    """
    Remove excessive whitespace from extracted PDF text.
    """
    return re.sub(r"\s+", " ", text).strip()


def split_into_chunks(
    text: str,
    chunk_size_words: int,
    overlap_words: int,
) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    """
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words")

    words = text.split()

    if not words:
        return []

    chunks = []
    step = chunk_size_words - overlap_words

    for start in range(0, len(words), step):
        end = start + chunk_size_words
        chunk_words = words[start:end]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break

    return chunks


def extract_pdf_chunks(
    pdf_path: Path,
    chunk_size_words: int,
    overlap_words: int,
) -> list[dict]:
    """
    Extract text from a PDF one page at a time and divide each page
    into chunks.

    PDF page numbers are stored so retrieved sources can be traced
    back to their original location.
    """
    reader = PdfReader(str(pdf_path))

    output = []

    for pdf_page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned_text = clean_text(raw_text)

        if not cleaned_text:
            continue

        chunks = split_into_chunks(
            cleaned_text,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
        )

        for chunk_number, chunk in enumerate(chunks):
            output.append(
                {
                    "text": chunk,
                    "pdf_page": pdf_page_number,
                    "chunk_number": chunk_number,
                }
            )

    return output