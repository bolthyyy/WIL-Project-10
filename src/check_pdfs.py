import json

from pypdf import PdfReader

from .settings import MANUAL_DIR, MANUAL_METADATA_FILE


def main():
    with open(MANUAL_METADATA_FILE, "r", encoding="utf-8") as file:
        manuals = json.load(file)

    print("\nChecking PDF files...\n")

    for manual in manuals:
        pdf_path = MANUAL_DIR / manual["file"]

        print(
            f"{manual['make']} {manual['model']}: "
            f"{pdf_path.name}"
        )

        if not pdf_path.exists():
            print("  ERROR: File does not exist.\n")
            continue

        reader = PdfReader(str(pdf_path))

        page_count = len(reader.pages)

        sample_page_indexes = [
            0,
            min(10, page_count - 1),
            page_count // 4,
            page_count // 2,
            (3 * page_count) // 4,
            page_count - 1,
        ]

        sample_page_indexes = sorted(set(sample_page_indexes))

        extracted_characters = 0

        for page_index in sample_page_indexes:
            text = reader.pages[page_index].extract_text() or ""
            extracted_characters += len(text)

        print(f"  Pages: {page_count}")
        print(
            f"  Characters extracted from "
            f"{len(sample_page_indexes)} sampled pages: "
            f"{extracted_characters}"
        )

        if extracted_characters == 0:
            print(
                "  WARNING: No text extracted. "
                "This PDF may require a different extraction method."
            )

        print()


if __name__ == "__main__":
    main()