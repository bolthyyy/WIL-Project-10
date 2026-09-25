import json

from src.pdf_utils import clean_text
from src.settings import MANUAL_DIR, MANUAL_METADATA_FILE
from eval.index import cache_dir

import pymupdf

def main():
    cache_dir.mkdir(exist_ok=True, parents=True)

    manuals = json.loads(MANUAL_METADATA_FILE.read_text(encoding="utf-8"))
    for manual in manuals:
        pdf_path = MANUAL_DIR / manual["file"]
        document = pymupdf.open(str(pdf_path))

        pages = []

        for page_index in range(len(document)):
            raw_text = document[page_index].get_text() or ""
            pages.append(
                {
                    "page": page_index + 1,
                    "text": clean_text(raw_text),
                }
            )

        out_path = cache_dir / f"{manual['vehicle_id']}.json"
        out_path.write_text(json.dumps(pages), encoding="utf-8")

        total_chars = 0
        for page in pages:
            total_chars += len(page["text"])

        print(f"{manual['vehicle_id']}: {len(pages)} pages, "
              f"{total_chars} characters cached")

if __name__ == "__main__":
    main()