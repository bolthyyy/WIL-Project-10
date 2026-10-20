"""
Ground-truth helper: search the cached manual text for candidate gold pages.

Reads eval/cache/{vehicle_id}.json (pymupdf text from extract_cache.py), so it
works for the Kia manuals too. Needs no packages beyond the standard library.

Usage (from repo root):
    python -m eval.find_pages "tire pressure" --vehicle toyota_corolla
    python -m eval.find_pages "jump start|booster" --vehicle kia_cerato
    python -m eval.find_pages --vehicle toyota_corolla --page 391
"""

import argparse
import json
import re
from pathlib import Path

eval_dir = Path(__file__).resolve().parent
cache_dir = eval_dir / "cache"


def load_pages(vehicle_id: str) -> list[dict]:
    path = cache_dir / f"{vehicle_id}.json"
    if not path.exists():
        raise SystemExit(f"No cache for {vehicle_id}. Run: python -m eval.extract_cache")
    return json.loads(path.read_text(encoding="utf-8"))


def looks_like_index(text: str) -> bool:
    # Contents and index pages are dense with page references.
    return len(re.findall(r"\b\d{1,3}\b", text)) > 60


def search(vehicle_id: str, pattern: re.Pattern, limit: int, context: int):
    hits = []
    for page in load_pages(vehicle_id):
        text = page["text"]
        matches = list(pattern.finditer(text))
        if not matches or looks_like_index(text):
            continue
        start = max(0, matches[0].start() - context // 2)
        hits.append((len(matches), page["page"], text[start:start + context]))

    hits.sort(key=lambda hit: (-hit[0], hit[1]))
    return hits[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phrase", nargs="?", help="Text or regex to search for.")
    parser.add_argument("--vehicle", required=True)
    parser.add_argument("--page", type=int, help="Print one page in full instead.")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--context", type=int, default=300)
    arguments = parser.parse_args()

    if arguments.page:
        pages = load_pages(arguments.vehicle)
        print(f"=== {arguments.vehicle} page {arguments.page} ===")
        print(pages[arguments.page - 1]["text"])
        return

    if not arguments.phrase:
        parser.error("give a phrase to search for, or --page N")

    pattern = re.compile(arguments.phrase, re.IGNORECASE)
    hits = search(arguments.vehicle, pattern, arguments.limit, arguments.context)

    print(f"=== {arguments.vehicle}: {len(hits)} best pages for {arguments.phrase!r} ===")
    for count, page, snippet in hits:
        print(f"\n[page {page}] {count} matches\n  {snippet}")


if __name__ == "__main__":
    main()
