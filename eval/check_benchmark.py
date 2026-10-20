"""
Validates eval/benchmark.json (Tasks 1 and 2).

Checks the question mix against the Milestone 1 commitments, that every
paraphrase points at a question with gold data, and that every gold quote
really appears on one of its gold pages in the cached manual text.

Standard library only. Usage (from repo root):
    python -m eval.check_benchmark
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

eval_dir = Path(__file__).resolve().parent
benchmark_path = eval_dir / "benchmark.json"
cache_dir = eval_dir / "cache"

TYPES = {"direct", "paraphrased", "ambiguous", "unsupported"}
BEHAVIOURS = {"answer", "ask", "refuse", "escalate", "partial"}
MIN_QUESTIONS, MAX_QUESTIONS, MIN_UNSUPPORTED = 40, 60, 10


def load_benchmark() -> list[dict]:
    return json.loads(benchmark_path.read_text(encoding="utf-8"))


def normalise(text: str) -> str:
    # PDF text breaks words across lines ("infla- tion") and uses ligatures.
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"(\w)- (\w)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip().lower()


_page_cache: dict[str, list[str]] = {}


def page_text(vehicle_id: str, page: int) -> str:
    if vehicle_id not in _page_cache:
        pages = json.loads((cache_dir / f"{vehicle_id}.json").read_text(encoding="utf-8"))
        _page_cache[vehicle_id] = [normalise(p["text"]) for p in pages]
    pages = _page_cache[vehicle_id]
    return pages[page - 1] if 1 <= page <= len(pages) else ""


def gold_for(question: dict, by_id: dict) -> dict | None:
    """Paraphrases inherit the gold data of the question they rephrase."""
    if "gold" in question:
        return question["gold"]
    parent = by_id.get(question.get("paraphrase_of") or "")
    return parent.get("gold") if parent else None


def main() -> int:
    bank = load_benchmark()
    by_id = {q["id"]: q for q in bank}
    problems, warnings = [], []

    ids = [q["id"] for q in bank]
    for qid, count in Counter(ids).items():
        if count > 1:
            problems.append(f"duplicate id {qid}")

    for q in bank:
        if q.get("type") not in TYPES:
            problems.append(f"{q['id']}: unknown type {q.get('type')!r}")
        if q.get("expected_behaviour") not in BEHAVIOURS:
            problems.append(f"{q['id']}: unknown expected_behaviour {q.get('expected_behaviour')!r}")
        if q.get("type") == "paraphrased":
            parent = by_id.get(q.get("paraphrase_of") or "")
            if not parent:
                problems.append(f"{q['id']}: paraphrase_of points at a missing question")
            elif parent.get("vehicle_id") != q.get("vehicle_id"):
                problems.append(f"{q['id']}: vehicle differs from its parent {parent['id']}")
        if q.get("expected_behaviour") == "answer" and not gold_for(q, by_id):
            problems.append(f"{q['id']}: answerable but has no gold data")

    # ---- gold quote verification ------------------------------------
    verified, statuses = 0, Counter()
    for q in bank:
        g = q.get("gold")
        if not g:
            continue
        statuses[g.get("status", "missing")] += 1
        if not g.get("pages"):
            problems.append(f"{q['id']}: gold has no pages")
            continue
        quote = normalise(g.get("quote", ""))
        found_on = [p for p in g["pages"] if quote and quote in page_text(q["vehicle_id"], p)]
        if found_on:
            verified += 1
        else:
            problems.append(f"{q['id']}: gold quote not found on pages {g['pages']} ({q['vehicle_id']})")

    # ---- report -----------------------------------------------------
    types = Counter(q["type"] for q in bank)
    behaviours = Counter(q["expected_behaviour"] for q in bank)
    per_vehicle = Counter(q["vehicle_id"] or "(none named)" for q in bank)
    with_gold = sum(1 for q in bank if q.get("gold"))

    print(f"Benchmark: {len(bank)} questions")
    for t in ("direct", "paraphrased", "ambiguous", "unsupported"):
        print(f"  {t:12} {types.get(t, 0):3}")
    print("Expected behaviour: " + ", ".join(f"{b} {n}" for b, n in behaviours.most_common()))
    print("Per vehicle: " + ", ".join(f"{v} {n}" for v, n in sorted(per_vehicle.items())))
    print(f"\nGold data: {with_gold} questions (+{types.get('paraphrased', 0)} paraphrases inherit)")
    print(f"  quote found on a gold page: {verified}/{with_gold}")
    print("  status: " + ", ".join(f"{s} {n}" for s, n in statuses.most_common()))

    unsupported = types.get("unsupported", 0)
    if not MIN_QUESTIONS <= len(bank) <= MAX_QUESTIONS:
        warnings.append(f"{len(bank)} questions, Milestone 1 committed to {MIN_QUESTIONS}-{MAX_QUESTIONS}")
    if unsupported < MIN_UNSUPPORTED:
        warnings.append(f"{unsupported} unsupported, Milestone 1 committed to at least {MIN_UNSUPPORTED}")
    if statuses.get("draft") or statuses.get("needs_verification"):
        warnings.append("gold answers are not verified yet: set status to "
                        "'verified' and fill verified_by once checked against the PDF")

    print()
    for w in warnings:
        print(f"WARN  {w}")
    for p in problems:
        print(f"FAIL  {p}")
    if not problems:
        print(f"PASS  structure, paraphrase links and all gold quotes check out"
              f" ({len(bank)} questions, {unsupported} unsupported)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
