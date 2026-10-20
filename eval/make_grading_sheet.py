"""
Builds a BLIND grading sheet from saved answers (Tasks 4, 5, 6 and the graded
check of Task 7). Grade it using eval/rubric.md.

Answers from every config are pooled, shuffled and stripped of the config
name, so graders can't favour one. The mapping goes to a separate key file.
A share of items is duplicated for a second grader, for Cohen's kappa.

Usage (from repo root):
    python -m eval.make_grading_sheet
    python -m eval.make_grading_sheet --configs minilm_strict product --double 0.2
"""

import argparse
import csv
import json
import random
from pathlib import Path

from eval.check_benchmark import gold_for, load_benchmark

results_dir = Path(__file__).resolve().parent / "results" / "extended_benchmark"

GRADE_COLUMNS = ["grader", "behaviour_correct_y_n", "correctness_0_1_2", "n_claims",
                 "n_unsupported_claims", "n_citations", "n_citations_supported",
                 "wrong_vehicle_y_n", "notes"]


def trim(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + " ..."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", help="Only these config ids (e.g. the finalists).")
    parser.add_argument("--double", type=float, default=0.2, help="Share double-graded for kappa.")
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    bank = load_benchmark()
    by_id = {q["id"]: q for q in bank}

    items = []
    for path in sorted(results_dir.glob("answers_*.jsonl")):
        config_id = path.stem.removeprefix("answers_")
        if arguments.configs and config_id not in arguments.configs:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            a = json.loads(line)
            gold = gold_for(by_id.get(a["id"], {}), by_id) or {}
            items.append({
                "config": config_id, "id": a["id"],
                "question": a["question"],
                "vehicle": a["vehicle_id"] or "(none named)",
                "expected_behaviour": a["expected_behaviour"],
                "gold_answer": gold.get("answer", ""),
                "gold_pages": ";".join(map(str, gold.get("pages", []))),
                "system_answer": a["answer"],
                "retrieved_passages": "\n\n".join(
                    f"[Source {p['n']}] {p['vehicle_id']} p.{p['pdf_page']}: {trim(p['text'], 450)}"
                    for p in a["passages"]),
            })

    if not items:
        raise SystemExit("No saved answers found. Run: python -m eval.run_answers")

    rng = random.Random(arguments.seed)
    rng.shuffle(items)

    sheet, key = [], []
    for n, item in enumerate(items, start=1):
        blind = f"ITEM{n:04d}"
        key.append({"blind_id": blind, "config": item.pop("config"), "id": item.pop("id")})
        sheet.append({"blind_id": blind, **item, **{c: "" for c in GRADE_COLUMNS}})

    for original in rng.sample(sheet, int(len(sheet) * arguments.double)):
        copy = {**original, "blind_id": original["blind_id"] + "-R2"}
        source = next(k for k in key if k["blind_id"] == original["blind_id"])
        key.append({**source, "blind_id": copy["blind_id"]})
        sheet.append(copy)
    rng.shuffle(sheet)

    results_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("grading_sheet.csv", sheet), ("grading_key.csv", key)):
        with open(results_dir / name, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    doubled = sum(1 for r in sheet if r["blind_id"].endswith("-R2"))
    print(f"{len(sheet)} items ({doubled} double-graded) from {len({k['config'] for k in key})} configs")
    print(f"  grade this : {results_dir / 'grading_sheet.csv'}  (rules: eval/rubric.md)")
    print(f"  don't open : {results_dir / 'grading_key.csv'}  (reveals which config wrote each answer)")


if __name__ == "__main__":
    main()
