"""
Connects draft grades to the blind grading sheet (Tasks 4-6).

Workflow:
    python -m eval.make_grading_sheet --configs product <finalist>
    python -m eval.apply_draft_grades fill      # draft grades -> main rows,
                                                # writes grading_verify.csv
    (grade grading_verify.csv blind, using eval/rubric.md)
    python -m eval.apply_draft_grades merge     # your grades -> the -R2 rows
    python -m eval.score_answers                # metrics + Cohen's kappa

Draft grades live in eval/results/extended_benchmark/draft_grades.json, keyed
"<config>::<question id>". They are written to the main rows of
grading_sheet.csv. The duplicated "-R2" rows are left for an independent
grader: grading_verify.csv holds only those rows, with no draft grade
visible, so the second grade is blind. score_answers then reports agreement
between the two graders.

Standard library only. Usage (from repo root): see above.
"""

import csv
import json
import sys
from pathlib import Path

results_dir = Path(__file__).resolve().parent / "results" / "extended_benchmark"
sheet_path = results_dir / "grading_sheet.csv"
key_path = results_dir / "grading_key.csv"
verify_path = results_dir / "grading_verify.csv"
drafts_path = results_dir / "draft_grades.json"

GRADE_COLUMNS = ["grader", "behaviour_correct_y_n", "correctness_0_1_2", "n_claims",
                 "n_unsupported_claims", "n_citations", "n_citations_supported",
                 "wrong_vehicle_y_n", "notes"]


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fill() -> None:
    sheet, key = read(sheet_path), {r["blind_id"]: r for r in read(key_path)}
    drafts = json.loads(drafts_path.read_text(encoding="utf-8"))
    filled, missing = 0, []
    for row in sheet:
        if row["blind_id"].endswith("-R2"):
            continue
        k = key[row["blind_id"]]
        grade = drafts.get(f"{k['config']}::{k['id']}")
        if grade is None:
            missing.append(f"{k['config']}::{k['id']}")
            continue
        for column in GRADE_COLUMNS:
            row[column] = grade.get(column, "")
        filled += 1
    write(sheet_path, sheet)

    verify = [{c: ("" if c in GRADE_COLUMNS else v) for c, v in row.items()}
              for row in sheet if row["blind_id"].endswith("-R2")]
    write(verify_path, verify)
    print(f"filled {filled} main rows from draft grades; {len(missing)} without a draft grade")
    if missing:
        print("  missing: " + ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""))
    print(f"wrote {verify_path.name}: {len(verify)} rows to grade blind with eval/rubric.md")


def merge() -> None:
    sheet = read(sheet_path)
    graded = {r["blind_id"]: r for r in read(verify_path) if r["behaviour_correct_y_n"].strip()}
    merged = 0
    for row in sheet:
        if row["blind_id"] in graded:
            for column in GRADE_COLUMNS:
                row[column] = graded[row["blind_id"]][column]
            merged += 1
    write(sheet_path, sheet)
    total = sum(r["blind_id"].endswith("-R2") for r in sheet)
    print(f"merged {merged} of {total} verification rows into {sheet_path.name}")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    if action == "fill":
        fill()
    elif action == "merge":
        merge()
    else:
        raise SystemExit("usage: python -m eval.apply_draft_grades fill|merge")
