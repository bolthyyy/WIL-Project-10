"""
Scores saved answers (Tasks 4, 5, 6, 7) and applies the pre-registered
decision rule in answer_configs.json to select the final configuration (Task 9).

Automatic metrics need only the answers_*.jsonl files. Graded metrics appear
once grading_sheet.csv has been filled in using eval/rubric.md.

Usage (from repo root):
    python -m eval.score_answers
"""

import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

eval_dir = Path(__file__).resolve().parent
results_dir = eval_dir / "results" / "extended_benchmark"

REFUSAL = re.compile(r"insufficient|not (?:covered|included|available|provided|mentioned)"
                     r"|(?:does|do) not (?:contain|include|cover|specify|provide|mention)"
                     r"|no information|outside the scope|cannot (?:answer|find|provide)|unable to answer"
                     r"|withheld by emergency guard", re.I)
ASKING = re.compile(r"which (?:vehicle|model|car)|what (?:vehicle|model|car) (?:is|are)"
                    r"|(?:please |could you )(?:confirm|specify|clarify)", re.I)
CITATION = re.compile(r"\[source\s*(\d+)\]", re.I)

MILESTONE = {"correct_manual": 0.90, "relevant_retrieval": 0.85,
             "citation_support": 0.90, "unsupported_identified": 0.70}


def load_answers() -> dict[str, list[dict]]:
    runs = {}
    for path in sorted(results_dir.glob("answers_*.jsonl")):
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if rows:
            runs[path.stem.removeprefix("answers_")] = rows
    return runs


def auto_behaviour(answer: str) -> str:
    if ASKING.search(answer):
        return "ask"
    if REFUSAL.search(answer):
        return "refuse"
    return "answer"


def auto_metrics(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["expected_behaviour"] == "answer"]
    unsupported = [r for r in rows if r["expected_behaviour"] in ("refuse", "escalate")]
    vague = [r for r in rows if r["expected_behaviour"] == "ask"]

    # Validity is checked on every answer: a made-up [Source 7] inside a
    # refusal is still a fabricated citation.
    cited, valid, answerable_cited, wrong_vehicle = 0, 0, 0, 0
    for r in rows:
        numbers = [int(n) for n in CITATION.findall(r["answer"])]
        if not numbers:
            continue
        cited += 1
        valid += all(1 <= n <= len(r["passages"]) for n in numbers)
        if r["expected_behaviour"] == "answer":
            answerable_cited += 1
            vehicles = {r["passages"][n - 1]["vehicle_id"] for n in numbers if 1 <= n <= len(r["passages"])}
            wrong_vehicle += bool(r["vehicle_id"]) and any(v != r["vehicle_id"] for v in vehicles)

    def share(group, predicate):
        return sum(predicate(r) for r in group) / len(group) if group else None

    return {
        "n": len(rows),
        "auto_false_refusal": share(answerable, lambda r: auto_behaviour(r["answer"]) != "answer"),
        "auto_unsupported_identified": share(unsupported, lambda r: auto_behaviour(r["answer"]) == "refuse"),
        "auto_asked_when_vague": share(vague, lambda r: auto_behaviour(r["answer"]) == "ask"),
        "citation_present": answerable_cited / len(answerable) if answerable else None,
        "citation_valid": valid / cited if cited else None,
        "auto_wrong_vehicle_cited": wrong_vehicle / len(answerable) if answerable else None,
        "median_seconds": statistics.median(r["seconds"] for r in rows),
    }


def number(value):
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def cohens_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    labels = {x for pair in pairs for x in pair}
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    expected = sum((sum(a == l for a, _ in pairs) / n) * (sum(b == l for _, b in pairs) / n) for l in labels)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def graded_metrics() -> tuple[dict, dict]:
    sheet_path, key_path = results_dir / "grading_sheet.csv", results_dir / "grading_key.csv"
    if not sheet_path.exists():
        return {}, {}
    with open(key_path, encoding="utf-8-sig") as f:
        key = {r["blind_id"]: r for r in csv.DictReader(f)}
    with open(sheet_path, encoding="utf-8-sig") as f:
        graded = [r for r in csv.DictReader(f) if r["behaviour_correct_y_n"].strip()]
    if not graded:
        return {}, {}

    by_config, by_item = defaultdict(list), defaultdict(list)
    for r in graded:
        k = key[r["blind_id"]]
        by_item[(k["config"], k["id"])].append(r)
        if not r["blind_id"].endswith("-R2"):
            by_config[k["config"]].append(r)

    agreement = {}
    for column in ("correctness_0_1_2", "behaviour_correct_y_n"):
        pairs = [(rs[0][column].strip(), rs[1][column].strip()) for rs in by_item.values()
                 if len(rs) > 1 and rs[0][column].strip() and rs[1][column].strip()]
        agreement[column] = (cohens_kappa(pairs), len(pairs))

    metrics = {}
    for config, rs in by_config.items():
        scores = [number(r["correctness_0_1_2"]) for r in rs if number(r["correctness_0_1_2"]) is not None]
        claims = sum(number(r["n_claims"]) or 0 for r in rs)
        unsupported = sum(number(r["n_unsupported_claims"]) or 0 for r in rs)
        citations = sum(number(r["n_citations"]) or 0 for r in rs)
        supported = sum(number(r["n_citations_supported"]) or 0 for r in rs)
        answerable = [r for r in rs if r["expected_behaviour"] == "answer"]
        refusable = [r for r in rs if r["expected_behaviour"] in ("refuse", "escalate")]
        yes = lambda group: sum(r["behaviour_correct_y_n"].strip().lower() == "y" for r in group) / len(group) if group else None
        metrics[config] = {
            "graded": len(rs),
            "answer_correctness": sum(scores) / (2 * len(scores)) if scores else None,
            "fully_correct": sum(s == 2 for s in scores) / len(scores) if scores else None,
            "faithfulness": 1 - unsupported / claims if claims else None,
            "answers_with_unsupported_claim": sum((number(r["n_unsupported_claims"]) or 0) > 0 for r in rs) / len(rs),
            "citation_support": supported / citations if citations else None,
            "behaviour_accuracy": yes(rs),
            "false_refusal": 1 - yes(answerable) if answerable else None,
            "unsupported_identified": yes(refusable),
            "wrong_vehicle": sum(r["wrong_vehicle_y_n"].strip().lower() == "y" for r in rs) / len(rs),
        }
    return metrics, agreement


def pct(value) -> str:
    return "  -  " if value is None else f"{value:5.0%}"


def main() -> None:
    runs = load_answers()
    if not runs:
        raise SystemExit("No saved answers. Run: python -m eval.run_answers")

    auto = {config: auto_metrics(rows) for config, rows in runs.items()}
    print("Automatic metrics (no grading needed; refusal detection is a heuristic, check it by hand)\n")
    print(f"{'config':18} {'n':>3} {'false refusal':>13} {'unsupported id.':>15} {'asked if vague':>14} "
          f"{'cited':>6} {'valid cite':>10} {'wrong vehicle':>13} {'median s':>9}")
    for config, m in auto.items():
        print(f"{config:18} {m['n']:3} {pct(m['auto_false_refusal']):>13} {pct(m['auto_unsupported_identified']):>15} "
              f"{pct(m['auto_asked_when_vague']):>14} {pct(m['citation_present']):>6} {pct(m['citation_valid']):>10} "
              f"{pct(m['auto_wrong_vehicle_cited']):>13} {m['median_seconds']:9.1f}")

    rubric_scores, agreement = graded_metrics()
    if not rubric_scores:
        print("\nNo graded rows yet. Run make_grading_sheet, grade with eval/rubric.md, then rerun.")
        return

    print("\nGrader agreement: " + ", ".join(
        f"{col.split('_')[0]} kappa {k:.2f} (n={n})" if k is not None else f"{col.split('_')[0]}: no double-graded rows"
        for col, (k, n) in agreement.items()))

    print(f"\nGraded metrics\n{'config':18} {'graded':>6} {'correct':>8} {'fully':>6} {'faithful':>9} "
          f"{'cite supp.':>10} {'behaviour':>9} {'false ref.':>10} {'unsupp. id':>10} {'wrong veh.':>10}")
    for config, m in rubric_scores.items():
        print(f"{config:18} {m['graded']:6} {pct(m['answer_correctness']):>8} {pct(m['fully_correct']):>6} "
              f"{pct(m['faithfulness']):>9} {pct(m['citation_support']):>10} {pct(m['behaviour_accuracy']):>9} "
              f"{pct(m['false_refusal']):>10} {pct(m['unsupported_identified']):>10} {pct(m['wrong_vehicle']):>10}")

    weights = json.loads((eval_dir / "answer_configs.json").read_text(encoding="utf-8"))["decision_weights"]
    print("\nDecision rule (from answer_configs.json): " + " + ".join(f"{w} x {name}" for name, w in weights.items()))
    decisions = []
    for config, m in rubric_scores.items():
        parts = {"answer_correctness": m["answer_correctness"], "faithfulness": m["faithfulness"],
                 "citation_support": m["citation_support"], "behaviour_accuracy": m["behaviour_accuracy"],
                 "not_wrong_vehicle": 1 - m["wrong_vehicle"]}
        missing = [k for k, v in parts.items() if v is None]
        score = None if missing else sum(weights[k] * v for k, v in parts.items())
        decisions.append((config, score, missing))
        print(f"  {config:18} " + (f"{score:.3f}" if score is not None else f"incomplete, missing {', '.join(missing)}"))

    complete = [d for d in decisions if d[1] is not None]
    if complete:
        best = max(complete, key=lambda d: d[1])
        print(f"\nSelected configuration (Task 9): {best[0]}  (score {best[1]:.3f})")
        m = rubric_scores[best[0]]
        print("Against Milestone 1 targets:")
        for label, value, target in (("correct supporting source", m["citation_support"], MILESTONE["citation_support"]),
                                     ("unsupported identified", m["unsupported_identified"], MILESTONE["unsupported_identified"])):
            if value is not None:
                print(f"  {label:26} {value:5.0%}  target {target:.0%}  {'PASS' if value >= target else 'MISS'}")

    rows = [{"config": c, **{k: v for k, v in auto[c].items()}, **rubric_scores.get(c, {}),
             "decision_score": next((s for n, s, _ in decisions if n == c), None)} for c in auto]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(results_dir / "answer_scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {results_dir / 'answer_scores.csv'}")


if __name__ == "__main__":
    main()
