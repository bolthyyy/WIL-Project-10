"""
Retrieval scored against the verified gold pages in benchmark.json
(Task 3 retrieval accuracy, Task 8 cross-model errors, per-vehicle fairness).

Reuses Jack's EvalIndex unchanged. Writes only to eval/results/extended_benchmark/.

A hit means a retrieved passage is from the gold vehicle AND within
PAGE_TOLERANCE of a gold page. Hit@1/3/5/10 and MRR come from one top-10
search per question.

Modes:
    filtered          search restricted to the question's vehicle (the app's behaviour)
    unfiltered_named  all manuals searched, vehicle named in the question text
    unfiltered_plain  all manuals searched, question as written (Jack's unfiltered run)

Usage (from repo root):
    python -m eval.score_retrieval_gold
    python -m eval.score_retrieval_gold --embedders bm25 minilm mpnet
    python -m eval.score_retrieval_gold --text-source pypdf     # the text production indexes
"""

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import eval.index as eval_index
from eval.check_benchmark import gold_for, load_benchmark
from eval.index import EvalIndex, load_manuals

results_dir = Path(__file__).resolve().parent / "results" / "extended_benchmark"
PAGE_TOLERANCE = 1
MAX_K = 10
KS = (1, 3, 5, 10)
CHUNKS = [(120, 20), (220, 40), (350, 60)]
MODES = ("filtered", "unfiltered_named", "unfiltered_plain")


def use_pypdf_text(manuals: list[dict]) -> None:
    """Point EvalIndex at pypdf-extracted text instead of the pymupdf cache."""
    from pypdf import PdfReader
    from src.pdf_utils import clean_text
    from src.settings import MANUAL_DIR

    cache = results_dir / "cache_pypdf"
    cache.mkdir(parents=True, exist_ok=True)
    for manual in manuals:
        path = cache / f"{manual['vehicle_id']}.json"
        if path.exists():
            continue
        print(f"  extracting {manual['file']} with pypdf ...")
        reader = PdfReader(str(MANUAL_DIR / manual["file"]))
        pages = [{"page": i, "text": clean_text(p.extract_text() or "")}
                 for i, p in enumerate(reader.pages, start=1)]
        path.write_text(json.dumps(pages), encoding="utf-8")
    eval_index.cache_dir = cache


def answerable_questions() -> list[dict]:
    bank = load_benchmark()
    by_id = {q["id"]: q for q in bank}
    out = []
    for q in bank:
        g = gold_for(q, by_id)
        if q["expected_behaviour"] == "answer" and g and g.get("pages"):
            out.append({**q, "gold_pages": g["pages"]})
    return out


def score(passages: list[dict], vehicle_id: str, gold_pages: list[int]) -> dict:
    first = None
    for rank, p in enumerate(passages, start=1):
        meta = p["metadata"]
        if meta["vehicle_id"] == vehicle_id and any(
                abs(meta["pdf_page"] - g) <= PAGE_TOLERANCE for g in gold_pages):
            first = rank
            break
    top5 = passages[:5]
    row = {f"hit@{k}": int(first is not None and first <= k) for k in KS}
    row["mrr"] = round(1 / first, 4) if first else 0.0
    row["first_hit_rank"] = first or ""
    row["top1_correct_manual"] = int(bool(passages) and passages[0]["metadata"]["vehicle_id"] == vehicle_id)
    row["wrong_manual_share_top5"] = round(
        sum(p["metadata"]["vehicle_id"] != vehicle_id for p in top5) / len(top5), 2) if top5 else ""
    row["top1_vehicle"] = passages[0]["metadata"]["vehicle_id"] if passages else ""
    row["top1_page"] = passages[0]["metadata"]["pdf_page"] if passages else ""
    return row


def mean(rows: list[dict], key: str) -> float:
    values = [float(r[key]) for r in rows if r[key] != ""]
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedders", nargs="+", default=["bm25", "minilm"])
    parser.add_argument("--chunks", nargs="+", default=[f"{c}/{o}" for c, o in CHUNKS])
    parser.add_argument("--text-source", choices=["pymupdf", "pypdf"], default="pymupdf")
    arguments = parser.parse_args()

    manuals = load_manuals()
    names = {m["vehicle_id"]: f"{m['make']} {m['model']}" for m in manuals}
    if arguments.text_source == "pypdf":
        use_pypdf_text(manuals)

    questions = answerable_questions()
    print(f"{len(questions)} answerable questions with gold pages, text from {arguments.text_source}")

    rows = []
    for embedder in arguments.embedders:
        for chunk in arguments.chunks:
            size, overlap = (int(x) for x in chunk.split("/"))
            print(f"\nbuilding {embedder} {chunk} ...")
            try:
                index = EvalIndex(embedder, size, overlap)
                index.build(manuals)
            except Exception as error:
                print(f"  skipped ({error.__class__.__name__}: {error})")
                continue

            for q in questions:
                for mode in MODES:
                    query = q["question"]
                    vehicle_filter = q["vehicle_id"] if mode == "filtered" else None
                    if mode == "unfiltered_named":
                        query = f"{query} ({names[q['vehicle_id']]})"
                    passages, latency_ms, _ = index.search(query, vehicle_id=vehicle_filter, top_k=MAX_K)
                    rows.append({
                        "text_source": arguments.text_source, "embedder": embedder, "chunk": chunk,
                        "mode": mode, "question_id": q["id"], "type": q["type"],
                        "vehicle_id": q["vehicle_id"], "gold_pages": ";".join(map(str, q["gold_pages"])),
                        **score(passages, q["vehicle_id"], q["gold_pages"]),
                        "latency_ms": round(latency_ms, 1),
                    })

    if not rows:
        print("nothing scored")
        return

    suffix = "" if arguments.text_source == "pymupdf" else "_pypdf"
    write_csv(results_dir / f"retrieval_gold{suffix}.csv", rows)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["embedder"], r["chunk"], r["mode"])].append(r)

    summary = []
    print(f"\n{'config':22} {'mode':17} {'Hit@1':>6} {'Hit@3':>6} {'Hit@5':>6} {'Hit@10':>7} {'MRR':>6} {'top1 manual':>12} {'wrong@5':>8}")
    for (embedder, chunk, mode), rs in groups.items():
        s = {"text_source": arguments.text_source, "embedder": embedder, "chunk": chunk, "mode": mode, "n": len(rs),
             **{f"hit@{k}": round(mean(rs, f"hit@{k}"), 3) for k in KS},
             "mrr": round(mean(rs, "mrr"), 3),
             "top1_correct_manual": round(mean(rs, "top1_correct_manual"), 3),
             "wrong_manual_share_top5": round(mean(rs, "wrong_manual_share_top5"), 3)}
        summary.append(s)
        print(f"{embedder + ' ' + chunk:22} {mode:17} {s['hit@1']:6.2f} {s['hit@3']:6.2f} {s['hit@5']:6.2f} "
              f"{s['hit@10']:7.2f} {s['mrr']:6.2f} {s['top1_correct_manual']:12.2f} {s['wrong_manual_share_top5']:8.2f}")
    write_csv(results_dir / f"retrieval_gold_summary{suffix}.csv", summary)

    print("\nPer vehicle, filtered Hit@5 (fairness):")
    for embedder in arguments.embedders:
        for chunk in arguments.chunks:
            rs = [r for r in rows if r["embedder"] == embedder and r["chunk"] == chunk and r["mode"] == "filtered"]
            if not rs:
                continue
            by_vehicle = defaultdict(list)
            for r in rs:
                by_vehicle[r["vehicle_id"]].append(r)
            scores = {v: mean(vr, "hit@5") for v, vr in sorted(by_vehicle.items())}
            line = "  ".join(f"{v.split('_', 1)[1]} {s:.2f}" for v, s in scores.items())
            spread = max(scores.values()) - min(scores.values())
            print(f"  {embedder} {chunk}: {line}  | spread {spread:.2f}, sd {statistics.pstdev(scores.values()):.2f}")

    print("\nDirect vs paraphrased, filtered Hit@5 (robustness to customer wording):")
    for (embedder, chunk, mode), rs in groups.items():
        if mode != "filtered":
            continue
        d = [r for r in rs if r["type"] == "direct"]
        p = [r for r in rs if r["type"] == "paraphrased"]
        print(f"  {embedder} {chunk}: direct {mean(d, 'hit@5'):.2f} (n={len(d)})  paraphrased {mean(p, 'hit@5'):.2f} (n={len(p)})")

    print(f"\nwrote {results_dir / f'retrieval_gold{suffix}.csv'} and the summary csv")


if __name__ == "__main__":
    main()
