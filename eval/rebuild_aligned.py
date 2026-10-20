import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import eval.index as eval_index
from eval.check_benchmark import load_benchmark
from eval.index import EvalIndex, load_manuals
from eval.score_retrieval_gold import answerable_questions, use_pypdf_text

eval_dir = Path(__file__).resolve().parent
jack_results = eval_dir / "results"
out_dir = eval_dir / "results" / "extended_benchmark" / "aligned"

PAGE_TOLERANCE = 1
BASE_EMBEDDER, BASE_CHUNK, BASE_K = "bm25", (220, 40), 5
TOP_K_VALUES = [1, 3, 5, 8]
CHUNKS = [(120, 20), (220, 40), (350, 60)]
EMBEDDERS = ["bm25", "minilm", "mpnet"]
JACK_17 = {f"Q{i:02d}" for i in range(1, 17)} | {"Q28"}

_original_cache_dir = eval_index.cache_dir
_indexes: dict[tuple, EvalIndex] = {}


# --------------------------------------------------------------- indexes
def get_index(text_source: str, embedder: str, chunk: tuple[int, int], manuals) -> EvalIndex | None:
    key = (text_source, embedder, chunk)
    if key in _indexes:
        return _indexes[key]
    if text_source == "pypdf":
        use_pypdf_text(manuals)          # points eval_index.cache_dir at the pypdf cache
    else:
        eval_index.cache_dir = _original_cache_dir
    print(f"  building {embedder} {chunk[0]}/{chunk[1]} on {text_source} text ...", flush=True)
    try:
        index = EvalIndex(embedder, *chunk)
        index.build(manuals)
    except Exception as error:
        print(f"    skipped ({error.__class__.__name__}: {error})")
        _indexes[key] = None
        return None
    finally:
        eval_index.cache_dir = _original_cache_dir
    _indexes[key] = index
    return index


# --------------------------------------------------------------- scoring
def run(index: EvalIndex, q: dict, filtered: bool, top_k: int, rerank: bool = False) -> dict:
    vehicle = q["vehicle_id"] if filtered else None
    passages, latency_ms, top1_score = index.search(q["question"], vehicle_id=vehicle,
                                                    top_k=top_k, rerank=rerank)
    first = None
    for rank, p in enumerate(passages, start=1):
        meta = p["metadata"]
        if meta["vehicle_id"] == q["vehicle_id"] and any(
                abs(meta["pdf_page"] - g) <= PAGE_TOLERANCE for g in q["gold_pages"]):
            first = rank
            break
    wrong = sum(p["metadata"]["vehicle_id"] != q["vehicle_id"] for p in passages)
    return {
        "question_id": q["id"], "type": q["type"], "vehicle_id": q["vehicle_id"],
        "jack_question": int(q["id"] in JACK_17),
        "gold_pages": ";".join(map(str, q["gold_pages"])),
        "hit": int(first is not None),
        "mrr": round(1 / first, 4) if first else 0.0,
        "first_hit_rank": first or "",
        "top1_correct_manual": int(bool(passages) and passages[0]["metadata"]["vehicle_id"] == q["vehicle_id"]),
        "wrong_manual_share": round(wrong / len(passages), 2) if passages else "",
        "top1_vehicle": passages[0]["metadata"]["vehicle_id"] if passages else "",
        "top1_page": passages[0]["metadata"]["pdf_page"] if passages else "",
        "top1_score": round(top1_score, 4) if top1_score is not None else "",
        "latency_ms": round(latency_ms, 1),
    }


def mean(rows, key):
    values = [float(r[key]) for r in rows if r.get(key) not in ("", None)]
    return round(sum(values) / len(values), 3) if values else ""


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------- Jack's original numbers
def jack_means(filename: str, column: str, value: str) -> dict:
    path = jack_results / filename
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get(column) == value]
    if not rows:
        return {}

    def rate(key):
        vals = [1.0 if r[key] == "True" else 0.0 for r in rows if r.get(key) in ("True", "False")]
        return round(sum(vals) / len(vals), 3) if vals else ""

    ndcg = [float(r["ndcg"]) for r in rows if r.get("ndcg") not in ("", None)]
    return {"jack_n": len(rows), "jack_page_hit": rate("page_hit"),
            "jack_ndcg": round(sum(ndcg) / len(ndcg), 3) if ndcg else "",
            "jack_top1_correct_manual": rate("top1_correct_manual")}


# --------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-mpnet", action="store_true")
    parser.add_argument("--skip-pypdf", action="store_true")
    args = parser.parse_args()

    manuals = load_manuals()
    bank = load_benchmark()
    questions = answerable_questions()               # 33: direct + paraphrased with gold pages
    print(f"{len(questions)} answerable questions ({sum(q['id'] in JACK_17 for q in questions)} of them Jack's)\n")

    per_question: list[dict] = []
    summary: list[dict] = []

    def experiment(name, condition, index, filtered, top_k, rerank=False, jack=None, text_source="pymupdf",
                   embedder=BASE_EMBEDDER, chunk=BASE_CHUNK):
        if index is None:
            return []
        rows = []
        for q in questions:
            r = run(index, q, filtered, top_k, rerank)
            rows.append({"experiment": name, "condition": condition, "text_source": text_source,
                         "embedder": embedder, "chunk": f"{chunk[0]}/{chunk[1]}", "top_k": top_k,
                         "filtered": int(filtered), "rerank": int(rerank), **r})
        per_question.extend(rows)
        jack_rows = [r for r in rows if r["jack_question"]]
        summary.append({
            "experiment": name, "condition": condition, "text_source": text_source, "n": len(rows),
            "hit_at_k": mean(rows, "hit"), "mrr": mean(rows, "mrr"),
            "top1_correct_manual": mean(rows, "top1_correct_manual"),
            "wrong_manual_share": mean(rows, "wrong_manual_share"),
            "latency_ms": mean(rows, "latency_ms"),
            "hit_at_k_jack17": mean(jack_rows, "hit"),
            **(jack_means(*jack) if jack else {}),
        })
        print(f"    {name:24} {condition:28} Hit@{top_k}={summary[-1]['hit_at_k']:.3f}  MRR={summary[-1]['mrr']:.3f}")
        return rows

    base = get_index("pymupdf", BASE_EMBEDDER, BASE_CHUNK, manuals)

    print("1. filtered vs unfiltered")
    experiment("vehicle_filtering", "filtered", base, True, BASE_K, jack=("filtered_vs_unfiltered.csv", "mode", "filtered"))
    experiment("vehicle_filtering", "unfiltered", base, False, BASE_K, jack=("filtered_vs_unfiltered.csv", "mode", "unfiltered"))

    print("2. top-k sweep")
    for k in TOP_K_VALUES:
        experiment("top_k", f"k={k}", base, True, k, jack=("top_k_sweep.csv", "top_k", str(k)))

    print("3. chunking sweep")
    for chunk in CHUNKS:
        idx = get_index("pymupdf", BASE_EMBEDDER, chunk, manuals)
        experiment("chunking", f"{chunk[0]}/{chunk[1]} words", idx, True, BASE_K, chunk=chunk,
                   jack=("chunking_sweep.csv", "chunk_config", f"{chunk[0]}/{chunk[1]}"))

    print("4. embedder sweep")
    for emb in EMBEDDERS:
        if emb == "mpnet" and args.skip_mpnet:
            continue
        idx = get_index("pymupdf", emb, BASE_CHUNK, manuals)
        experiment("search_method", emb, idx, True, BASE_K, embedder=emb,
                   jack=("embedder_sweep.csv", "embedder", emb))

    print("5. re-ranking")
    experiment("reranking", "off", base, True, BASE_K, rerank=False, jack=("reranking.csv", "rerank", "False"))
    experiment("reranking", "on", base, True, BASE_K, rerank=True, jack=("reranking.csv", "rerank", "True"))

    print("6. product check (MiniLM, what the app runs)")
    minilm = get_index("pymupdf", "minilm", BASE_CHUNK, manuals)
    experiment("product_check", "MiniLM filtered, clean text", minilm, True, BASE_K, embedder="minilm")
    experiment("product_check", "MiniLM unfiltered, clean text", minilm, False, BASE_K, embedder="minilm")
    minilm_pypdf = None
    if not args.skip_pypdf:
        minilm_pypdf = get_index("pypdf", "minilm", BASE_CHUNK, manuals)
        experiment("product_check", "MiniLM filtered, pypdf text", minilm_pypdf, True, BASE_K,
                   embedder="minilm", text_source="pypdf")
        experiment("product_check", "MiniLM unfiltered, pypdf text", minilm_pypdf, False, BASE_K,
                   embedder="minilm", text_source="pypdf")

    write_csv(out_dir / "retrieval_per_question.csv", per_question)
    write_csv(out_dir / "retrieval_summary.csv", summary)

    # ---- per-vehicle fairness (filtered, Hit@5) -------------------------------
    print("7. per-vehicle fairness")
    fairness = []
    views = [("BM25 (baseline)", "search_method", "bm25", "pymupdf"),
             ("MiniLM, clean text", "search_method", "minilm", "pymupdf"),
             ("MPNet", "search_method", "mpnet", "pymupdf"),
             ("MiniLM, pypdf text (app)", "product_check", "MiniLM filtered, pypdf text", "pypdf")]
    for label, exp, cond, src in views:
        rows = [r for r in per_question if r["experiment"] == exp and r["condition"] == cond and r["text_source"] == src]
        if not rows:
            continue
        by_vehicle = defaultdict(list)
        for r in rows:
            by_vehicle[r["vehicle_id"]].append(r)
        scores = {v: mean(vr, "hit") for v, vr in sorted(by_vehicle.items())}
        fairness.append({"view": label, **scores,
                         "spread": round(max(scores.values()) - min(scores.values()), 3),
                         "sd": round(statistics.pstdev(scores.values()), 3)})
    write_csv(out_dir / "fairness_per_vehicle.csv", fairness)

    # ---- no-vehicle questions -------------------------------------------------
    print("8. no-vehicle questions")
    no_vehicle = []
    for q in [q for q in bank if q["type"] == "ambiguous"]:
        row = {"question_id": q["id"], "question": q["question"]}
        for label, idx in (("bm25", base), ("minilm", minilm)):
            if idx is None:
                continue
            passages, _, top1 = idx.search(q["question"], vehicle_id=None, top_k=BASE_K)
            vehicles = sorted({p["metadata"]["vehicle_id"] for p in passages})
            row[f"{label}_distinct_manuals"] = len(vehicles)
            row[f"{label}_manuals"] = ";".join(vehicles)
            row[f"{label}_top1_score"] = round(top1, 4) if top1 is not None else ""
        no_vehicle.append(row)
    write_csv(out_dir / "no_vehicle_questions.csv", no_vehicle)

    # ---- out-of-scope score separation ----------------------------------------
    print("9. out-of-scope questions")
    scope_rows, scope_summary = [], []
    unsupported = [q for q in bank if q["type"] == "unsupported"]
    for label, idx in (("bm25", base), ("minilm", minilm)):
        if idx is None:
            continue
        groups = {"answerable": [], "unsupported": []}
        for group, qs in (("answerable", questions), ("unsupported", unsupported)):
            for q in qs:
                _, _, top1 = idx.search(q["question"], vehicle_id=q["vehicle_id"], top_k=BASE_K)
                if top1 is None:
                    continue
                groups[group].append(top1)
                scope_rows.append({"embedder": label, "group": group, "question_id": q["id"],
                                   "expected_behaviour": q["expected_behaviour"],
                                   "question": q["question"], "top1_score": round(top1, 4)})
        a, u = groups["answerable"], groups["unsupported"]
        scope_summary.append({
            "embedder": label, "n_answerable": len(a), "n_unsupported": len(u),
            "mean_answerable": round(statistics.mean(a), 3), "mean_unsupported": round(statistics.mean(u), 3),
            "min_answerable": round(min(a), 3), "max_unsupported": round(max(u), 3),
            "unsupported_below_every_answerable": sum(s < min(a) for s in u),
        })
    write_csv(out_dir / "out_of_scope_scores.csv", scope_rows)
    write_csv(out_dir / "out_of_scope_summary.csv", scope_summary)

    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
