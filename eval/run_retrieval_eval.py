import csv
import json
from pathlib import Path

from eval.index import EvalIndex, load_manuals
from eval.metrics import(correct_manual_rate, keyword_hit, mean, ncdcg, page_hit, top1_correct_manual)

eval_dir = Path(__file__).resolve().parent
results_dir = eval_dir / "results"
questions_path = eval_dir / "questions.json"

baseline_chunk_size = 220
baseline_overlap = 40
baseline_top_k = 5
baseline_embedder = "bm25"

chunk_configs = [(120, 20), (220, 40), (350, 60)]
top_k_values = [1, 3, 5, 8]
embedders_to_try = ["bm25", "minilm", "mpnet"]

def load_questions() -> list[dict]:
    return json.loads(questions_path.read_text(encoding="utf-8"))

def scope_questions(questions: list[dict]) -> list[dict]:
    result = []
    for question in questions:
        if question["category"] in ("common_topic", "unique_topic"):
            result.append(question)
    return result


def write_csv(name: str, rows: list[dict]) -> None:
    results_dir.mkdir(exist_ok=True, parents=True)
    out_path = results_dir / name

    if not rows:
        print("no rows to write for " + name)
        return

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f" wrote eval/results/{name}")


def run_question(
            index: EvalIndex, question: dict, vehicle_id: str | None,top_k: int, 
            rerank: bool) -> dict: 
    passages, latency_ms, top1_score = index.search(
        question["question"], vehicle_id = vehicle_id, top_k=top_k, rerank=rerank)

    expected_vehicle_id = question.get("vehicle_id")
    expected_keywords = question.get("expected_keywords", [])
    page_hint = question.get("page_hint")

    ndcg = ncdcg(passages, expected_vehicle_id, expected_keywords, page_hint, k=top_k)
                            
    row = {
        "question_id": question["id"],
        "expected_vehicle_id": expected_vehicle_id,
        "filtered": vehicle_id is not None,
        "keyword_hit": keyword_hit(passages, expected_keywords),
        "page_hit": page_hit(passages, page_hint),
        "ndcg": round(ndcg, 4) if ndcg is not None else None,
        "correct_manual_rate": None,
        "top1_correct_manual": None,
        "top1_score": round(top1_score, 4) if top1_score is not None else None,
        "latency_ms": round(latency_ms, 2),
        "num_passages": len(passages)
    }

    if expected_vehicle_id:
        row["correct_manual_rate"] = round(correct_manual_rate(passages, expected_vehicle_id), 2)
        row["top1_correct_manual"] = top1_correct_manual(passages, expected_vehicle_id)

    return row

def summarise(rows: list[dict], group_key:str) -> None:
    groups = set()
    for row in rows:
        groups.add(row[group_key])
    groups = sorted(groups, key=str)

    for group in groups:
        subset = []
        for row in rows:
            if row[group_key] == group:
                subset.append(row)

        keyword_hits = []
        page_hits = []
        ndcg_scores = []
        manual_hits = []
        latencies = []

        for row in subset:
            keyword_hits.append(1.0 if row["keyword_hit"] else 0.0)
            latencies.append(row["latency_ms"])

            if row["page_hit"] is not None:
                page_hits.append(1.0 if row["page_hit"] else 0.0)

            if row["ndcg"] is not None:
                ndcg_scores.append(row["ndcg"])

            if row["top1_correct_manual"] is not None:
                manual_hits.append(1.0 if row["top1_correct_manual"] else 0.0)

        keyword_rate = mean(keyword_hits)
        page_rate = mean(page_hits)
        ndcg_rate = mean(ndcg_scores)
        manual_rate = mean(manual_hits)
        latency = mean(latencies)

        print(str(group) + 
              "\nkeyword_hit=" + str(round(keyword_rate, 2)) +
              "\npage_hit=" + str(round(page_rate, 2)) +
              "\nndcg=" + str(round(ndcg_rate, 2)) + 
              "\ntop1_correct_manuals=" + str(round(manual_rate, 2)) +
              "\nlatency_ms=" + str(round(latency, 1)))


#experiemnt filtered vs unfiltered
def filtered_vs_unfiltered(questions: list[dict], baseline_index: EvalIndex) -> None:
    print("\nFiltered vs unfiltered retrieval")

    rows = []
    for question in scope_questions(questions):
        filtered_row = run_question(baseline_index, question, question["vehicle_id"], baseline_top_k, rerank=False)
        filtered_row["mode"] = "filtered"
        rows.append(filtered_row)

        unfiltered_row = run_question(baseline_index, question, None, baseline_top_k, rerank=False)
        unfiltered_row["mode"] = "unfiltered"
        rows.append(unfiltered_row)

    write_csv("filtered_vs_unfiltered.csv", rows)
    summarise(rows, "mode")


def top_k_sweep(questions: list[dict], baseline_index: EvalIndex) -> None:
    print("\n Top-K sweep")

    rows = []
    for top_k in top_k_values:
        for question in scope_questions(questions):
            row = run_question(baseline_index, question, question["vehicle_id"], top_k, rerank=False)
            row["top_k"] = top_k
            rows.append(row)

    write_csv("top_k_sweep.csv", rows)
    summarise(rows, "top_k")


def chunking_sweep(questions: list[dict], manuals: list[dict]) -> None:
    print("Chunk size / overlap sweep")

    rows = []
    for chunk_size, overlap in chunk_configs:
        label = f"{chunk_size}/{overlap}"

        index = EvalIndex(baseline_embedder, chunk_size, overlap)
        index.build(manuals)

        for question in scope_questions(questions):
            row = run_question(index, question, question["vehicle_id"], baseline_top_k, rerank=False)
            row["chunk_config"] = label
            rows.append(row)

    write_csv("chunking_sweep.csv", rows)
    summarise(rows, "chunk_config")


def embedder_sweep(questions: list[dict], manuals: list[dict]) -> None:
    print("Embedding backedn sweep")

    rows = []
    for embedder_name in embedders_to_try:
        try:
            index = EvalIndex(embedder_name, baseline_chunk_size, baseline_overlap)
            index.build(manuals)
        except Exception as error:
           print(f" skipped ({error.__class__.__name__}: {error})")

        for question in scope_questions(questions):
            row = run_question(index, question, question["vehicle_id"], baseline_top_k, rerank=False)
            row["embedder"] = embedder_name
            rows.append(row)

    write_csv("embedder_sweep.csv", rows)

    if rows:
        summarise(rows, "embedder")
    else:
        print("no embedders could be built")


def re_ranking(questions: list[dict], manuals: list[dict]) -> None:
    print("Reranking on vs off")

    index = EvalIndex("bm25", baseline_chunk_size, baseline_overlap)
    index.build(manuals)

    rows = []
    for rerank in (False, True):
        for question in scope_questions(questions):
            row = run_question(index, question, question["vehicle_id"], baseline_top_k, rerank=rerank)
            row["rerank"] = rerank
            rows.append(row)

    write_csv("reranking.csv", rows)
    summarise(rows, "rerank")


def random_vehicle(questions: list[dict], baseline_index: EvalIndex) -> None:
    print("Random vehicle questions")

    rows = []
    for question in questions:
        if question["category"] != "random_vehicle":
            continue

        passages, latency_ms, top1_score = baseline_index.search(question["question"], vehicle_id=None, top_k=baseline_top_k, rerank=False)

        vehicles_seen = set()
        for passage in passages:
            vehicles_seen.add(passage["metadata"]["vehicle_id"])
        vehicles_seen = sorted(vehicles_seen)

        rows.append(
            {
                "question_id": question["id"],
                "question": question["question"],
                "distinct_vehicles_top5": len(vehicles_seen),
                "vehicles_in_top5": ";".join(vehicles_seen),
                "top1_score": round(top1_score, 4) if top1_score is not None else None,
            }
        )
        print(f" {question['id']}: top 5 pulled from {len(vehicles_seen)} different manuals. ({', '.join(vehicles_seen)})")

    write_csv("random_vehicle.csv", rows)



def out_of_scope(questions: list[dict], baseline_index: EvalIndex) -> None:
    print("Out of scope questions")

    rows = []
    for question in questions:
        if question["category"] != "out_of_scope":
            continue

        passages, latency_ms, top1_score = baseline_index.search(question["question"], vehicle_id=question.get("vehicle_id"), top_k=baseline_top_k, rerank=False)

        rows.append(
            {
                "question_id": question["id"],
                "question": question["question"],
                "top1_score": round(top1_score, 4) if top1_score is not None else None
            }
        )

    write_csv("out_of_scope.csv", rows)

    in_scope_scores = []
    baseline_rows_path = results_dir / "filtered_vs_unfiltered.csv"
    if baseline_rows_path.exists():
        with open(baseline_rows_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["mode"] == "filtered" and r["top1_score"]:
                    in_scope_scores.append(float(r["top1_score"]))


    out_of_scope_scores = []
    for row in rows:
        if row["top1_score"] is not None:
            out_of_scope_scores.append(row["top1_score"])

    if in_scope_scores and out_of_scope_scores:
        print(f" mean top1_score: \nin scope = {mean(in_scope_scores):.4f} \nout of scope = {mean(out_of_scope_scores):.4f}")


def main() -> None:
    questions = load_questions()
    manuals = load_manuals()

    baseline_index = EvalIndex(baseline_embedder, baseline_chunk_size, baseline_overlap)
    baseline_index.build(manuals)

    filtered_vs_unfiltered(questions, baseline_index)
    top_k_sweep(questions, baseline_index)
    chunking_sweep(questions, manuals)
    embedder_sweep(questions, manuals)
    re_ranking(questions, manuals)
    random_vehicle(questions, baseline_index)
    out_of_scope(questions, baseline_index)

    print("done, all results are in eval/results/.")


if __name__ == "__main__":
    main()