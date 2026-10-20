"""
Generates and SAVES answers for every benchmark question (prerequisite for
Tasks 4, 5, 6 and the graded check of Task 7).

Unlike run_generation_eval.py, this keeps the full answer text and the exact
passages the model was given, so answers can be graded afterwards.

Needs Ollama running with the model pulled. The 'product' config also needs
the Chroma index built (python -m src.ingest).

Usage (from repo root):
    python -m eval.run_answers --config minilm_strict --limit 3    # smoke test
    python -m eval.run_answers                                      # all configs
    python -m eval.run_answers --config product

Re-running resumes: questions already in the output file are skipped.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from eval.check_benchmark import load_benchmark
from src.settings import MANUAL_METADATA_FILE, OLLAMA_BASE_URL, OLLAMA_MODEL

eval_dir = Path(__file__).resolve().parent
results_dir = eval_dir / "results" / "extended_benchmark"
configs_path = eval_dir / "answer_configs.json"

# Mirrors EMERGENCY_WORDS in app/app.py, so eval runs see what the app shows.
# app.py cannot be imported here because it starts Streamlit on import.
EMERGENCY_WORDS = (
    "collision", "crash", "accident", "injur", "on fire",
    "smoke", "airbag deployed", "brake failure", "brakes failed",
)
EMERGENCY_ANSWER = ("[withheld by emergency guard] If anyone is hurt or in danger, call "
                    "emergency services (000) first, then roadside assistance.")

STRICT_RULES = """

Additional rules:
9.  If no vehicle is identified and the answer could differ between vehicles,
    ask which vehicle the question is about instead of answering.
10. If the question is about pricing, law, insurance, rental terms or
    anything other than operating this vehicle, say it is outside the scope
    of the owner's manual.
11. If asked to diagnose a fault, do not guess a cause. Say what the manual
    advises and recommend a qualified mechanic or roadside assistance.
12. Never quote a figure from a passage about a different vehicle.
"""


def load_configs() -> list[dict]:
    return json.loads(configs_path.read_text(encoding="utf-8"))["configs"]


def vehicle_names() -> dict:
    manuals = json.loads(MANUAL_METADATA_FILE.read_text(encoding="utf-8"))
    return {m["vehicle_id"]: f"{m['year']} {m['make']} {m['model']}" for m in manuals}


def system_prompt(name: str) -> str:
    from src.generator import SYSTEM_PROMPT
    if name == "baseline":
        return SYSTEM_PROMPT
    if name == "strict":
        return SYSTEM_PROMPT + STRICT_RULES
    raise ValueError(f"unknown prompt {name!r}")


def call_ollama(system: str, user: str, temperature: float) -> str:
    response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", timeout=300, json={
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": temperature},
    })
    response.raise_for_status()
    return response.json()["message"]["content"]


class ProductPipeline:
    """Exactly what the app does: src.rag.ask."""

    def answer(self, question: str, vehicle_id: str | None, vehicle_name: str | None):
        from src.rag import ask
        result = ask(question=question, vehicle_id=vehicle_id, vehicle_name=vehicle_name)
        passages = [{"n": s["source_number"], "vehicle_id": s["vehicle_id"], "pdf_page": s["pdf_page"],
                     "distance": round(s["distance"], 4), "text": s["text"]} for s in result["sources"]]
        return result["answer"], passages


class EvalIndexPipeline:
    """Jack's EvalIndex for retrieval, the product's prompt format for generation."""

    def __init__(self, config: dict):
        from eval.index import EvalIndex, load_manuals
        self.config = config
        self.index = EvalIndex(config["embedder"], config["chunk_size"], config["overlap"])
        self.index.build(load_manuals())

    def answer(self, question: str, vehicle_id: str | None, vehicle_name: str | None):
        from src.generator import build_context
        passages, _, _ = self.index.search(question, vehicle_id=vehicle_id, top_k=self.config["top_k"])
        if not passages:
            return "No relevant owner's-manual information was retrieved.", []
        user = (f"Vehicle:\n{vehicle_name or 'Vehicle not explicitly specified'}\n\n"
                f"Retrieved owner's-manual passages:\n\n{build_context(passages)}\n\n"
                f"Customer/support question:\n{question}\n\n"
                "Answer the question using only the retrieved passages.")
        answer = call_ollama(system_prompt(self.config["prompt"]), user, self.config["temperature"])
        saved = [{"n": i, "vehicle_id": p["metadata"]["vehicle_id"], "pdf_page": p["metadata"]["pdf_page"],
                  "text": p["text"]} for i, p in enumerate(passages, start=1)]
        return answer, saved


def run(config: dict, questions: list[dict]) -> None:
    out_path = results_dir / f"answers_{config['id']}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [q for q in questions if q["id"] not in done]
    print(f"\n[{config['id']}] {len(done)} already done, {len(todo)} to run")
    if not todo:
        return

    pipeline = ProductPipeline() if config["pipeline"] == "product" else EvalIndexPipeline(config)
    names = vehicle_names()
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(out_path, "a", encoding="utf-8") as out:
        for i, q in enumerate(todo, start=1):
            vehicle_id = q.get("vehicle_id")
            started = time.perf_counter()
            guard = config.get("emergency_guard") and any(w in q["question"].lower() for w in EMERGENCY_WORDS)
            if guard:
                answer, passages = EMERGENCY_ANSWER, []
            else:
                try:
                    answer, passages = pipeline.answer(q["question"], vehicle_id, names.get(vehicle_id))
                except (requests.RequestException, RuntimeError) as error:
                    print(f"\n  stopped: could not reach Ollama or the index ({error})")
                    return
            record = {
                "id": q["id"], "type": q["type"], "category": q.get("category"),
                "expected_behaviour": q["expected_behaviour"], "vehicle_id": vehicle_id,
                "question": q["question"], "config": config["id"], "answer": answer,
                "passages": passages, "emergency_guard": bool(guard),
                "seconds": round(time.perf_counter() - started, 2), "model": OLLAMA_MODEL,
                "temperature": 0.1 if config["pipeline"] == "product" else config.get("temperature"),
                "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"  [{i}/{len(todo)}] {q['id']} {record['seconds']}s", end="\r", flush=True)
    print(f"\n  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Run one config id from answer_configs.json.")
    parser.add_argument("--limit", type=int, help="Only the first N questions (smoke test).")
    arguments = parser.parse_args()

    configs = load_configs()
    if arguments.config:
        configs = [c for c in configs if c["id"] == arguments.config]
        if not configs:
            raise SystemExit(f"no config {arguments.config!r} in answer_configs.json")

    questions = load_benchmark()
    if arguments.limit:
        questions = questions[:arguments.limit]

    for config in configs:
        run(config, questions)


if __name__ == "__main__":
    main()
