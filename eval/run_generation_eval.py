import csv
import json
import re
from pathlib import Path
import requests

from eval.index import EvalIndex, load_manuals
from src.generator import SYSTEM_PROMPT as BASELINE_SYSTEM_PROMPT
from src.generator import build_context
from src.settings import OLLAMA_BASE_URL, OLLAMA_MODEL

eval_dir = Path(__file__).resolve().parent
results_dir = eval_dir / "results"
questions_path = eval_dir / "questions.json"

baseline_chunk_size = 220
baseline_overlap = 40
baseline_top_k = 5
retrieval_embedder = "bm25"

#add second model after ollama pull to compare llms
models_to_try = [OLLAMA_MODEL]

llm_only_prompt = ("Answer using your own knowledge")

prompt_variants = {
    "baseline": BASELINE_SYSTEM_PROMPT,
    "extended": ("Answer the question using only the provided manuals, if they dont contain the answer say so, and ensure you cite your sources")
}

refusal_phrases = ("insufficient", "not covered", "do not contain", "don't contain",
                   "does not contain", "no information", "cannot find", "can't find",
                   "not available in the manual", " not enough information", "unable to answer")

citation_pattern = re.compile(r"\[source\s*(\d+)\]", re.IGNORECASE)

def load_questions() -> list[dict]:
    return json.loads(questions_path.read_text(encoding="utf-8"))

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

def call_ollama(system_prompt: str, user_prompt: str, model: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream":False,
        "options": {"temperature": 0.1}
    }

    response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=300)
    response.raise_for_status()

    return response.json()["message"]["content"]


def build_user_prompt(question: dict, context: str) -> str:
    return (
        f"Vehicle:\n{question.get('vehicle_id') or 'Not specified'}\n"
        f"Retrieved manual passages:\n{context}\n"
        f"Support question:\n{question['question']}\n"
        "Answer the question using only the retrieved passages"
    )

def grade_answer(answer: str, question: dict, num_passages_given: int) -> dict:
    lowered = answer.lower()

    refusal_detected = False
    for phrase in refusal_phrases:
        if phrase in lowered:
            refusal_detected = True
            break

    is_out_of_scope = question["category"] == "out_of_scope"
    if is_out_of_scope:
        #refusuing to answer if question is out of scope
        refusal_correct = refusal_detected
    else:
        refusal_correct = not refusal_detected

    expected_keywords = question.get("expected_keywords", [])
    keyword_hit = None
    if expected_keywords:
        keyword_hit = False
        for keyword in expected_keywords:
            if keyword.lower() in lowered:
                keyword_hit = True
                break

    citation_numbers = []
    for match in citation_pattern.findall(answer):
        citation_numbers.append(int(match))

    citation_present = len(citation_numbers) > 0

    citation_in_range = None
    if citation_numbers:
        citation_in_range = True
        for number in citation_numbers:
            if number < 1 or number > num_passages_given:
                citation_in_range = False
                break

    return {
        "question_id": question["id"],
        "category": question["category"],
        "refusal_detected": refusal_detected,
        "refusal_correct": refusal_correct,
        "keyword_hit": keyword_hit,
        "citation_present": citation_present,
        "citation_in_range": citation_in_range,
        "answers_chars": len(answer)
    }

def build_retrieval_index(manuals: list[dict]) -> EvalIndex:
    index = EvalIndex(retrieval_embedder, baseline_chunk_size, baseline_overlap)
    index.build(manuals)
    return index

#experiment prompt and llm variations
def experiment_prompt_llm(questions:list[dict], index: EvalIndex) -> None:
    print("\nPrompt and LLM variations")
    rows = []

    for model in models_to_try:
        for prompt_name, system_prompt in prompt_variants.items():
            print("Running Model = " + model + ", prompt = " + prompt_name)

            for question in questions:
                passages, latency_mls, top_1 = index.search(question["question"], vehicle_id=question.get("vehicle_id"), top_k = baseline_top_k)
                if not passages:
                    continue

                context = build_context(passages)
                user_prompt = build_user_prompt(question, context)

                try: 
                    answer = call_ollama(system_prompt, user_prompt, model)
                except requests.RequestException as error:
                    print(f" could not reach Ollam ({error})")
                    return

                row = grade_answer(answer, question, len(passages))
                row["model"] = model
                row["prompt_variant"] = prompt_name
                rows.append(row)

    write_csv("prompt_and_llm_variations.csv", rows)


#experiment rag vs llm ony
def rag_vs_llm(questions: list[dict], index: EvalIndex) -> None:
    print("RAG vs LLM only")

    rows = []
    model = models_to_try[0]

    for question in questions:
        passages, latency_mls, top_1 = index.search(question["question"], vehicle_id=question.get("vehicle_id"), top_k = baseline_top_k)

        #rag
        if passages:
            context = build_context(passages)
            rag_user_prompt = build_user_prompt(question, context)
    
            try: 
                rag_answer = call_ollama(BASELINE_SYSTEM_PROMPT, rag_user_prompt, model)
            except requests.RequestException as error:
                print(f" could not reach Ollam ({error})")
                return
    
            rag_row = grade_answer(rag_answer, question, len(passages))
            rag_row["mode"] = "rag"
            rows.append(rag_row)

        #llm only
        try:
            llm_answer = call_ollama(llm_only_prompt, question["question"], model)
        except requests.RequestException as error:
            print(f" could not reach Ollam ({error})")
            return

        llm_row = grade_answer(llm_answer, question, num_passages_given=0)
        llm_row["mode"] = "llm_only"
        rows.append(llm_row)

    write_csv("rag_vs_llm_only.csv", rows)


def main() -> None:
    questions = load_questions()
    manuals = load_manuals()

    print(f"Loaded {len(questions)} benchmark questions")
    index = build_retrieval_index(manuals)

    experiment_prompt_llm(questions, index)
    rag_vs_llm(questions, index)

    print("results are in eval/results/.")

if __name__ == "__main__":
    main()