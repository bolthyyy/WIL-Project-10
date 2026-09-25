Retrieval and generation evaluation

questions.json has benchmark questions for both scripts
extract_cache.py caches manual PDF text to eval/cache/
embedders.py bm25, minilm / mpnet embedding backends
index.py in memeory retrieval index used
reranker.py TF-IDF based reranker
metrics.py keyword_hit, page_hit, ndcg, correct_manual_rate
run_retrieval_eval.py retrieval experiments
run_generation_eval.py generation experiments (ollama)
requirements.txt
results/

How to run:

Run from repo root

.venv\Scripts\Activate.ps1

pip install -r eval/requirements.txt
python -m eval.extract_cache
python -m eval.run_retrieval_eval

ollama pull gemma3:4b
python -m eval.run_generate_eval