# Evaluation: results, method and reproduction

This extends retrieval experiments in `eval/`. Everything here writes to `eval/results/extended_benchmark/`. All commands run from the repo root.

## Headline results

**Retrieval** (33 answerable questions, vehicle filter on, Recall@5 = the
supporting page is among the top five passages):

| Configuration | Recall@5 | MRR |
|---|---|---|
| BM25 keyword search (the original baseline) | 73% | 0.48 |
| MiniLM semantic search (the app's model) | **100%** | **0.92** |
| MPNet semantic search | 100% | 0.89 |
| **The app as built** (MiniLM on pypdf text) | **79%** | 0.62 |

The app scores lower because pypdf turns both Kia manuals into unreadable
text: Recall@5 is 0% for the Cerato and 25% for the Sportage, against 100% for
every other vehicle. Without the vehicle filter, MiniLM finds the supporting
page for only 39% of questions.

**Answers** (graded against the expected answers with `rubric.md`):

| Measure | App as built | MiniLM, strict prompt (selected) |
|---|---|---|
| Correct or partially correct | 61% | **79%** |
| Contains a wrong figure or instruction | 39% | **21%** |
| Fully correct | 3% | 6% |
| Claims supported by the passages (faithfulness) | 79% | **91%** |
| Answers with an unsupported claim | 38% | **24%** |
| Citations supporting their claim (target 90%) | 72% | 81% |
| Answerable questions declined | 15% | 6% |
| Out-of-scope questions declined (target 70%) | 100% | 100% |
| Asked which vehicle when none was named | 0 of 5 | 1 of 5 |
| Answers using another vehicle's information | 0% | 0% |
| Decision score (weights in `answer_configs.json`) | 0.67 | **0.75** |


## Status of the nine tasks

| Task | Status | Where |
|---|---|---|
| 1 Question bank | Done: 50 questions | `benchmark.json`, `check_benchmark.py` |
| 2 Ground truth | Done: all 25 verified against the PDF pages | `benchmark.json` (`gold`), `find_pages.py` |
| 3 Retrieval accuracy / Top-K | Done | `rebuild_aligned.py` |
| 4 Answer correctness | Done: 100 answers graded, 20% second-graded blind (κ = 0.79) | `run_answers.py`, `rubric.md`, `score_answers.py` |
| 5 Source correctness | Done: 81% of citations support their claim (target 90%) | as above |
| 6 Faithfulness | Done: 91% of claims supported | as above |
| 7 Unsupported handling | Done (automatic and graded) | `score_answers.py` |
| 8 Cross-model errors | Done | `rebuild_aligned.py` (unfiltered runs) |
| 9 Final configuration | Done: MiniLM, 220/40 chunks, top 5, filter on, strict prompt | `score_answers.py`, `answer_configs.json` |

## Files

| File | Purpose |
|---|---|
| `benchmark.json` | 50 questions with expected behaviour, expected answers and supporting pages |
| `check_benchmark.py` | Validates the bank: structure, paraphrase links, every quote on its page |
| `find_pages.py` | Searches the manuals for a phrase, to locate supporting pages |
| `rebuild_aligned.py` | Jack's retrieval experiments re-run on the extended benchmark, on clean and pypdf text |
| `score_retrieval_gold.py` | Helper imported by `rebuild_aligned.py` (question selection, pypdf text) |
| `make_appendix_tables.py` | Writes the LaTeX appendix sections |
| `answer_configs.json` | Answer-generation configurations and the pre-registered decision weights |
| `run_answers.py` | Generates and saves answers with the passages the model saw (resumable) |
| `make_grading_sheet.py` | Blind, shuffled grading sheet, with 20% duplicated for a second grader |
| `apply_draft_grades.py` | Moves draft grades into the sheet; writes and merges the verification file |
| `rubric.md` | Grading rules |
| `score_answers.py` | Automatic and graded metrics, Cohen's kappa, decision rule |
| `results/extended_benchmark/` | Answers (`answers_*.jsonl`), grades and `answer_scores.csv` |
| `results/extended_benchmark/aligned/` | Retrieval results (`retrieval_summary.csv`, `retrieval_per_question.csv`, `fairness_per_vehicle.csv`, `no_vehicle_questions.csv`, `out_of_scope_*.csv`), `appendix_tables.tex`, `report_paragraphs.tex` |

## How the results line up with the report

On the report's own 17 in-scope questions, `rebuild_aligned` reproduces the
report exactly: correct manual first in 100% of filtered and 17.6% of
unfiltered searches; 3, 4 and 4 manuals for the no-vehicle questions; and
out-of-scope search scores identical to four decimal places. The supporting-page
measures (Recall@K, MRR) are added alongside the report's measures rather than
replacing them.


## Ground-truth corrections

Several figures sit in specification tables whose text extracts without its
column headings, so they were read from the rendered PDF page. Eight expected
answers were corrected; each carries a `check_note`:

- Q31 Mazda 3, Q32 CX-5, Q33 i30, Q35 Cerato and Q36 Sportage said the
  manual does not print tyre pressures. All five do, in the specifications
  chapter (p.707, p.801, p.634, p.596 and p.560).
- Q30 RAV4 listed 35 and 36 psi only; the 17- to 19-inch variants use 33, 35
  or 36 psi (p.501-504).
- Q34 Tucson: maximum-load pressure is 240 kPa front, 275 kPa rear.
- Q11 Mazda 3 now states the service intervals. This is the European manual:
  a maximum of 20,000 km or 12 months, and every 10,000 km under severe use.

## Limitations

- **Small samples.** 33 answerable questions, 4-5 per vehicle. One question
  moves a vehicle's score by 20-25 points, and differences of one or two
  questions (for example between chunk sizes) are not meaningful.
- **Temperature.** The app runs at 0.1 (fixed in `src/generator.py`); the
  alternative configurations ran at 0, so their answers are reproducible but
  the app's may vary slightly between runs.
- **The rubric is strict.** An answer that omits one condition the manual gives
  scores 1, which is why few answers are fully correct.

## Grading files

- `draft_grades.json`: the first grader's grades, keyed `<config>::<question>`.
- `grading_sheet.csv`: the blind sheet (first grades in the main rows,
  second grades in the `-R2` rows); `grading_key.csv` maps rows to configs.
- `grading_verify.csv`: the 20 `-R2` rows alone, as graded by the second grader.
- `answer_scores.csv`: every automatic and graded measure, per configuration.

To have a person regrade the 20% instead, clear the grade columns in
`grading_verify.csv`, grade it with `rubric.md` without opening the files
above, then run `python -m eval.apply_draft_grades merge` and
`python -m eval.score_answers`.

## Reproducing

```
python -m eval.check_benchmark                          # tasks 1-2
python -m eval.find_pages "tyre pressure" --vehicle kia_cerato
python -m eval.rebuild_aligned                          # tasks 3, 8, incl. the app's pypdf text
python -m src.ingest                                    # once, for the product config
python -m eval.run_answers                              # needs Ollama + gemma3:4b; resumable
python -m eval.score_answers                            # automatic metrics
python -m eval.make_grading_sheet --configs product minilm_strict
python -m eval.apply_draft_grades fill
python -m eval.score_answers                            # graded metrics, task 9
python -m eval.make_appendix_tables
```

Generating the 200 answers takes about 25-30 seconds each on a laptop CPU.
