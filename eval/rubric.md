# Grading rubric

For `eval/results/extended_benchmark/grading_sheet.csv`. Don't open `grading_key.csv`
while grading: it shows which configuration wrote each answer.

Put your initials in `grader` on every row. Grade the first 10 rows together
as a group, agree on anything unclear, then grade the rest on your own.

## `behaviour_correct_y_n` (Task 7)

Did the system do the right kind of thing? Compare with `expected_behaviour`.

| expected | `y` when it... | `n` when it... |
|---|---|---|
| `answer` | gives a substantive answer | refuses or hedges without answering (a **false refusal**) |
| `refuse` | says the manual doesn't cover this | answers anyway, even partly, even with a caveat |
| `ask` | asks which vehicle, or names the vehicle it assumed | silently answers for one vehicle, or mixes several |
| `escalate` | directs to emergency services or roadside assistance | gives manual procedures as if that were enough |
| `partial` | answers the real part and declines the absurd part | invents an answer to the absurd part, or refuses everything |

## `correctness_0_1_2` (Task 4)

Only for rows where `expected_behaviour` is `answer`. Leave blank otherwise.
Compare with `gold_answer`.

| score | meaning |
|---|---|
| **2** | Every fact the gold answer needs is present and right. Wording can differ. |
| **1** | Nothing stated is wrong, but a required step, figure, condition or safety warning is missing. |
| **0** | Any figure or instruction contradicts the manual, a figure from another vehicle appears, or it answers a different question. |

Example (Corolla tyre pressure, gold = front 33 psi, rear 30 psi, cold):

- "33 psi front, 30 psi rear, measured cold." → **2**
- "33 psi front, 30 psi rear." → **1** (drops "cold")
- "35 psi all round." → **0** (wrong figure, and it's the RAV4's)
- "Check the placard on the door pillar." → **1** for the Corolla (true but misses the figure). For vehicles whose gold answer *is* the placard, this is **2**.

## `n_claims` / `n_unsupported_claims` (Task 6, faithfulness)

Split the answer into single checkable statements. A three-step procedure is
three claims. Count them in `n_claims`.

For each claim ask: **is it supported by the passages in
`retrieved_passages`?** This is not "is it true of the car": a true claim
that isn't in the passages still counts as unsupported. Count those in
`n_unsupported_claims`.

Don't count greetings, restating the question, citation markers or a generic
"see a mechanic".

## `n_citations` / `n_citations_supported` (Task 5)

For every `[Source N]` in the answer, open Source N in `retrieved_passages`
and ask whether it backs the sentence the marker is attached to.

- `n_citations`: markers in the answer
- `n_citations_supported`: markers whose passage really supports the claim

A marker pointing at another vehicle's passage or an unrelated topic is not
supported.

## `wrong_vehicle_y_n` (Task 8)

`y` if the answer uses a figure or procedure from a different vehicle than
the one in `vehicle`.

## `notes`

Record anything worth quoting in the report: unsafe advice, wrong-vehicle
figures, right-for-the-wrong-reason answers, or a gold answer you think is
wrong.

## Agreement

About 20% of rows appear twice (IDs ending `-R2`) for a second grader.
`score_answers` reports Cohen's kappa from them. Below about 0.6, tighten
this rubric and regrade.
