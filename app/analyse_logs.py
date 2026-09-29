"""Analyse logs produced by app.py.

  python analyse_logs.py                      # latency, verdicts, auto-flagged failures
  python analyse_logs.py --value --saved 1 2 3 --volume 2000 5000 8000 --rate 40

--saved   minutes saved per query (low expected high), from timed sessions
--volume  queries per year (low expected high), an ASSUMPTION, must justify
--rate    hourly staff cost, from a cited public wage source
"""
import argparse
import json
import re
import statistics as st
from pathlib import Path

LOGS = Path("app/logs")


def read(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def flags(q):
    out = []
    sel, srcs, ans = q["vehicle_selected"], q["sources"], q["answer"]
    if sel and any(s["vehicle_id"] != sel for s in srcs):
        out.append("wrong_manual_retrieval")
    if not sel:
        out.append("no_vehicle_selected(ambiguous)")
    cited = [int(n) for n in re.findall(r"\[Source (\d+)\]", ans)]
    if any(n < 1 or n > len(srcs) for n in cited):
        out.append("incorrect_citation(out_of_range)")
    if not cited and "insufficient" not in ans.lower():
        out.append("no_citation(check_unsupported)")
    return out


def report():
    queries = read(LOGS / "queries.jsonl")
    verdicts = {f["query_id"]: f["verdict"] for f in read(LOGS / "feedback.jsonl")}
    if not queries:
        print("No queries logged yet.")
        return
    secs = [q["seconds"] for q in queries]
    print(f"Queries: {len(queries)}  |  latency mean {st.mean(secs):.1f}s, "
          f"median {st.median(secs):.1f}s, max {max(secs):.1f}s")
    if verdicts:
        counts = {}
        for v in verdicts.values():
            counts[v] = counts.get(v, 0) + 1
        print("Verdicts:", counts)
    print("\nAuto-flagged (manual tagging still needed for hallucination,")
    print("irrelevant passages, unsupported answers):\n")
    for q in queries:
        f = flags(q)
        if f:
            print(f"{q['query_id']}  {q['question'][:60]!r}  -> {', '.join(f)}")


def value(saved, volume, rate):
    print(f"{'Scenario':<10}{'min/query':>10}{'queries/yr':>12}{'hours/yr':>10}{'value/yr':>12}")
    for name, s, v in zip(["Low", "Expected", "High"], saved, volume):
        hours = s / 60 * v
        print(f"{name:<10}{s:>10.1f}{v:>12,.0f}{hours:>10,.0f}{hours * rate:>12,.0f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--value", action="store_true")
    p.add_argument("--saved", nargs=3, type=float)
    p.add_argument("--volume", nargs=3, type=float)
    p.add_argument("--rate", type=float)
    a = p.parse_args()
    if a.value:
        if not (a.saved and a.volume and a.rate):
            p.error("--value needs --saved, --volume and --rate")
        value(a.saved, a.volume, a.rate)
    else:
        report()