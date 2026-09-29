"""Browser interface for the Rental Fleet RAG assistant.

Run:  python -m streamlit run app.py

Logs every query to logs/queries.jsonl and ratings to logs/feedback.jsonl.
"""
import json
import time
import uuid
from datetime import datetime

import streamlit as st

from src.rag import ask
from src.settings import MANUAL_METADATA_FILE, PROJECT_ROOT

LOG_DIR = PROJECT_ROOT / "app" / "logs"
LOG_DIR.mkdir(exist_ok=True)
QUERY_LOG = LOG_DIR / "queries.jsonl"
FEEDBACK_LOG = LOG_DIR / "feedback.jsonl"

NO_CHOICE = ""
EMERGENCY_WORDS = (
    "collision", "crash", "accident", "injur", "on fire",
    "smoke", "airbag deployed", "brake failure", "brakes failed",
)
LOW_MATCH_DISTANCE = None


@st.cache_data
def load_vehicles() -> dict:
    with open(MANUAL_METADATA_FILE, encoding="utf-8") as f:
        return {m["vehicle_id"]: m for m in json.load(f)}


def append(path, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


st.set_page_config(page_title="Fleet Manual Assistant", layout="wide")
st.title("Rental Fleet Manual Assistant")

vehicles = load_vehicles()
labels = {vid: f"{v['make']} {v['model']} ({v['year']})" for vid, v in vehicles.items()}

with st.sidebar:
    st.header("Test session")
    participant = st.text_input("Participant ID", value="P01")
    task_id = st.text_input("Task ID", value="T01")
    allow_all = st.checkbox("Allow all-vehicle search (testing only)")

options = [NO_CHOICE] + ([None] if allow_all else []) + list(labels)


def fmt(x):
    if x == NO_CHOICE:
        return "Select a vehicle..."
    if x is None:
        return "All vehicles (testing only)"
    return labels[x]


choice = st.selectbox("Vehicle", options=options, format_func=fmt)
question = st.text_area("Customer question", height=90)

if st.button("Get answer", type="primary"):
    q = question.strip()
    if not q:
        st.warning("Please enter a question.")
    elif choice == NO_CHOICE:
        st.warning("Select a vehicle first. Without one, every manual is "
                   "searched and the wrong vehicle's manual is often used.")
    else:
        record = {
            "query_id": uuid.uuid4().hex[:8],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "participant": participant,
            "task": task_id,
            "vehicle_selected": choice,
            "question": q,
        }
        if any(w in q.lower() for w in EMERGENCY_WORDS):
            record.update(answer="[withheld by emergency guard]", seconds=0.0,
                          sources=[], guard="emergency")
        else:
            name = None
            if choice:
                v = vehicles[choice]
                name = f"{v['year']} {v['make']} {v['model']}"
            try:
                with st.spinner("Searching manuals..."):
                    start = time.perf_counter()
                    result = ask(question=q, vehicle_id=choice, vehicle_name=name)
                    seconds = round(time.perf_counter() - start, 2)
            except RuntimeError as err:
                st.error(f"{err}")
                st.stop()
            record.update(
                answer=result["answer"],
                seconds=seconds,
                sources=[
                    {"n": s["source_number"], "vehicle_id": s["vehicle_id"],
                     "pdf_page": s["pdf_page"], "distance": s["distance"],
                     "text": s["text"]}
                    for s in result["sources"]
                ],
            )
        append(QUERY_LOG, record)
        st.session_state["last"] = record

last = st.session_state.get("last")
if last:
    if last.get("guard") == "emergency":
        st.error("If anyone is hurt or in danger, call emergency services (000) "
                 "first, then roadside assistance. This tool cannot advise on a "
                 "live incident.")
    else:
        st.subheader("Answer")
        st.write(last["answer"])
        st.caption(f"Response time: {last['seconds']} s")

        if (LOW_MATCH_DISTANCE is not None and last["sources"]
                and min(s["distance"] for s in last["sources"]) > LOW_MATCH_DISTANCE):
            st.warning("Low match: the manual may not cover this question. "
                       "Verify before relying on the answer.")

        st.subheader("Supporting sources")
        for s in last["sources"]:
            with st.expander(f"[Source {s['n']}] {s['vehicle_id']} - page {s['pdf_page']}"):
                st.caption(f"Vector distance: {s['distance']:.4f}")
                st.write(s["text"])

        st.divider()
        st.write("Was the answer correct? (for testing)")
        for col, verdict in zip(st.columns(3),
                                ["correct", "incorrect", "refused_insufficient"]):
            if col.button(verdict.replace("_", " ").title(), key=verdict):
                append(FEEDBACK_LOG, {
                    "query_id": last["query_id"], "verdict": verdict,
                    "timestamp": datetime.now().isoformat(timespec="seconds")})
                st.success("Saved.")