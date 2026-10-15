"""Read-only diagnostic of estimated-frontier opportunities; no gold or model calls."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oec.frontier_scheduler import estimated_frontier
from oec.runtime import ProtocolError


def audit(run):
    manifest = json.loads((run/"manifest.json").read_text())
    expected = manifest["task_ids"]
    tasks, seen = [], set()
    for directory in sorted(run.glob("task_*")):
        result = json.loads((directory/"result.json").read_text())
        task_id = result["task_id"]
        if task_id in seen:
            raise ProtocolError("Duplicate trace task id")
        seen.add(task_id)
        events = [json.loads(line) for line in (directory/"events.jsonl").read_text().splitlines() if line.strip()]
        state, records = None, []
        for event in events:
            if event["event"] in ("phase1c_initial_state", "phase1c_state"):
                state = event["state"]
            elif event["event"] == "phase1c_selected":
                if state is None:
                    raise ProtocolError("Missing state before a selection")
                options = estimated_frontier(state["goals"], state["unresolved_goal_ids"])
                records.append({"revision": state.get("state_revision"), "frontier": options,
                                "opportunity": len(options) >= 2, "selected_doc_id": event["doc_id"]})
        tasks.append({"task_id": task_id, "decisions": records,
                      "has_opportunity": any(row["opportunity"] for row in records)})
    if seen != set(expected) or len(seen) != len(expected):
        raise ProtocolError("Trace task ids differ from manifest")
    all_decisions = [row for task in tasks for row in task["decisions"]]
    if not all_decisions:
        raise ProtocolError("No selection decisions found")
    return {"run": str(run), "n": len(tasks), "decision_n": len(all_decisions),
            "opportunity_task_n": sum(task["has_opportunity"] for task in tasks),
            "opportunity_task_rate": sum(task["has_opportunity"] for task in tasks)/len(tasks),
            "frontier_size_counts": dict(Counter(len(row["frontier"]) for row in all_decisions)),
            "warning": "Frontier uses observer-estimated unresolved status, not verified support. "
                       "An opportunity is necessary but does not imply useful policy disagreement.", "tasks": tasks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reports = [audit(run) for run in args.runs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(reports, handle, ensure_ascii=False, indent=2)
    print(json.dumps([{key: value for key,value in report.items() if key != "tasks"} for report in reports], indent=2))


if __name__ == "__main__":
    main()
