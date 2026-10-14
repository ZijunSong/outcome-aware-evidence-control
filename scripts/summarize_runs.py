import argparse
import json
from pathlib import Path


def percentile(values, probability):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(directory: Path, gold_path: Path | None):
    gold = {} if gold_path is None else {row["_id"]: row for row in json.loads(gold_path.read_text(encoding="utf-8"))}
    latencies, counters, rows = [], {}, []
    for task_directory in sorted(directory.glob("task_*")):
        result = json.loads((task_directory / "result.json").read_text(encoding="utf-8"))
        events = [json.loads(line) for line in (task_directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        duration = events[-1]["elapsed_s"]
        latencies.append(duration)
        for key, value in result["usage"].items():
            counters.setdefault(key, []).append(value)
        candidates, shown = set(), set()
        for event in events:
            if event["event"] == "response" and event.get("kind") == "search":
                candidates.update(item["title"] for item in event["response"])
            if event["event"] == "response" and event.get("kind") == "read":
                shown.add(event["response"]["title"])
        row = {"task_id": result["task_id"], "runtime_valid": result["runtime_valid"],
               "latency_s": duration, "repairs": result["repairs"]}
        if result["task_id"] in gold:
            needed = {title for title, sentence in gold[result["task_id"]]["supporting_facts"]}
            row |= {"all_gold_titles_in_candidates": needed <= candidates,
                    "all_gold_titles_read": needed <= shown,
                    "candidate_title_coverage": len(needed & candidates) / max(1, len(needed)),
                    "read_title_coverage": len(needed & shown) / max(1, len(needed))}
        rows.append(row)
    if not rows:
        raise ValueError("No completed task result files")
    return {"n": len(rows), "invalid_runs": sum(not row["runtime_valid"] for row in rows),
            "latency_mean_s": sum(latencies)/len(latencies),
            "latency_p50_s": percentile(latencies, .5), "latency_p95_s": percentile(latencies, .95),
            "mean_usage_when_reported": {key: sum(values)/len(values) for key, values in counters.items()},
            "tasks": rows,
            "coverage_note": "Gold titles are read offline only. Title coverage is document-level "
                             "headroom, not sentence entailment or complete reasoning-chain coverage.",
            "latency_note": "First local-model task includes model loading. Report cold start separately "
                            "and warm latency explicitly; cached network runs are not comparable to cold runs."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(summarize(args.run, args.gold), ensure_ascii=False, indent=2), encoding="utf-8")
