"""Read-only selection-phase accounting. Does NOT claim complete end-to-end latency."""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def event_cost(events):
    total = Counter()
    for event in events:
        if event["event"] == "request":
            kind = event["kind"]
            total["physical_"+kind+"_requests"] += 1
            if kind == "decision":
                meta = event["request"]
                if "examples" in meta and "lengths" in meta:
                    if len(meta["lengths"]) != meta["examples"]:
                        raise ValueError("Native forward length/example metadata mismatch")
                    total["native_examples"] += meta["examples"]
                    total["native_encoded_tokens"] += sum(meta["lengths"])
                else:
                    total["decision_requests_without_encoded_lengths"] += 1
        if event["event"] in ("response", "request_error"):
            total["physical_service_seconds"] += event.get("latency_s", 0)
        if event["event"] == "response" and event.get("kind") == "llm":
            usage = event["response"].get("usage", {})
            if not usage:
                total["llm_responses_without_usage"] += 1
            total["physical_llm_input_tokens"] += usage.get("prompt_tokens", usage.get("input_tokens", 0))
            total["physical_llm_output_tokens"] += usage.get("completion_tokens", usage.get("output_tokens", 0))
        if event["event"] == "phase3_observer_cache_hit":
            total["cached_logical_llm_requests"] += event["logical_used"].get("llm", 0)
            tokens = event["logical_tokens"]
            total["cached_logical_llm_input_tokens"] += tokens.get("llm.prompt_tokens", tokens.get("llm.input_tokens", 0))
            total["cached_logical_llm_output_tokens"] += tokens.get("llm.completion_tokens", tokens.get("llm.output_tokens", 0))
    total["logical_llm_requests"] = total["physical_llm_requests"]+total["cached_logical_llm_requests"]
    for side in ("input", "output"):
        total["logical_llm_"+side+"_tokens"] = total["physical_llm_"+side+"_tokens"]+total["cached_logical_llm_"+side+"_tokens"]
    return dict(total)


def summarize(run):
    manifest = json.loads((run/"manifest.json").read_text())
    if manifest.get("selection_family") != "phase3-controlled-selection-v1" or not manifest.get("completed"):
        raise ValueError("Expected completed phase3 controlled selection")
    expected = manifest["task_ids"]
    rows = {}
    for directory in sorted(run.glob("task_*")):
        result = json.loads((directory/"result.json").read_text())
        key = result["task_id"]
        if key in rows or result["status"] != "SELECTION_ONLY_NOT_EVALUATED":
            raise ValueError("Duplicate/incomplete selection result")
        events = lines(directory/"events.jsonl")
        if any(event["event"] == "invalid_run" for event in events):
            raise ValueError("Invalid upstream selection cannot be treated as a normal cost observation")
        rows[key] = event_cost(events)
    if set(rows) != set(expected) or len(rows) != len(expected):
        raise ValueError("Task manifest mismatch")
    keys = set().union(*(set(row) for row in rows.values()))
    means = {key: statistics.mean(row.get(key, 0) for row in rows.values()) for key in sorted(keys)}
    return {"policy": manifest["phase3_policy"], "n": len(rows), "means": means,
            "scope": "selection phase only: includes observations and selector calls; excludes shared plan and final reader",
            "warning": "Physical service time depends on ledger reuse and is not an end-to-end latency comparison. "
                       "Native encoded tokens and autoregressive tokens are different units; do not add them.",
            "tasks": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reports = [summarize(run) for run in args.runs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(reports, handle, ensure_ascii=False, indent=2)
    print(json.dumps([{key: value for key, value in row.items() if key != "tasks"} for row in reports], indent=2))


if __name__ == "__main__":
    main()
