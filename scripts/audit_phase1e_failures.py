"""Read-only failure inventory; never reads gold or computes efficacy tests."""
import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = []
    for run in args.runs:
        directories = sorted(run.glob("task_*"))
        failures, valid, missing = [], 0, []
        for directory in directories:
            if not (directory/"result.json").is_file() or not (directory/"events.jsonl").is_file():
                missing.append(str(directory))
                continue
            result = json.loads((directory/"result.json").read_text())
            if result["runtime_valid"]:
                valid += 1
                continue
            events = [json.loads(line) for line in (directory/"events.jsonl").read_text().splitlines() if line.strip()]
            current_role, final_calls, json_rejects, request_errors, finish_reasons = "unknown", 0, [], [], []
            for event in events:
                if event["event"] == "llm_role":
                    current_role = event["role"]
                if event["event"] == "request" and event.get("kind") == "llm":
                    try:
                        schema_name = json.loads(event["request"]["messages"][-1]["content"])["schema"]["title"]
                        current_role = {"ShortAnswer": "phase1_answer"}.get(schema_name, schema_name)
                    except (KeyError, TypeError, ValueError):
                        pass
                    final_calls += int(current_role == "phase1_answer")
                if event["event"] == "response" and event.get("kind") == "llm" and current_role == "phase1_answer":
                    finish_reasons.extend(choice.get("finish_reason") for choice in event["response"].get("choices", []))
                if event["event"] == "json_rejected":
                    json_rejects.append({key: event.get(key) for key in ("role", "attempt", "reason", "raw_response_hash")})
                if event["event"] in ("request_error", "invalid_run", "budget_exceeded"):
                    request_errors.append(event)
            failures.append({"task_id": result["task_id"], "directory": str(directory),
                "status": result["status"], "final_answer_http_calls": final_calls,
                "answer_finish_reasons": finish_reasons, "json_rejections": json_rejects,
                "errors": request_errors, "usage": result["usage"],
                "raw_events_path": str(directory/"events.jsonl")})
        manifest_path = run/"manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        expected = manifest.get("task_ids")
        actual = [json.loads((directory/"result.json").read_text())["task_id"] for directory in directories
                  if (directory/"result.json").is_file()]
        complete = expected is not None and set(actual) == set(expected) and len(actual) == len(set(actual)) and not missing
        report.append({"run": str(run), "outputs_complete": complete, "valid": valid,
            "invalid": len(failures), "missing": missing, "failures": failures,
            "efficacy_inference_permitted_by_zero_invalid_gate": complete and not failures})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps([{key: value for key, value in row.items() if key != "failures"} for row in report], indent=2))


if __name__ == "__main__":
    main()
