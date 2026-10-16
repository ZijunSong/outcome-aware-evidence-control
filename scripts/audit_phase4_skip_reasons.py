"""Read-only classification of observation_skipped events. Not an online input."""
import argparse
import json
from collections import Counter
from pathlib import Path


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def bucket(error, rejected):
    text = " ".join([error or ""] + rejected).lower()
    if "sentence id" in text or "outside 0" in text:
        return "illegal_sentence_id"
    if "unknown" in text and "goal" in text:
        return "illegal_goal_id"
    if "truncat" in text or "finish_reason" in text:
        return "truncated_or_unfinished"
    if "http" in text:
        return "http_error"
    if "json" in text or "schema" in text:
        return "json_or_schema"
    if "two output attempts failed" in text:
        return "bounded_format_failure_unspecified"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    counts, examples = Counter(), []
    for directory in sorted(args.run.glob("task_*")):
        events = lines(directory / "events.jsonl")
        result = json.loads((directory / "result.json").read_text())
        position = 0
        pending = []
        for event in events:
            if event["event"] == "json_rejected":
                pending.append(event.get("reason") or event.get("error") or "")
            if event["event"] == "phase1c_selected":
                position += 1
                pending = []
            if event["event"] == "observation_skipped":
                kind = bucket(event.get("error"), pending)
                counts[kind] += 1
                counts[f"position_{position}"] += 1
                if len(examples) < 20:
                    examples.append({"task_id": result["task_id"], "position": position,
                                     "bucket": kind, "error": event.get("error", "")[:300],
                                     "json_rejected": pending[:4]})
                pending = []
    body = {"run": str(args.run), "counts": dict(counts), "examples": examples}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"counts": dict(counts)}, indent=2))


if __name__ == "__main__":
    main()
