"""Check that a larger K extends the same policy's earlier selections."""
import argparse
import json
from pathlib import Path


def sequences(run):
    output = {}
    for path in Path(run).glob("task_*/events.jsonl"):
        events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        task_id = json.loads((path.parent / "result.json").read_text())["task_id"]
        output[task_id] = [row["doc_id"] for row in events if row["event"] == "phase1c_selected"]
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    loaded = []
    for run in args.runs:
        manifest = json.loads((run / "manifest.json").read_text())
        if manifest.get("phase3_policy") != args.policy or not manifest.get("completed"):
            raise ValueError("Run does not match the requested policy")
        loaded.append((manifest["read_k"], sequences(run)))
    loaded.sort()
    mismatches = []
    base_k, base = loaded[0]
    for read_k, chosen in loaded[1:]:
        for task_id, sequence in chosen.items():
            if sequence[:base_k] != base[task_id][:base_k]:
                mismatches.append({"task_id": task_id, "shorter_k": base_k, "longer_k": read_k})
            for earlier_k, earlier in loaded:
                if earlier_k < read_k and sequence[:earlier_k] != earlier[task_id][:earlier_k]:
                    mismatches.append({"task_id": task_id, "shorter_k": earlier_k, "longer_k": read_k})
    body = {"policy": args.policy, "read_ks": [item[0] for item in loaded],
            "prefix_mismatches": len({(row["task_id"], row["shorter_k"], row["longer_k"]) for row in mismatches}),
            "examples": mismatches[:20]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: body[key] for key in ("policy", "read_ks", "prefix_mismatches")}, indent=2))


if __name__ == "__main__":
    main()
