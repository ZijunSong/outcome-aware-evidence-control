"""Hash recorded historical task files. Completeness is still an owner declaration."""
import argparse
import hashlib
import json
from pathlib import Path


FILES = [
    "data/pilot/tasks.jsonl",
    "data/phase1_smoke/tasks.jsonl",
    "data/phase1_patch_smoke/tasks.jsonl",
    "data/phase1c_smoke/tasks.jsonl",
    "data/fullwiki_smoke/tasks.jsonl",
    "data/demo_prepared/tasks.jsonl",
    "data/phase1d/tune/tasks.jsonl",
    "data/phase1d/smoke/tasks.jsonl",
    "data/phase1d/confirm/tasks.jsonl",
    "data/phase1e_base/tune/tasks.jsonl",
    "data/phase1e_base/smoke/tasks.jsonl",
    "data/phase1e_base/confirm/tasks.jsonl",
    "data/phase1e_studies/main/tasks.jsonl",
    "data/phase1e_studies/mechanism/tasks.jsonl",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data/ppnm/oec_project"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for relative in FILES:
        path = args.root / relative
        records.append({"path": str(path.resolve()),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    body = {"history_complete_declared": True,
            "note": "Declared complete relative to recorded local task files in this repository. Unrecorded manual inspection cannot be certified.",
            "files": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(records)}, indent=2))


if __name__ == "__main__":
    main()
