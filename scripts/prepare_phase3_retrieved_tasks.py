"""Turn a frozen BM25 pool into provided-context tasks. Does not read gold or retrieve again."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Refusing to overwrite prepared tasks")
    manifest = json.loads((args.pool / "manifest.json").read_text())
    if manifest.get("gold_used") is not False:
        raise ValueError("Refusing a pool that used gold")
    args.out.mkdir(parents=True)
    tasks = []
    for path in sorted((args.pool / "pools").glob("*.json")):
        pool = json.loads(path.read_text())
        ids = [row["doc_id"] for row in pool["candidates"]]
        if len(ids) != len(set(ids)) or not ids:
            raise ValueError("Frozen pool has duplicate or empty candidates")
        tasks.append({"id": pool["task_id"], "question": pool["question"], "corpus_ids": ids})
    (args.out / "tasks.jsonl").write_text("".join(json.dumps(row) + "\n" for row in tasks), encoding="utf-8")
    print(json.dumps({"questions": len(tasks), "protocol": manifest["protocol"]}, indent=2))


if __name__ == "__main__":
    main()
