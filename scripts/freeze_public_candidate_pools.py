"""Turn already-assigned public candidates into a hashed pool. Does not retrieve or read gold."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oec.runtime import digest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_id(title, sentences):
    return hashlib.sha256((title + "\n" + "\n".join(sentences)).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Refusing to overwrite a frozen pool")
    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    corpus = {}
    for line in args.corpus.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            corpus[row["id"]] = row
    ids = [task["id"] for task in tasks]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Assigned questions must be unique")
    (args.out / "pools").mkdir(parents=True)
    used = set()
    for task in tasks:
        rows = []
        for rank, doc_id in enumerate(task["corpus_ids"], start=1):
            document = corpus[doc_id]
            used.add(doc_id)
            rows.append({"rank": rank, "doc_id": doc_id, "title": document["title"],
                         "snippet": " ".join(document["sentences"])[:240],
                         "retrieval_score": 0.0,
                         "content_sha256": content_id(document["title"], document["sentences"])})
        (args.out / "pools" / f"{task['id']}.json").write_text(json.dumps({
            "task_id": task["id"], "question": task["question"], "candidates": rows},
            ensure_ascii=False, indent=2), encoding="utf-8")
    corpus_path = args.out / "corpus.jsonl"
    corpus_path.write_text("".join(json.dumps(corpus[doc_id], ensure_ascii=False) + "\n"
                                   for doc_id in sorted(used)), encoding="utf-8")
    pool_files = {task["id"]: sha(args.out / "pools" / f"{task['id']}.json") for task in tasks}
    config = {"source": "assigned public candidates", "snippet_chars": 240, "retrieval_score": 0.0}
    manifest = {"protocol": "phase3-pooled-hotpot-bm25-v1", "corpus_name": "assigned public Hotpot paragraphs",
                "not_fullwiki": True, "top_n": "assigned", "questions": len(tasks),
                "gold_used": False, "task_ids": ids,
                "task_ids_sha256": digest(sorted(ids)), "tasks_sha256": sha(args.tasks),
                "source_code_sha256": sha(Path(__file__)), "config": config, "config_sha256": digest(config),
                "pool_files": pool_files, "pool_files_sha256": digest(pool_files),
                "corpus_sha256": sha(corpus_path)}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"questions": len(tasks), "pooled_documents": len(used)}, indent=2))


if __name__ == "__main__":
    main()
