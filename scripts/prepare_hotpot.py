import argparse
import hashlib
import json
from pathlib import Path


def prepare(source: Path, destination: Path, limit: int, offset: int, pooled: bool, fullwiki: bool = False):
    source_rows = json.loads(source.read_text(encoding="utf-8"))
    rows = source_rows[offset:offset + limit]
    if not rows or len({row["_id"] for row in rows}) != len(rows):
        raise ValueError("Empty selection or duplicate task ids")
    documents, tasks = {}, []
    for row in rows:
        context_ids = []
        for title, sentences in row["context"]:
            doc_id = hashlib.sha256(title.encode()).hexdigest()
            document = {"id": doc_id, "title": title, "sentences": sentences, "source": "HotpotQA context"}
            if doc_id in documents and documents[doc_id] != document:
                raise ValueError("Same title has inconsistent source text; resolve versions before running")
            documents[doc_id] = document
            context_ids.append(doc_id)
        tasks.append({"id": row["_id"], "question": row["question"],
                      "corpus_ids": None if pooled or fullwiki else context_ids})
    destination.mkdir(parents=True, exist_ok=False)
    for name, values in (("corpus.jsonl", documents.values()), ("tasks.jsonl", tasks)):
        (destination / name).write_text("".join(json.dumps(value, ensure_ascii=False) + "\n"
                                              for value in values), encoding="utf-8")
    # Gold is a separate offline-only file: never pass this path to oec.run.
    (destination / "gold.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    (destination / "manifest.json").write_text(json.dumps({
        "setting": ("wiki18-e5-fullwiki" if fullwiki else
                    "pooled-context-pilot" if pooled else "provided-context"),
        "agent_reads_gold": False,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "offset": offset, "n": len(rows), "documents": len(documents)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--pooled", action="store_true")
    parser.add_argument("--fullwiki", action="store_true",
                        help="Leave corpus_ids empty so search uses the full Wikipedia index")
    args = parser.parse_args()
    if args.limit <= 0 or args.offset < 0:
        parser.error("limit must be positive and offset non-negative")
    if args.pooled and args.fullwiki:
        parser.error("--pooled and --fullwiki select different corpora")
    prepare(args.source, args.out, args.limit, args.offset, args.pooled, args.fullwiki)
