"""Freeze question-only BM25 pools. This script has no gold argument."""
import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oec.runtime import digest


def tokenize(text):
    return re.findall(r"\w+", text.casefold())


def content_id(title, sentences):
    body = title + "\n" + "\n".join(sentences)
    return hashlib.sha256(body.encode()).hexdigest()


def load_corpus(hotpot_path):
    raw = json.loads(Path(hotpot_path).read_text())
    documents, seen = {}, set()
    for row in raw:
        # Question and public context only. Answer and supporting facts are not read.
        for title, sentences in row["context"]:
            doc_id = content_id(title, sentences)
            if doc_id in seen:
                continue
            seen.add(doc_id)
            documents[doc_id] = {"id": doc_id, "title": title, "sentences": list(sentences)}
    return documents


def search(documents, counts, df, average_length, query, top_n):
    query_terms = tokenize(query)
    size = len(documents)
    scores = []
    for doc_id, counter in counts.items():
        length = sum(counter.values())
        score = 0.0
        for token in query_terms:
            frequency = counter[token]
            if not frequency:
                continue
            idf = math.log(1 + (size - df[token] + 0.5) / (df[token] + 0.5))
            denominator = frequency + 1.2 * (0.25 + 0.75 * length / max(1, average_length))
            score += idf * frequency * 2.2 / denominator
        if score > 0:
            scores.append((-score, doc_id))
    scores.sort()
    return scores[:top_n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotpot", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Refusing to overwrite a frozen pool")
    documents = load_corpus(args.hotpot)
    counts, df = {}, Counter()
    for doc_id, document in documents.items():
        tokens = tokenize(document["title"] + " " + " ".join(document["sentences"]))
        counts[doc_id] = Counter(tokens)
        df.update(set(tokens))
    average = sum(sum(counter.values()) for counter in counts.values()) / max(1, len(documents))
    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    ids = [task["id"] for task in tasks]
    if not ids or len(set(ids)) != len(ids) or any(not task.get("question") for task in tasks):
        raise ValueError("Assigned questions must be unique and nonempty")
    (args.out / "pools").mkdir(parents=True)
    pools = []
    for task in tasks:
        hits = search(documents, counts, df, average, task["question"], args.top_n)
        rows = []
        for rank, (negative, doc_id) in enumerate(hits, start=1):
            document = documents[doc_id]
            rows.append({"rank": rank, "doc_id": doc_id, "title": document["title"],
                         "snippet": " ".join(document["sentences"])[:240],
                         "retrieval_score": -negative,
                         "content_sha256": doc_id})
        pools.append({"task_id": task["id"], "question": task["question"], "candidates": rows})
        (args.out / "pools" / f"{task['id']}.json").write_text(
            json.dumps(pools[-1], ensure_ascii=False, indent=2), encoding="utf-8")
    used = {row["doc_id"] for pool in pools for row in pool["candidates"]}
    corpus_path = args.out / "corpus.jsonl"
    corpus_path.write_text("".join(json.dumps(documents[doc_id], ensure_ascii=False) + "\n" for doc_id in sorted(used)), encoding="utf-8")
    pool_files = {task["id"]: hashlib.sha256((args.out / "pools" / f"{task['id']}.json").read_bytes()).hexdigest()
                  for task in tasks}
    if set(pool_files) != set(ids):
        raise ValueError("Pool files do not match the assigned questions")
    config = {"k1": 1.2, "b": 0.75, "query_term_duplicates": True, "tokenizer": r"\w+ casefold",
              "positive_scores_only": True, "tie_break": "(-score, doc_id)", "snippet_chars": 240,
              "content_id": "sha256(title + newline-joined sentences)"}
    manifest = {"protocol": "phase3-pooled-hotpot-bm25-v1", "corpus_name": "pooled HotpotQA distractor paragraphs",
                "not_fullwiki": True, "top_n": args.top_n, "questions": len(tasks),
                "source_documents": len(documents), "pooled_documents": len(used),
                "tokenizer": config["tokenizer"], "bm25": "LocalBM25 formula, positive scores only",
                "gold_used": False, "task_ids": ids,
                "task_ids_sha256": hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest(),
                "tasks_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
                "hotpot_sha256": hashlib.sha256(args.hotpot.read_bytes()).hexdigest(),
                "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "config": config, "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                "pool_files": pool_files,
                "pool_files_sha256": digest(pool_files),
                "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest()}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("protocol", "questions", "source_documents",
                                                     "pooled_documents", "gold_used")}, indent=2))


if __name__ == "__main__":
    main()
