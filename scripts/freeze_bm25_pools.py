"""Freeze question-only BM25 pools. This script has no gold argument."""
import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


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
    manifest = {"protocol": "phase3-pooled-hotpot-bm25-v1", "corpus_name": "pooled HotpotQA distractor paragraphs",
                "not_fullwiki": True, "top_n": args.top_n, "questions": len(tasks),
                "source_documents": len(documents), "pooled_documents": len(used),
                "tokenizer": r"\w+ casefold", "bm25": "LocalBM25 formula, positive scores only",
                "gold_used": False}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
