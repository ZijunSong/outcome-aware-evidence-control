"""Offline coverage of a frozen BM25 pool. Do not pass this output to a selector."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.freeze_bm25_pools import content_id, load_corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--hotpot", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.pool / "manifest.json").read_text())
    if manifest.get("gold_used") is not False or manifest.get("protocol") != "phase3-pooled-hotpot-bm25-v1":
        raise ValueError("Unexpected pool manifest")
    source = load_corpus(args.hotpot)
    gold = {row["_id"]: row for row in json.loads(args.gold.read_text())}
    both_source = both_pool = both_top2 = 0
    recall_sum = 0
    for task_id, expected in gold.items():
        pool = json.loads((args.pool / "pools" / f"{task_id}.json").read_text())["candidates"]
        pool_ids = [row["doc_id"] for row in pool]
        support_titles = {title for title, _ in expected["supporting_facts"]}
        support_ids = []
        for title, sentences in expected["context"]:
            if title in support_titles:
                support_ids.append(content_id(title, sentences))
        if not support_ids:
            raise ValueError("Question has no support paragraphs")
        present = [doc_id in source and source[doc_id]["sentences"] == next(
            sentences for title, sentences in expected["context"] if content_id(title, sentences) == doc_id)
                   for doc_id in support_ids]
        both_source += all(doc_id in source for doc_id in support_ids)
        both_pool += all(doc_id in pool_ids for doc_id in support_ids)
        both_top2 += all(doc_id in pool_ids[:2] for doc_id in support_ids)
        recall_sum += sum(doc_id in pool_ids for doc_id in support_ids) / len(support_ids)
        if not all(present):
            raise ValueError("Support paragraph text does not match the pooled corpus")
    count = len(gold)
    body = {"protocol": "phase3-OFFLINE-bm25-coverage-v1", "n": count,
            "both_support_paragraphs_in_corpus": both_source / count,
            "support_paragraph_recall_at_50": recall_sum / count,
            "both_support_paragraphs_in_pool": both_pool / count,
            "both_support_paragraphs_in_bm25_top2": both_top2 / count,
            "pool_manifest": manifest,
            "warning": "Offline coverage only. Not a selector result and not FullWiki."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: body[key] for key in body if key != "pool_manifest"}, indent=2))


if __name__ == "__main__":
    main()
