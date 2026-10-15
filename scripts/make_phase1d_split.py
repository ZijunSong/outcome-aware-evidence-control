"""Deterministic disjoint splits. Gold is written for offline evaluation only."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_ids(path):
    text = path.read_text(encoding="utf-8")
    rows = json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()]
    return {row.get("id", row.get("task_id", row.get("_id"))) for row in rows}


def build(source, excluded, out, tune_n=60, confirm_n=500, seed=20261014):
    rows = json.loads(source.read_text(encoding="utf-8"))
    if len({row["_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate source ids")
    used = set().union(*(load_ids(path) for path in excluded))
    if None in used:
        raise ValueError("An exclusion row has no task id")
    candidates = [row for row in rows if row["_id"] not in used]
    candidates.sort(key=lambda row: hashlib.sha256(
        f"{seed}:{row['_id']}".encode()).hexdigest())
    eligible, ineligible = [], []
    for row in candidates:
        titles = [title for title, _ in row["context"]]
        if len(row["context"]) == 10 and len(set(titles)) == 10:
            eligible.append(row)
        else:
            ineligible.append(row["_id"])
    if min(tune_n, confirm_n) <= 0 or len(eligible) < tune_n + confirm_n:
        raise ValueError("Not enough distinct unused tasks")
    splits = {"tune": eligible[:tune_n],
              "confirm": eligible[tune_n:tune_n + confirm_n]}
    splits["smoke"] = splits["tune"][:min(20, tune_n)]
    documents, tasks_by_id, title_versions = {}, {}, {}
    for row in splits["tune"] + splits["confirm"]:
        context = row["context"]
        if len(context) != 10 or len({title for title, _ in context}) != 10:
            raise ValueError(f"Expected 10 unique public candidates: {row['_id']}")
        scope = []
        for title, sentences in context:
            if not sentences or not all(isinstance(x, str) for x in sentences):
                raise ValueError("Invalid context sentences")
            content = json.dumps(sentences, ensure_ascii=False, separators=(",", ":"))
            versions = title_versions.setdefault(title, {})
            if content not in versions:
                suffix = "" if not versions else "_" + hashlib.sha256(content.encode()).hexdigest()[:12]
                doc_id = hashlib.sha256(title.encode()).hexdigest() + suffix
                versions[content] = doc_id
                documents[doc_id] = {"id": doc_id, "title": title, "sentences": sentences,
                                     "source": "HotpotQA context"}
            scope.append(versions[content])
        tasks_by_id[row["_id"]] = {"id": row["_id"], "question": row["question"],
                                    "corpus_ids": scope}
    versioned_titles = sorted(title for title, versions in title_versions.items() if len(versions) > 1)
    out.mkdir(parents=True, exist_ok=False)
    corpus = out / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(documents[key], ensure_ascii=False) + "\n"
                              for key in sorted(documents)), encoding="utf-8")
    for name, subset in splits.items():
        directory = out / name
        directory.mkdir()
        (directory / "tasks.jsonl").write_text("".join(
            json.dumps(tasks_by_id[row["_id"]], ensure_ascii=False) + "\n"
            for row in subset), encoding="utf-8")
        (directory / "gold.json").write_text(json.dumps(subset, ensure_ascii=False), encoding="utf-8")
    manifest = {"protocol": "phase1d-disjoint-v1", "seed": seed,
        "source_sha256": digest(source), "excluded_files": [
            {"path": str(path), "sha256": digest(path)} for path in excluded],
        "excluded_id_count": len(used),
        "ineligible_not_10_unique_titles": ineligible,
        "versioned_titles": versioned_titles,
        "corpus_sha256": digest(corpus),
        "bm25_scope": "one frozen corpus for every split; per-task candidate scope is exactly 10",
        "splits": {name: {"n": len(subset), "task_ids": [row["_id"] for row in subset],
            "tasks_sha256": digest(out / name / "tasks.jsonl"),
            "gold_sha256": digest(out / name / "gold.json")} for name, subset in splits.items()}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exclude-tasks", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tune-n", type=int, default=60)
    parser.add_argument("--confirm-n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    result = build(args.source, args.exclude_tasks, args.out, args.tune_n, args.confirm_n, args.seed)
    print(json.dumps({"splits": {key: value["n"] for key, value in result["splits"].items()},
                      "corpus_sha256": result["corpus_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
