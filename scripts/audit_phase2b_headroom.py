"""OFFLINE ONLY: descriptive residual-error audit. Never pass its output to an agent."""
import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path


def unique(rows, key):
    output = {row[key]: row for row in rows}
    if not rows or len(output) != len(rows):
        raise ValueError("Empty/duplicate input ids")
    return output


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(tasks, predictions, corpus, gold, exact_match):
    tasks, predictions = unique(tasks, "id"), unique(predictions, "task_id")
    corpus, gold = unique(corpus, "id"), unique(gold, "_id")
    if not set(tasks) == set(predictions) == set(gold):
        raise ValueError("Task/prediction/gold ids must match exactly")
    counts, details = Counter(), []
    for key, task in tasks.items():
        row, expected = predictions[key], gold[key]
        scope, selected = task["corpus_ids"], row["selected_doc_ids"]
        if len(scope) != 10 or len(set(scope)) != 10 or not set(scope) <= set(corpus):
            raise ValueError("Expected exactly ten known public documents")
        if len(selected) != 2 or len(set(selected)) != 2 or not set(selected) <= set(scope):
            raise ValueError("Expected K=2 source selection")
        if type(row["runtime_valid"]) is not bool or not isinstance(row["answer"], str):
            raise ValueError("Invalid reader result contract")
        if not row["runtime_valid"] and row.get("failure_stage") != "bounded_reader_output":
            raise ValueError("Structural failures cannot be audited as bounded reader failures")
        if task["question"] != expected["question"]:
            raise ValueError("Question mismatch")
        for title, sentences in expected["context"]:
            matches = [corpus[doc_id] for doc_id in scope if corpus[doc_id]["title"] == title]
            if len(matches) != 1 or matches[0]["sentences"] != sentences:
                raise ValueError("Frozen public source differs from gold context")
        needed = {title for title, _ in expected["supporting_facts"]}
        titles = {corpus[doc_id]["title"] for doc_id in selected}
        if not needed or not needed <= {corpus[doc_id]["title"] for doc_id in scope}:
            raise ValueError("Support titles missing from public source")
        correct = bool(row["runtime_valid"] and exact_match(row["answer"], expected["answer"]))
        missing = needed - titles
        if not row["runtime_valid"]:
            category = "bounded_reader_failure"
        elif correct:
            category = "correct"
        elif not missing:
            category = "wrong_with_all_support_titles_read"
        else:
            category = "wrong_with_missing_support_title"
        counts[category] += 1
        # A source-coverage proxy, not a guaranteed answer-repair ceiling.
        possible_one = category == "wrong_with_missing_support_title" and len(missing) == 1
        counts["valid_wrong"] += int(row["runtime_valid"] and not correct)
        counts["one_extra_doc_support_proxy"] += int(possible_one)
        details.append({"task_id": key, "category": category,
                        "missing_support_titles": sorted(missing),
                        "one_extra_doc_support_proxy": possible_one})
    return {"protocol": "phase2b-OFFLINE-headroom-v1", "n": len(tasks), "counts": dict(counts),
            "warning": "Descriptive source-coverage proxy; no verifier or repair benefit established. "
                       "Contains gold. Forbidden as online input.", "tasks": details}


def main():
    parser = argparse.ArgumentParser()
    for name in ("tasks", "predictions", "corpus", "gold", "official-evaluator", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("official_for_offline_audit", args.official_evaluator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = audit(jsonl(args.tasks), jsonl(args.predictions), jsonl(args.corpus),
                   json.loads(args.gold.read_text()), module.exact_match_score)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key != "tasks"}, indent=2))


if __name__ == "__main__":
    main()
