"""Exploratory phase3 quality table. This is not a frozen confirmation."""
import argparse
import importlib.util
import json
from pathlib import Path


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_arms(reader_root):
    report = json.loads((reader_root / "report.json").read_text())
    if report.get("protocol") != "phase1e-shared-reader-evaluation-v1":
        raise ValueError("Expected the shared reader report")
    arms = {}
    for row in report["arms"]:
        manifest = json.loads((Path(row["arm"]) / "manifest.json").read_text())
        policy = manifest["selection_manifest"].get("phase3_policy")
        if not policy or policy in arms:
            raise ValueError("Arm policy is missing or duplicated")
        predictions = {item["task_id"]: item for item in lines(Path(row["arm"]) / "predictions.jsonl")}
        arms[policy] = {"arm": row["arm"], "source_run": row["source_run"], "predictions": predictions,
                        "invalid_runs": row["invalid_runs"], "manifest": manifest}
    return report, arms


def sequences(run):
    output = {}
    for path in Path(run).glob("task_*/events.jsonl"):
        events = lines(path)
        selected = [row["doc_id"] for row in events if row["event"] == "phase1c_selected"]
        task_id = json.loads((path.parent / "result.json").read_text())["task_id"]
        output[task_id] = selected
        output[task_id + ":skipped"] = sum(row["event"] == "observation_skipped" for row in events)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader-root", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--official-evaluator", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("official_phase3_dev", args.official_evaluator)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    gold = {row["_id"]: row for row in json.loads(args.gold.read_text())}
    report, arms = load_arms(args.reader_root)
    policies = {}
    stored = {}
    for policy, arm in arms.items():
        rows = arm["predictions"]
        if set(rows) != set(gold):
            raise ValueError("Prediction ids do not match gold")
        exact = f1 = coverage = full = invalid = 0
        for task_id, row in rows.items():
            needed = {title for title, _ in gold[task_id]["supporting_facts"]}
            have = set(row["selected_titles"])
            coverage += len(needed & have) / len(needed)
            full += needed <= have
            invalid += not row["runtime_valid"]
            if row["runtime_valid"]:
                exact += int(official.exact_match_score(row["answer"], gold[task_id]["answer"]))
                exact_f1, _, _ = official.f1_score(row["answer"], gold[task_id]["answer"])
                f1 += exact_f1
        count = len(rows)
        chosen = sequences(arm["source_run"])
        policies[policy] = {"n": count, "answer_em": exact / count, "answer_f1": f1 / count,
                            "support_title_recall": coverage / count, "all_support_titles": full / count,
                            "bounded_reader_failures": invalid,
                            "observation_skipped": sum(chosen[key + ":skipped"] for key in rows)}
        stored[policy] = {key: value for key, value in chosen.items() if not key.endswith(":skipped")}
    reference = stored["utility"]
    changes = {}
    for policy, chosen in stored.items():
        if policy == "utility":
            continue
        first = second = same = 0
        for task_id, sequence in chosen.items():
            base = reference[task_id]
            same += set(sequence) == set(base)
            first += sequence[:1] != base[:1]
            second += sequence[1:2] != base[1:2]
        count = len(chosen)
        changes[policy] = {"same_set": same / count, "first_doc_changed": first / count,
                           "second_doc_changed": second / count, "n": count}
    body = {"protocol": "phase3-development-quality-v1", "status": "EXPLORATORY_NOT_CONFIRMATION",
            "reader_ledger_sha256": report["reader_ledger_sha256"], "policies": policies,
            "changes_versus_utility": changes,
            "note": "All assigned questions. Bounded reader failure scores 0. This file does not open a confirmation."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(body, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
