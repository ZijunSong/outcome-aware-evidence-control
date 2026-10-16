"""Offline table for nested-prefix reader diagnostics. Not a C2 confirmation."""
import argparse
import importlib.util
import json
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def unique(values, key):
    output = {row[key]: row for row in values}
    if not values or len(output) != len(values):
        raise ValueError("Empty or duplicate ids")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader-root", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--official-evaluator", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("official_prefix", args.official_evaluator)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    gold = unique(json.loads(args.gold.read_text()), "_id")
    report = json.loads((args.reader_root / "report.json").read_text())
    max_k = report["source_header"]["max_k"]
    by_k = {}
    previous = None
    flips = {}
    for k in range(1, max_k + 1):
        predictions = unique(rows(args.reader_root / f"k{k}" / "predictions.jsonl"), "task_id")
        if set(predictions) != set(gold):
            raise ValueError("Prefix predictions do not match gold")
        exact = f1 = full = invalid = abstain = nonempty = 0
        points = {}
        for key, row in predictions.items():
            needed = {title for title, _ in gold[key]["supporting_facts"]}
            full += needed <= set(row["selected_titles"])
            invalid += not row["runtime_valid"]
            if row["runtime_valid"] and row.get("status") == "ANSWER_ATTEMPTED_ABSTAINED":
                abstain += 1
            if row["runtime_valid"] and row.get("status") != "ANSWER_ATTEMPTED_ABSTAINED":
                nonempty += int(not official.exact_match_score(row["answer"], gold[key]["answer"]))
            if row["runtime_valid"]:
                exact += int(official.exact_match_score(row["answer"], gold[key]["answer"]))
                value, _, _ = official.f1_score(row["answer"], gold[key]["answer"])
                f1 += value
            points[key] = int(row["runtime_valid"] and official.exact_match_score(row["answer"], gold[key]["answer"]))
        count = len(predictions)
        by_k[k] = {"n": count, "answer_em": exact / count, "answer_f1": f1 / count,
                   "all_support_titles": full / count, "bounded_reader_failures": invalid,
                   "legal_abstentions": abstain, "nonempty_wrong": nonempty}
        if previous is not None:
            wr = sum(previous[key] == 0 and points[key] == 1 for key in points)
            rw = sum(previous[key] == 1 and points[key] == 0 for key in points)
            flips[f"{k-1}_to_{k}"] = {"wrong_to_right": wr, "right_to_wrong": rw,
                                       "right_to_wrong_ids": [key for key in sorted(points)
                                                              if previous[key] == 1 and points[key] == 0]}
        previous = points
    body = {"protocol": "phase4-prefix-reader-diagnostic-v1", "order": report["source_header"]["reader_order_policy"],
            "by_k": by_k, "flips": flips, "reader_ledger_sha256": report["reader_ledger_sha256"],
            "note": "Nested prefixes from one frozen trajectory. This is not a new C2 confirmation."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(body, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
