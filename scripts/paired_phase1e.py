"""New prospective endpoints ONLY. Never unlocks phase1d's zero-invalid gate."""
import argparse
import importlib.util
import json
import math
import random
from pathlib import Path


def unique(rows, key):
    output = {row[key]: row for row in rows}
    if len(output) != len(rows) or not output:
        raise ValueError("Duplicate ids or empty input")
    return output


def read_arm(path):
    manifest = json.loads((path/"manifest.json").read_text())
    if manifest.get("protocol") != "phase1e-shared-reader-evaluation-v1":
        raise ValueError("Only the new shared-reader protocol is accepted")
    rows = unique([json.loads(line) for line in (path/"predictions.jsonl").read_text().splitlines() if line.strip()], "task_id")
    if set(rows) != set(manifest["task_ids"]):
        raise ValueError("Task manifest mismatch")
    for row in rows.values():
        if type(row["runtime_valid"]) is not bool or not isinstance(row["answer"], str):
            raise ValueError("Malformed result contract")
        if not row["runtime_valid"] and row.get("failure_stage") != "bounded_reader_output":
            raise ValueError("Structural/selection failures cannot be scored as ordinary reader failures")
    return manifest, rows


def compare(baseline, method, gold_path, evaluator, endpoint, zero_gate=False):
    bm, before = read_arm(baseline)
    mm, after = read_arm(method)
    gold = unique(json.loads(gold_path.read_text()), "_id")
    if set(before) != set(after) or set(before) != set(gold):
        raise ValueError("Identical paired ids and full denominator are required")
    if bm["reader_policy"] != mm["reader_policy"]:
        raise ValueError("Mixed reader policies")
    if not bm.get("reader_ledger_sha256") or bm["reader_ledger_sha256"] != mm.get("reader_ledger_sha256"):
        raise ValueError("Comparisons must use the same immutable shared reader ledger")
    for key in ("selection_family", "tasks_sha256", "corpus_sha256", "plans_sha256", "source_sha256",
                "additional_source_sha256", "read_k", "limits", "llm_protocol"):
        if bm["selection_manifest"].get(key) != mm["selection_manifest"].get(key):
            raise ValueError("Mixed selection input/protocol: " + key)
    invalid = [sum(not row["runtime_valid"] for row in values.values()) for values in (before, after)]
    if zero_gate and any(invalid):
        raise ValueError("The prospectively selected zero-invalid gate blocks this comparison")
    spec = importlib.util.spec_from_file_location("hotpot_phase1e_pair", evaluator)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    def score(row, expected):
        if endpoint == "answer_em":
            return int(official.exact_match_score(row["answer"], expected["answer"])) if row["runtime_valid"] else 0
        titles = {title for title, _ in expected["supporting_facts"]}
        if not titles:
            raise ValueError("No offline support titles")
        return int(titles <= set(row["selected_titles"]))
    pairs = [(score(before[key], gold[key]), score(after[key], gold[key])) for key in sorted(gold)]
    differences = [b-a for a,b in pairs]
    wr = sum(a == 0 and b == 1 for a,b in pairs)
    rw = sum(a == 1 and b == 0 for a,b in pairs)
    rng, n = random.Random(17), len(pairs)
    draws = sorted(sum(rng.choice(differences) for _ in range(n))/n for _ in range(10000))
    discordant = wr+rw
    p = min(1., 2*sum(math.comb(discordant,k) for k in range(min(wr,rw)+1))/2**discordant) if discordant else 1.
    return {"protocol": "phase1e-prospective-paired-v1", "endpoint": endpoint, "n": n,
        "baseline_mean": sum(a for a,b in pairs)/n, "method_mean": sum(b for a,b in pairs)/n,
        "delta": sum(differences)/n, "paired_bootstrap_95ci": [draws[250], draws[9750]],
        "wrong_to_right": wr, "right_to_wrong": rw, "mcnemar_exact_p": p,
        "reader_invalid_baseline": invalid[0], "reader_invalid_method": invalid[1],
        "zero_invalid_gate": zero_gate,
        "endpoint_rule": "All assigned questions; bounded reader failure contributes 0 to answer EM. "
                         "Support coverage is computed from selected titles regardless of reader success.",
        "warning": "Requires prospective registration. This cannot change old phase1d's inference status; "
                   "intervals on questions do not include all reader/model instability or prove equivalence."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--official-evaluator", type=Path, required=True)
    parser.add_argument("--endpoint", choices=["answer_em", "support_coverage"], default="answer_em")
    parser.add_argument("--require-zero-invalid", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.method, args.gold, args.official_evaluator,
                     args.endpoint, args.require_zero_invalid)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
