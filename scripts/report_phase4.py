"""Phase4 report. Bounded final-reader failures score 0; structural failures block."""
import argparse
import hashlib
import importlib.util
import json
import math
import random
from pathlib import Path
sys_path_note = True
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oec.phase4_endpoints import paired_points
from oec.runtime import digest
from scripts.replay_phase1e_reader import lines


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_index(rows, key):
    output = {row[key]: row for row in rows}
    if not rows or len(output) != len(rows):
        raise ValueError("Empty or duplicate ids")
    return output


def load_branch(root, name):
    manifest = json.loads((root / "branches" / name / "manifest.json").read_text())
    if manifest.get("branch") != name or manifest.get("protocol") != "phase4-shared-reader-evaluation-v1":
        raise ValueError("Branch identity does not match its directory")
    rows = lines(root / "branches" / name / "predictions.jsonl")
    if any(row.get("branch") != name for row in rows):
        raise ValueError("Prediction branch labels were rewritten")
    return manifest, unique_index(rows, "task_id")


def summarize(pairs):
    differences = [method - baseline for baseline, method in pairs]
    wrong_to_right = sum(baseline == 0 and method == 1 for baseline, method in pairs)
    right_to_wrong = sum(baseline == 1 and method == 0 for baseline, method in pairs)
    count = len(pairs)
    rng = random.Random(17)
    draws = sorted(sum(rng.choice(differences) for _ in range(count)) / count for _ in range(10000))
    discordant = wrong_to_right + right_to_wrong
    p_value = 1.0 if not discordant else min(1.0, 2 * sum(
        math.comb(discordant, index) for index in range(min(wrong_to_right, right_to_wrong) + 1)) / 2 ** discordant)
    return {"n": count, "baseline_mean": sum(item[0] for item in pairs) / count,
            "method_mean": sum(item[1] for item in pairs) / count, "delta": sum(differences) / count,
            "paired_bootstrap_95ci": [draws[250], draws[9750]], "wrong_to_right": wrong_to_right,
            "right_to_wrong": right_to_wrong, "mcnemar_exact_p": p_value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--official-evaluator", type=Path, required=True)
    parser.add_argument("--mode", choices=("development", "confirmation"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--assignment", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.study / "report.json").read_text())
    if report.get("protocol") != "phase4-shared-reader-evaluation-v1":
        raise ValueError("This report refuses phase1e/phase3 ledgers")
    initial_manifest, initial = load_branch(args.study, "initial_once")
    current_manifest, current = load_branch(args.study, "current_once")
    if initial_manifest["reader_ledger_sha256"] != current_manifest["reader_ledger_sha256"]:
        raise ValueError("Reader ledgers differ")
    if initial_manifest["reader_ledger_sha256"] != report["reader_ledger_sha256"]:
        raise ValueError("Reader ledger does not match the study report")
    gold = json.loads(args.gold.read_text())
    spec = importlib.util.spec_from_file_location("official_phase4", args.official_evaluator)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    if args.mode == "confirmation":
        if args.config is None or args.assignment is None or args.registry is None:
            raise ValueError("Confirmation requires frozen config, assignment, and exposure registry")
        config = json.loads(args.config.read_text())
        assignment = json.loads(args.assignment.read_text())
        registry = json.loads(args.registry.read_text())
        if config.get("status") != "FROZEN_BEFORE_CONFIRMATION" or str(config.get("base_commit", "")).startswith("TO_FILL"):
            raise ValueError("Refusing to confirm an unfrozen phase4 config")
        if config.get("primary") != "current_once minus initial_once; all-assigned end-to-end answer EM":
            raise ValueError("Primary comparison is not the frozen current_once vs initial_once test")
        if config.get("read_k") != 2 or config.get("n") != len(gold) or config.get("alpha") != 0.05:
            raise ValueError("Frozen n/K/alpha do not match the scored sample")
        if assignment.get("n") != config["n"] or assignment.get("task_ids_sha256") != config.get("task_ids_hash"):
            raise ValueError("Assignment hashes do not match the frozen config")
        if set(assignment["assigned_ids"]) != set(current) or set(assignment["assigned_ids"]) != {row["_id"] for row in gold}:
            raise ValueError("Gold, predictions, and assignment ids differ")
        if sha(args.registry) != config.get("exposure_manifest"):
            raise ValueError("Exposure registry hash does not match the frozen config")
        used = set()
        for record in registry["files"]:
            path = Path(record["path"])
            if not path.is_absolute():
                path = args.registry.parent / path
            if sha(path) != record["sha256"]:
                raise ValueError("Exposure file hash changed")
            values = json.loads(path.read_text()) if path.read_text().lstrip().startswith("[") else [
                json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            used.update(row.get("id", row.get("task_id", row.get("_id"))) for row in values)
        if used & set(assignment["assigned_ids"]):
            raise ValueError("Assigned confirmation ids intersect recorded exposure")
        if current_manifest["reader_policy"] != json.loads(json.dumps(config["reader_policy"])):
            pass  # compared below via digest of live policy
        if digest(current_manifest["reader_policy"]) != config.get("reader_policy_sha256"):
            raise ValueError("Reader policy does not match the frozen config")
        if current_manifest["selection_manifest"]["model_identity_sha256"] != config.get("model_identity"):
            raise ValueError("Model identity does not match the frozen config")
        if current_manifest["selection_manifest"]["source_sha256"] != config.get("source_hashes"):
            raise ValueError("Source hashes do not match the frozen config")
        if current_manifest["selection_manifest"]["read_k"] != 2:
            raise ValueError("Confirmation is only defined for K=2")
    gold_index = {row["_id"]: row for row in gold}
    pairs = paired_points(
        [{"task_id": key, **initial[key]} for key in initial],
        [{"task_id": key, **current[key]} for key in current],
        gold, official.exact_match_score)

    def secondary(rows):
        exact = f1 = coverage = 0
        for key, row in rows.items():
            needed = {title for title, _ in gold_index[key]["supporting_facts"]}
            coverage += int(needed <= set(row.get("selected_titles") or []))
            if row["runtime_valid"]:
                exact += int(official.exact_match_score(row["answer"], gold_index[key]["answer"]))
                value, _, _ = official.f1_score(row["answer"], gold_index[key]["answer"])
                f1 += value
        count = len(rows)
        return {"n": count, "answer_em": exact / count, "answer_f1": f1 / count,
                "all_support_titles": coverage / count}

    body = {"protocol": "phase4-report-v1", "mode": args.mode,
            "primary": summarize(pairs),
            "initial_once": secondary(initial), "current_once": secondary(current),
            "same_second_document": sum(initial[key]["selected_doc_ids"][1] == current[key]["selected_doc_ids"][1]
                                        for key in current) / len(current),
            "current_first_observation_failures": sum(
                row["first_observation_status"] != "success" for row in current.values()),
            "reader_invalid_initial": sum(not row["runtime_valid"] for row in initial.values()),
            "reader_invalid_current": sum(not row["runtime_valid"] for row in current.values()),
            "note": "All assigned questions. Bounded reader failure scores 0. Development results are not confirmation."}
    if args.mode == "confirmation":
        passed = body["primary"]["delta"] > 0 and body["primary"]["mcnemar_exact_p"] < 0.05
        body["decision"] = "PRIMARY_PASSED" if passed else "PRIMARY_FAILED"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: body[key] for key in body if key != "note"}, indent=2))


if __name__ == "__main__":
    main()
