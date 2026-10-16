"""Strict phase3 confirmation. Refuses an unfrozen config and does not test the secondary early."""
import argparse
import importlib.util
import json
import math
import random
from pathlib import Path


REQUIRED = ("study_id", "base_commit", "population", "n", "primary", "secondary", "alpha",
            "read_k", "split_seed", "task_ids_hash", "source_hashes", "observer_ledger_policy",
            "reader_policy", "model_identity", "failure_rules", "exposure_manifest")


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pair(before, after, gold, exact_match):
    pairs = []
    for key in sorted(gold):
        def point(row):
            if row.get("status") == "INVALID_RUNTIME_EVALUATION" or row.get("failure_stage") == "upstream_selection":
                raise ValueError("Upstream selection failure blocks confirmation")
            if not row["runtime_valid"]:
                if row.get("failure_stage") != "bounded_reader_output":
                    raise ValueError("Structural failures cannot be scored as zero")
                return 0
            return int(exact_match(row["answer"], gold[key]["answer"]))
        pairs.append((point(before[key]), point(after[key])))
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


def primary_succeeded(result, alpha):
    return result["delta"] > 0 and result["mcnemar_exact_p"] < alpha


def load_policy(root, policy):
    report = json.loads((root / "report.json").read_text())
    found = None
    for row in report["arms"]:
        manifest = json.loads((Path(row["arm"]) / "manifest.json").read_text())
        if manifest["selection_manifest"].get("phase3_policy") == policy:
            if found is not None:
                raise ValueError("Duplicate policy")
            found = (manifest, {item["task_id"]: item for item in lines(Path(row["arm"]) / "predictions.jsonl")})
    if found is None:
        raise ValueError("Policy is not in the reader report: " + policy)
    return report, found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader-root", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--official-evaluator", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--exposure", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if any(key not in config for key in REQUIRED):
        raise ValueError("Confirmation config is missing a frozen field")
    if config.get("status") != "FROZEN_BEFORE_CONFIRMATION" or str(config.get("base_commit", "")).startswith("TO_FILL"):
        raise ValueError("Refusing to confirm an unfrozen phase3 config")
    exposure = json.loads(args.exposure.read_text())
    if exposure.get("unexposed_confirmation_available") is not True:
        raise ValueError("No audited unexposed confirmation sample is available")
    spec = importlib.util.spec_from_file_location("official_phase3_confirm", args.official_evaluator)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    gold = {row["_id"]: row for row in json.loads(args.gold.read_text())}
    report, utility = load_policy(args.reader_root, "utility")
    _, state = load_policy(args.reader_root, "state_relevance")
    if utility[0]["reader_ledger_sha256"] != state[0]["reader_ledger_sha256"]:
        raise ValueError("Reader ledgers differ")
    if set(utility[1]) != set(gold) or len(gold) != config["n"]:
        raise ValueError("Confirmation denominator does not match the frozen sample")
    primary = pair(state[1], utility[1], gold, official.exact_match_score)
    secondary = None
    secondary_status = "not_tested_because_primary_failed"
    if primary_succeeded(primary, config["alpha"]):
        _, setwise = load_policy(args.reader_root, "setwise_llm")
        secondary = pair(setwise[1], utility[1], gold, official.exact_match_score)
        secondary_status = "tested_after_primary_success"
    body = {"protocol": "phase3-confirmation-v1", "primary": primary, "secondary": secondary,
            "secondary_status": secondary_status, "reader_ledger_sha256": report["reader_ledger_sha256"],
            "decision": "PRIMARY_PASSED" if secondary_status.startswith("tested") else "PRIMARY_FAILED"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"decision": body["decision"], "secondary_status": secondary_status}, indent=2))


if __name__ == "__main__":
    main()
