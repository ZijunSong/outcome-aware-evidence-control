"""Development or confirmation report. Policy identity comes from manifests, not directory order."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math
import random
from oec.runtime import ProtocolError
from scripts.replay_phase1e_reader import lines


def load_policy(root, name):
    manifest = json.loads((root / "policies" / name / "manifest.json").read_text())
    if manifest.get("policy") != name or manifest.get("protocol") != "phase2b-shared-reader-evaluation-v1":
        raise ProtocolError("Policy identity does not match its directory")
    rows = lines(root / "policies" / name / "predictions.jsonl")
    if any(row.get("policy") != name for row in rows):
        raise ProtocolError("Prediction policy label was reordered or rewritten")
    return manifest, rows


def exact_compare(baseline_rows, method_rows, gold, exact_match):
    if set(baseline_rows) != set(method_rows) or set(baseline_rows) != set(gold):
        raise ProtocolError("Paired comparison requires the full assigned denominator")
    pairs = []
    for key in sorted(gold):
        def point(row):
            if not row["runtime_valid"]:
                if row.get("failure_stage") != "bounded_reader_output":
                    raise ProtocolError("Structural failures cannot be scored as zero")
                return 0
            return int(exact_match(row["answer"], gold[key]["answer"]))
        pairs.append((point(baseline_rows[key]), point(method_rows[key])))
    differences = [method - baseline for baseline, method in pairs]
    wrong_to_right = sum(baseline == 0 and method == 1 for baseline, method in pairs)
    right_to_wrong = sum(baseline == 1 and method == 0 for baseline, method in pairs)
    rng, count = random.Random(17), len(pairs)
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
    parser.add_argument("--headroom", type=Path)
    parser.add_argument("--probe-run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.study / "report.json").read_text())
    if report.get("protocol") != "phase2b-shared-reader-evaluation-v1":
        raise ProtocolError("This report refuses phase1e or C1 ledgers")
    names = [row["policy"] for row in report["policies"]]
    loaded = {name: load_policy(args.study, name) for name in names}
    ledgers = {manifest["reader_ledger_sha256"] for manifest, _ in loaded.values()}
    if ledgers != {report["reader_ledger_sha256"]}:
        raise ProtocolError("Reader ledgers differ across policies")
    if args.mode == "confirmation":
        if args.config is None:
            raise ProtocolError("Confirmation requires the frozen config")
        config = json.loads(args.config.read_text())
        if config.get("status") != "FROZEN_BEFORE_CONFIRMATION" or str(config.get("base_commit", "")).startswith("TO_FILL"):
            raise ProtocolError("Confirmation config is not frozen")
    spec = importlib.util.spec_from_file_location("official_for_phase2b", args.official_evaluator)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    gold = {row["_id"]: row for row in json.loads(args.gold.read_text())}
    primary = secondary = None
    if {"verify_local", "continue_dynamic"} <= set(names):
        primary = exact_compare(dict((row["task_id"], row) for row in loaded["continue_dynamic"][1]),
                                dict((row["task_id"], row) for row in loaded["verify_local"][1]),
                                gold, official.exact_match_score)
    if primary and primary["delta"] > 0 and "verify_broad" in names:
        secondary = exact_compare(dict((row["task_id"], row) for row in loaded["verify_broad"][1]),
                                  dict((row["task_id"], row) for row in loaded["verify_local"][1]),
                                  gold, official.exact_match_score)
    mechanism = {}
    if args.probe_run:
        groups = {}
        for name in names:
            for path in (args.probe_run / "policies" / name).glob("*/selection.json"):
                row = json.loads(path.read_text())
                groups.setdefault(row["task_id"], {})[name] = row
        changed_local = changed_broad = triggered = 0
        for task_id, rows in groups.items():
            if "verify_local" in rows and "continue_dynamic" in rows:
                changed_local += rows["verify_local"]["third_doc_id"] != rows["continue_dynamic"]["third_doc_id"]
                triggered += rows["verify_local"]["target"] is not None
            if "verify_local" in rows and "verify_broad" in rows:
                changed_broad += rows["verify_local"]["third_doc_id"] != rows["verify_broad"]["third_doc_id"]
        mechanism = {"local_vs_continue_third_doc": changed_local, "local_vs_broad_third_doc": changed_broad,
                     "localizable_nondefer": triggered, "n": len(groups)}
    headroom = json.loads(args.headroom.read_text()) if args.headroom else None
    if headroom and headroom.get("protocol") != "phase2b-OFFLINE-headroom-v1":
        raise ProtocolError("Headroom file is not the offline audit")
    gates = {}
    if headroom:
        gates["natural_errors"] = headroom["counts"].get("valid_wrong", 0) >= 15
        gates["one_extra_doc_proxy"] = headroom["counts"].get("one_extra_doc_support_proxy", 0) >= 10
    if mechanism:
        gates["localizable"] = mechanism["localizable_nondefer"] >= 12
        gates["local_changes_document"] = mechanism["local_vs_continue_third_doc"] >= 10
        gates["local_differs_from_broad"] = mechanism["local_vs_broad_third_doc"] >= 8
    positive = bool(primary and primary["delta"] > 0 and primary["wrong_to_right"] > primary["right_to_wrong"])
    decision = "NO-GO" if not gates or not all(gates.values()) or not positive else "DEVELOPMENT_GATES_PASSED"
    if args.mode == "development" and decision != "NO-GO":
        decision = "DEVELOPMENT_ONLY_NOT_CONFIRMATION"
    body = {"protocol": "phase2b-report-v1", "mode": args.mode, "decision": decision, "gates": gates,
            "primary_exploratory": primary, "secondary_exploratory": secondary, "mechanism": mechanism,
            "headroom_counts": None if headroom is None else headroom["counts"],
            "note": "Development intervals are not a confirmation. No 800-question set is authorized by this file."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"decision": decision, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
