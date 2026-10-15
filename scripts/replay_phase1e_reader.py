"""Gold-free shared reader. One first bounded response per full context/policy key."""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pydantic import ValidationError
from oec.reader_v2 import ContextReader
from oec.runtime import BudgetExceeded, HTTPFailure, ProtocolError, Runtime, digest
from oec.types import Limits


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def indexed(values, key):
    output = {row[key]: row for row in values}
    if len(output) != len(values):
        raise ProtocolError("Duplicate ids")
    return output


def collect(run, tasks, corpus, tasks_path, corpus_path, k):
    manifest = json.loads((run/"manifest.json").read_text())
    if manifest.get("protocol") != "phase1e-selection-deferred-v1" or not manifest.get("completed"):
        raise ProtocolError("Expected a completed deferred-selection run")
    if manifest["tasks_sha256"] != sha(tasks_path) or manifest["corpus_sha256"] != sha(corpus_path):
        raise ProtocolError("Selection input hash mismatch")
    if manifest.get("online_gold_access") is not False or manifest.get("offline_oracle"):
        raise ProtocolError("Offline/oracle runs forbidden")
    if manifest["read_k"] != k or set(manifest["task_ids"]) != set(tasks):
        raise ProtocolError("Selection K or task ids mismatch")
    directories = list(run.glob("task_*"))
    if len(directories) != len(tasks):
        raise ProtocolError("Missing or extra task outputs")
    predictions = indexed(lines(run/"predictions.jsonl"), "task_id")
    if set(predictions) != set(tasks):
        raise ProtocolError("Missing or extra selection ids")
    output = {}
    for directory in directories:
        bundle = json.loads((directory/"reader_context.json").read_text())
        task_id = bundle["task_id"]
        if task_id in output or task_id not in tasks:
            raise ProtocolError("Duplicate or unknown selection context")
        task = tasks[task_id]
        if len(task["corpus_ids"]) != 10 or len(set(task["corpus_ids"])) != 10:
            raise ProtocolError("Expected 10 public candidates")
        result = json.loads((directory/"result.json").read_text())
        if result != predictions[task_id] or result["status"] != "SELECTION_ONLY_NOT_EVALUATED":
            raise ProtocolError("Selection result is not a deferred output")
        events = lines(directory/"events.jsonl")
        if any(row["event"] == "invalid_run" for row in events):
            raise ProtocolError("Upstream selection protocol failed")
        selected = [row["doc_id"] for row in events if row["event"] == "phase1c_selected"]
        if not bundle["selection_stage_valid"] or len(set(selected)) != k or len(selected) != k:
            raise ProtocolError("Selection did not read exactly K unique documents")
        if result["usage"].get("read") != k or not set(selected) <= set(task["corpus_ids"]):
            raise ProtocolError("Read count or public scope mismatch")
        read_responses = [row["response"] for row in events
                          if row["event"] == "response" and row.get("kind") == "read"]
        if len(read_responses) != k or {row["id"] for row in read_responses} != set(selected):
            raise ProtocolError("Read events differ from selected documents")
        for row in read_responses:
            expected = corpus[row["id"]]
            if row["title"] != expected["title"] or row["sentences"] != expected["sentences"]:
                raise ProtocolError("Read text differs from frozen corpus")
        for row in events:
            if row["event"] == "local_typed_response" and any(
                value.get("truncated") for value in row["raw_results"].values()):
                raise ProtocolError("Native decision context truncated")
        documents = [corpus[key] for key in sorted(selected)]
        context = {"question": task["question"], "documents": [{"document_ref": f"d{index+1}",
            "title": document["title"], "sentences": [{"sentence_id": sid, "text": sentence}
            for sid, sentence in enumerate(document["sentences"])]} for index, document in enumerate(documents)]}
        if bundle["context"] != context or bundle["context_hash"] != digest(context):
            raise ProtocolError("Reader context is not the canonical selected source text")
        if bundle["selected_doc_ids"] != sorted(selected):
            raise ProtocolError("Reader selected set mismatch")
        output[task_id] = {"context": context, "selection_result": result,
                           "selected_sequence": selected, "source_directory": str(directory)}
    return manifest, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--read-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or len(set(map(str, args.runs))) != len(args.runs):
        parser.error("Use a new output directory and unique input runs")
    identity = json.loads(args.identity.read_text())
    if not identity or not isinstance(identity, dict):
        parser.error("--identity must contain the frozen, nonempty model/environment identity object")
    tasks, corpus = indexed(lines(args.tasks), "id"), indexed(lines(args.corpus), "id")
    if not tasks:
        parser.error("No tasks")
    collected = [collect(run, tasks, corpus, args.tasks, args.corpus, args.read_k) for run in args.runs]
    common_keys = ("tasks_sha256", "corpus_sha256", "plans_sha256", "limits", "read_k",
                   "llm_protocol", "source_sha256", "selection_family", "additional_source_sha256")
    for manifest, _ in collected[1:]:
        if any(manifest.get(key) != collected[0][0].get(key) for key in common_keys):
            raise ProtocolError("Mixed selection protocols or model/plan signatures")
    base_url = os.environ.get("OEC_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    model = os.environ.get("OEC_LLM_MODEL", "Qwen3-14B")
    if collected[0][0]["llm_protocol"]["model"] != model:
        raise ProtocolError("Reader and selection observer must use the frozen generator model")
    args.out.mkdir(parents=True)
    limits = Limits(search_calls=1, read_calls=1, llm_calls=8, decision_calls=5,
                    repair_search_reserve=0, repair_read_reserve=0,
                    terminal_llm_reserve=6, terminal_decision_reserve=1)
    prototype = ContextReader(None, base_url, model, sha(args.identity), seed=args.seed)
    policy = prototype.signature() | {"reader_source_sha256": sha(
        Path(__file__).resolve().parents[1]/"oec/reader_v2.py")}
    pool, keys_by_arm = {}, []
    for manifest, values in collected:
        keys = {}
        for task_id, row in values.items():
            key = digest({"reader_policy": policy, "context": row["context"]})
            pool[key] = row["context"]
            keys[task_id] = key
        keys_by_arm.append(keys)
    entries = {}
    for key in sorted(pool):  # independent of method order or answer correctness
        runtime = Runtime(args.out/"reader_pool"/key, limits)
        reader = ContextReader(runtime, base_url, model, sha(args.identity),
                               os.environ.get("OEC_LLM_API_KEY", "local"), args.seed)
        try:
            answer = reader.read(pool[key])
            entry = {"runtime_valid": True, "answer": answer.model_dump(), "error": None}
        except HTTPFailure as error:
            if error.status in (400, 401, 403, 404, 422):
                raise ProtocolError("Reader service/schema configuration failed; abort study") from error
            entry = {"runtime_valid": False, "answer": None,
                     "error": {"type": type(error).__name__, "message": str(error)}}
        except (ProtocolError, OSError, BudgetExceeded, ValidationError) as error:
            entry = {"runtime_valid": False, "answer": None,
                     "error": {"type": type(error).__name__, "message": str(error)}}
        entry.update(key=key, context_hash=digest(pool[key]), context=pool[key],
                     usage=dict(runtime.used) | dict(runtime.tokens), reader_policy=policy)
        (runtime.path.parent/"entry.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2))
        entries[key] = entry
    ledger_hash = digest({key: digest(entry) for key, entry in sorted(entries.items())})
    arms, any_invalid = [], False
    for arm_index, ((manifest, values), keys) in enumerate(zip(collected, keys_by_arm)):
        directory = args.out/"arms"/f"arm_{arm_index:02d}"
        directory.mkdir(parents=True)
        predictions, invalid = [], 0
        for task_id in tasks:
            entry = entries[keys[task_id]]
            answer = entry["answer"]
            documents = {document["document_ref"]: document for document in values[task_id]["context"]["documents"]}
            valid = entry["runtime_valid"]
            status = ("ANSWER_ATTEMPTED_ABSTAINED" if answer["abstain"] else "ANSWER_ATTEMPTED_UNVERIFIED") \
                     if valid else "INVALID_RUNTIME_EVALUATION"
            usage = Counter(values[task_id]["selection_result"]["usage"])
            usage.update(entry["usage"])
            row = {"task_id": task_id, "answer": answer["answer"].strip() if valid else "",
                "supporting_facts": sorted({(documents[item["document_ref"]]["title"], item["sentence_id"])
                                          for item in answer["citations"]}) if valid else [],
                "runtime_valid": valid, "verified": False, "status": status, "repairs": 0,
                "answer_attempted": True, "usage": dict(usage), "shared_reader_key": keys[task_id],
                "failure_stage": None if valid else "bounded_reader_output",
                "selected_doc_ids": sorted(values[task_id]["selected_sequence"]),
                "selected_titles": sorted(document["title"] for document in documents.values()),
                "answer_execution": "one shared first bounded response per context-policy key"}
            predictions.append(row)
            invalid += int(not valid)
        (directory/"predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in predictions))
        arm_manifest = {"protocol": "phase1e-shared-reader-evaluation-v1", "selection_run": str(args.runs[arm_index]),
            "selection_manifest": manifest, "reader_policy": policy, "task_ids": list(tasks),
            "reader_ledger_sha256": ledger_hash,
            "invalid_runs": invalid, "physical_reader_cost": "count the reader_pool entries once, not each arm usage",
            "reader_keys": keys}
        (directory/"manifest.json").write_text(json.dumps(arm_manifest, ensure_ascii=False, indent=2))
        arms.append({"arm": str(directory), "source_run": str(args.runs[arm_index]), "invalid_runs": invalid})
        any_invalid |= bool(invalid)
    report = {"protocol": "phase1e-shared-reader-evaluation-v1", "n": len(tasks), "arms": arms,
              "unique_reader_contexts": len(entries), "logical_context_uses": len(tasks)*len(arms),
              "reader_policy": policy, "all_runtime_valid": not any_invalid,
              "reader_ledger_sha256": ledger_hash,
              "note": "No gold was used; failures are retained; this is a new reader protocol, not repair of old scores."}
    (args.out/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if any_invalid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
