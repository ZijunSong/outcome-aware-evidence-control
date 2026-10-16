"""Shared final reader for the phase4 shared-first family. Gold is not an input."""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pydantic import ValidationError
from oec.frozen_pool import sha
from oec.reader_v2 import ContextReader
from oec.runtime import BudgetExceeded, HTTPFailure, ProtocolError, Runtime, digest
from oec.types import Limits
from scripts.replay_phase1e_reader import indexed, lines


def canonical(task, corpus, doc_ids):
    if len(doc_ids) != 2 or len(set(doc_ids)) != 2 or not set(doc_ids) <= set(task["corpus_ids"]):
        raise ProtocolError("Final context must be two assigned frozen-pool documents")
    documents = [{"document_ref": f"d{index + 1}", "title": corpus[doc_id]["title"],
                  "sentences": [{"sentence_id": sid, "text": sentence}
                                for sid, sentence in enumerate(corpus[doc_id]["sentences"])]}
                 for index, doc_id in enumerate(sorted(doc_ids))]
    return {"question": task["question"], "documents": documents}


def collect_branch(root, branch, tasks, corpus, k, shared_first):
    directory = root / "branches" / branch
    predictions = indexed(lines(directory / "predictions.jsonl"), "task_id")
    if set(predictions) != set(tasks) or len(list(directory.glob("*"))) < len(tasks):
        raise ProtocolError("Missing or extra branch outputs")
    output = {}
    for task_id, task in tasks.items():
        path = directory / task_id
        bundle = json.loads((path / "reader_context.json").read_text())
        result = json.loads((path / "result.json").read_text())
        selection = json.loads((path / "selection.json").read_text())
        prefix = json.loads((root / "prefix" / task_id / "checkpoint.json").read_text())
        first = prefix["first_document"]
        expected_first = corpus[first["id"]]
        if first["title"] != expected_first["title"] or first["sentences"] != expected_first["sentences"]:
            raise ProtocolError("Shared first document differs from the frozen corpus")
        if result != predictions[task_id] or result["status"] != "SELECTION_ONLY_NOT_EVALUATED":
            raise ProtocolError("Selection result is not a deferred output")
        if selection["logical_reads"] != k or selection["observation_after_second"]:
            raise ProtocolError("A branch observed after the second document or did not read K=2")
        if selection["first_doc_id"] != shared_first[task_id]["first_doc_id"]:
            raise ProtocolError("Second-hop branch changed the shared first document")
        selected = selection["selected_sequence"]
        if selected[0] != selection["first_doc_id"] or selected[1] != selection["second_doc_id"]:
            raise ProtocolError("Selected sequence is not the shared first plus one remaining document")
        events = lines(path / "events.jsonl")
        if any(row["event"] == "invalid_run" for row in events):
            raise ProtocolError("Upstream selection protocol failed")
        if any(row["event"] == "observation_skipped" for row in events):
            raise ProtocolError("Second-hop observation is forbidden")
        reads = [row["response"] for row in events if row["event"] == "response" and row.get("kind") == "read"]
        if len(reads) != 1 or reads[0]["id"] != selected[1]:
            raise ProtocolError("Second document read does not match the selected remaining document")
        expected = corpus[reads[0]["id"]]
        if reads[0]["title"] != expected["title"] or reads[0]["sentences"] != expected["sentences"]:
            raise ProtocolError("Read text differs from the frozen corpus")
        for row in events:
            if row["event"] == "local_typed_response" and any(
                    value.get("truncated") for value in row["raw_results"].values()):
                raise ProtocolError("Native decision context truncated")
        context = canonical(task, corpus, selected)
        if bundle["context"] != context or bundle["context_hash"] != digest(context):
            raise ProtocolError("Reader context is not the canonical selected source text")
        if bundle["selected_doc_ids"] != sorted(selected):
            raise ProtocolError("Reader selected set mismatch")
        output[task_id] = {"context": context, "selection_result": result,
                           "selected_sequence": selected, "selection": selection}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-run", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--read-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.read_k != 2:
        parser.error("Use a new output directory and K=2")
    manifest = json.loads((args.selection_run / "manifest.json").read_text())
    if manifest.get("protocol") != "phase4-shared-first-selection-v1" or not manifest.get("completed"):
        raise ProtocolError("Expected a completed phase4 shared-first run")
    if manifest.get("selection_family") != "phase4-shared-first-selection-v1":
        raise ProtocolError("Do not replay a different family through this reader")
    if sha(args.tasks) != manifest["tasks_sha256"] or sha(args.corpus) != manifest["corpus_sha256"]:
        raise ProtocolError("Selection input hash mismatch")
    if sha(args.identity) != manifest["model_identity_sha256"]:
        raise ProtocolError("Reader and selection model identity mismatch")
    tasks, corpus = indexed(lines(args.tasks), "id"), indexed(lines(args.corpus), "id")
    if set(tasks) != set(manifest["task_ids"]):
        raise ProtocolError("Assigned task ids differ")
    shared_first = {row["task_id"]: row for row in manifest["prefixes"]}
    if set(shared_first) != set(tasks):
        raise ProtocolError("Shared first-hop ledger is incomplete")
    collected = {branch: collect_branch(args.selection_run, branch, tasks, corpus, args.read_k, shared_first)
                 for branch in manifest["branches"]}
    model = os.environ.get("OEC_LLM_MODEL", "Qwen3-14B")
    identity = json.loads(args.identity.read_text())
    if identity.get("llm_served_model", model) != model:
        raise ProtocolError("Served model does not match the frozen identity")
    base_url = os.environ.get("OEC_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    prototype = ContextReader(None, base_url, model, sha(args.identity), seed=args.seed)
    reader_policy = prototype.signature() | {
        "reader_source_sha256": sha(Path(__file__).resolve().parents[1] / "oec/reader_v2.py")}
    pool = {}
    keys = {branch: {} for branch in collected}
    for branch, values in collected.items():
        for task_id, row in values.items():
            key = digest({"reader_policy": reader_policy, "context": row["context"]})
            pool[key] = row["context"]
            keys[branch][task_id] = key
    args.out.mkdir(parents=True)
    limits = Limits(search_calls=1, read_calls=1, llm_calls=8, decision_calls=5,
                    repair_search_reserve=0, repair_read_reserve=0,
                    terminal_llm_reserve=6, terminal_decision_reserve=1)
    entries = {}
    for key in sorted(pool):
        runtime = Runtime(args.out / "reader_pool" / key, limits)
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
                     usage=dict(runtime.used) | dict(runtime.tokens), reader_policy=reader_policy)
        (runtime.path.parent / "entry.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
        entries[key] = entry
    ledger_hash = digest({key: digest(entry) for key, entry in sorted(entries.items())})
    arms, any_invalid = [], False
    for branch in manifest["branches"]:
        directory = args.out / "branches" / branch
        directory.mkdir(parents=True)
        predictions, invalid = [], 0
        for task_id in sorted(tasks):
            entry = entries[keys[branch][task_id]]
            answer, valid = entry["answer"], entry["runtime_valid"]
            documents = {item["document_ref"]: item for item in collected[branch][task_id]["context"]["documents"]}
            titles = [document["title"] for document in collected[branch][task_id]["context"]["documents"]]
            status = ("ANSWER_ATTEMPTED_ABSTAINED" if answer["abstain"] else "ANSWER_ATTEMPTED_UNVERIFIED") \
                if valid else "INVALID_RUNTIME_EVALUATION"
            row = {"task_id": task_id, "answer": answer["answer"].strip() if valid else "",
                   "supporting_facts": sorted({(documents[item["document_ref"]]["title"], item["sentence_id"])
                                               for item in answer["citations"]}) if valid else [],
                   "runtime_valid": valid, "verified": False, "status": status, "repairs": 0,
                   "failure_stage": None if valid else "bounded_reader_output",
                   "selected_doc_ids": collected[branch][task_id]["selected_sequence"],
                   "selected_titles": titles, "branch": branch, "shared_reader_key": keys[branch][task_id],
                   "first_observation_status": collected[branch][task_id]["selection"]["observation_status"],
                   "usage": Counter()}
            predictions.append(row)
            invalid += int(not valid)
        (directory / "predictions.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8")
        arm = {"protocol": "phase4-shared-reader-evaluation-v1", "branch": branch,
               "reader_ledger_sha256": ledger_hash, "invalid_runs": invalid, "reader_policy": reader_policy,
               "selection_manifest": manifest, "task_ids": list(sorted(tasks))}
        (directory / "manifest.json").write_text(json.dumps(arm, ensure_ascii=False, indent=2), encoding="utf-8")
        arms.append({"branch": branch, "invalid_runs": invalid})
        any_invalid |= bool(invalid)
    report = {"protocol": "phase4-shared-reader-evaluation-v1", "n": len(tasks), "branches": arms,
              "unique_reader_contexts": len(entries), "reader_ledger_sha256": ledger_hash,
              "selection_run": str(args.selection_run), "all_final_reader_valid": not any_invalid}
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("n", "unique_reader_contexts", "all_final_reader_valid")}, indent=2))
    if any_invalid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
