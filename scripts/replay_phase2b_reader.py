"""One bounded final reader call per K=3 context. Gold is not an input."""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oec.reader_v2 import ContextReader
from oec.runtime import ProtocolError, Runtime, digest
from oec.types import Limits
from scripts.replay_phase1e_reader import indexed, lines, sha


def canonical(task, corpus, doc_ids):
    if len(doc_ids) != 3 or len(set(doc_ids)) != 3 or not set(doc_ids) <= set(task["corpus_ids"]):
        raise ProtocolError("Final context must be three assigned public documents")
    documents = []
    for index, doc_id in enumerate(sorted(doc_ids)):
        document = corpus[doc_id]
        documents.append({"document_ref": f"d{index + 1}", "title": document["title"],
                          "sentences": [{"sentence_id": sid, "text": sentence}
                                        for sid, sentence in enumerate(document["sentences"])]})
    return {"question": task["question"], "documents": documents}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-run", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a new reader directory")
    manifest = json.loads((args.probe_run / "manifest.json").read_text())
    if manifest.get("protocol") != "phase2b-selection-deferred-v1" or not manifest.get("completed"):
        raise ProtocolError("Expected a completed phase2b deferred selection")
    tasks, corpus = indexed(lines(args.tasks), "id"), indexed(lines(args.corpus), "id")
    if manifest["tasks_sha256"] != sha(args.tasks) or manifest["corpus_sha256"] != sha(args.corpus):
        raise ProtocolError("Probe inputs do not match")
    identity = json.loads(args.identity.read_text())
    model = os.environ.get("OEC_LLM_MODEL", "Qwen3-14B")
    expected = identity.get("llm_served_model") or identity.get("served", {}).get("model")
    if manifest["backend"] == "searchjev" and expected and expected != model:
        raise ProtocolError("Reader model does not match the frozen identity")
    prefix = json.loads((args.probe_run / "verifier.jsonl").read_text().splitlines()[0])
    verifier = {json.loads(line)["task_id"]: json.loads(line) for line in (args.probe_run / "verifier.jsonl").read_text().splitlines()}
    if set(verifier) != set(tasks):
        raise ProtocolError("Verifier rows do not match the assigned tasks")
    del prefix
    pool, by_policy = {}, {}
    for policy in manifest["policies"]:
        rows = {}
        for task_id in sorted(tasks):
            path = args.probe_run / "policies" / policy / task_id / "selection.json"
            if not path.is_file():
                raise ProtocolError("Missing policy selection")
            row = json.loads(path.read_text())
            if row["logical_reads"] != 3 or len(row["selected_doc_ids"]) != 3:
                raise ProtocolError("A policy read a number of documents other than three")
            context = canonical(tasks[task_id], corpus, row["selected_doc_ids"])
            if row["context"] != context:
                raise ProtocolError("Stored context differs from the frozen corpus text")
            if set(row["context"]) != {"question", "documents"}:
                raise ProtocolError("Reader context contains non-source fields")
            rows[task_id] = row
            pool[digest({"context": context})] = context
        by_policy[policy] = rows
    args.out.mkdir(parents=True)
    base_url = os.environ.get("OEC_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    prototype = ContextReader(None, base_url, model, sha(args.identity), seed=args.seed)
    reader_policy = prototype.signature() | {"reader_source_sha256": sha(Path(__file__).resolve().parents[1] / "oec/reader_v2.py")}
    limits = Limits(search_calls=1, read_calls=1, llm_calls=8, decision_calls=5,
                    repair_search_reserve=0, repair_read_reserve=0,
                    terminal_llm_reserve=6, terminal_decision_reserve=1)
    entries = {}
    for context_hash in sorted(pool):
        key = digest({"reader_policy": reader_policy, "context": pool[context_hash]})
        runtime = Runtime(args.out / "reader_pool" / key, limits)
        reader = ContextReader(runtime, base_url, model, sha(args.identity),
                               os.environ.get("OEC_LLM_API_KEY", "local"), args.seed)
        try:
            answer = reader.read(pool[context_hash])
            entry = {"runtime_valid": True, "answer": answer.model_dump(), "error": None}
        except ProtocolError as error:
            entry = {"runtime_valid": False, "answer": None,
                     "error": {"type": type(error).__name__, "message": str(error)},
                     "failure_stage": "bounded_reader_output"}
        entry.update(key=key, context=pool[context_hash], reader_policy=reader_policy)
        (runtime.path.parent / "entry.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
        entries[key] = entry
    ledger_hash = digest({key: digest(entry) for key, entry in sorted(entries.items())})
    arms, any_invalid = [], False
    for policy in manifest["policies"]:
        directory = args.out / "policies" / policy
        directory.mkdir(parents=True)
        predictions, invalid = [], 0
        for task_id in sorted(tasks):
            row = by_policy[policy][task_id]
            key = digest({"reader_policy": reader_policy, "context": row["context"]})
            entry = entries[key]
            answer, valid = entry["answer"], entry["runtime_valid"]
            documents = {item["document_ref"]: item for item in row["context"]["documents"]}
            prediction = {"task_id": task_id, "answer": answer["answer"].strip() if valid else "",
                          "supporting_facts": sorted({(documents[item["document_ref"]]["title"], item["sentence_id"])
                                                      for item in answer["citations"]}) if valid else [],
                          "runtime_valid": valid, "failure_stage": None if valid else "bounded_reader_output",
                          "selected_doc_ids": row["selected_doc_ids"], "policy": policy,
                          "shared_reader_key": key, "draft_runtime_valid": verifier[task_id]["draft_runtime_valid"],
                          "verifier_valid": verifier[task_id]["verifier_valid"],
                          "final_reader_valid": valid, "usage": Counter()}
            predictions.append(prediction)
            invalid += int(not valid)
        (directory / "predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8")
        arm = {"protocol": "phase2b-shared-reader-evaluation-v1", "policy": policy,
               "reader_ledger_sha256": ledger_hash, "invalid_runs": invalid, "reader_policy": reader_policy}
        (directory / "manifest.json").write_text(json.dumps(arm, ensure_ascii=False, indent=2), encoding="utf-8")
        arms.append({"policy": policy, "invalid_runs": invalid})
        any_invalid |= bool(invalid)
    report = {"protocol": "phase2b-shared-reader-evaluation-v1", "probe_manifest": manifest,
              "policies": arms, "reader_ledger_sha256": ledger_hash, "n": len(tasks),
              "unique_reader_contexts": len(entries), "all_final_reader_valid": not any_invalid}
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("n", "unique_reader_contexts", "all_final_reader_valid")}, indent=2))
    if any_invalid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
