"""One shared first document, then current_once vs initial_once. No gold."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oec.frozen_pool import load_manifest, sha
from oec.phase1c import ProbePlan
from oec.phase3_observer import SharedObserver
from oec.phase4_shared import SharedFirstHop
from oec.retrieval import LocalBM25
from oec.runtime import ProtocolError, Runtime, digest
from oec.types import Limits, Result, Task


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--local-model-run", type=Path, required=True)
    parser.add_argument("--observer-cache", type=Path, required=True)
    parser.add_argument("--branches", nargs="+", default=["current_once", "initial_once"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Refusing to overwrite a selection run")
    if args.branches != ["current_once", "initial_once"]:
        parser.error("This runner compares current_once and initial_once")
    identity = json.loads(args.identity.read_text())
    if not isinstance(identity, dict) or not identity:
        parser.error("Need a nonempty frozen model identity")
    identity_hash = file_hash(args.identity)
    tasks = [Task.model_validate_json(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    if not tasks or len({task.id for task in tasks}) != len(tasks):
        parser.error("Tasks must be unique")
    pool_manifest = load_manifest(args.pool)
    if file_hash(args.corpus) != pool_manifest.get("corpus_sha256"):
        raise ProtocolError("Corpus is not the frozen pool export")
    plans = json.loads(args.plans.read_text())
    if set(plans["tasks"]) != {task.id for task in tasks}:
        raise ProtocolError("Shared plan task ids do not match")
    from oec.phase1c import RoleJSONClient
    expected_protocol = RoleJSONClient.__new__(RoleJSONClient)
    expected_protocol.model = os.environ.get("OEC_LLM_MODEL", "Qwen3-14B")
    expected_protocol.base_url = os.environ.get("OEC_LLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    expected_protocol.schema_mode = os.environ.get("OEC_JSON_MODE", "json_schema")
    if plans.get("llm_protocol") != expected_protocol.signature():
        raise ProtocolError("Shared plan model/prompt settings do not match the live observer")
    retriever = LocalBM25(args.corpus)
    from oec.local_searchjev import LocalSearchJev
    args.out.mkdir(parents=True)
    limits = Limits(search_calls=1, read_calls=2, llm_calls=8, decision_calls=200,
                    repair_search_reserve=0, repair_read_reserve=0,
                    terminal_llm_reserve=2, terminal_decision_reserve=4)
    backend = LocalSearchJev(Runtime(args.out / "model_load", limits), args.local_model_run)
    prefixes, ledgers = [], {name: [] for name in args.branches}
    predicate = ("The visible candidate is relevant to answering the question given the "
                 "supplied current evidence state and unresolved target")
    for index, task in enumerate(sorted(tasks, key=lambda row: row.id)):
        cached = plans["tasks"][task.id]
        if cached["question_hash"] != digest(task.question):
            raise ProtocolError("Shared plan question mismatch")
        plan = ProbePlan.model_validate(cached["plan"])
        prefix_runtime = Runtime(args.out / "prefix" / task.id, limits)
        backend.runtime = prefix_runtime
        observer = SharedObserver(prefix_runtime, args.observer_cache, identity_hash)
        hop = SharedFirstHop(backend, observer, retriever, prefix_runtime, args.pool, pool_manifest, identity_hash)
        checkpoint, initial_memory, updated_memory, candidates, scores, first = hop.prefix(task, plan)
        checkpoint["checkpoint_sha256"] = digest({key: value for key, value in checkpoint.items()
                                                  if key != "checkpoint_sha256"})
        (prefix_runtime.path.parent / "checkpoint.json").write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
        prefixes.append({"task_id": task.id, "first_doc_id": first.id,
                         "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                         "observation_status": checkpoint["observation_status"]})
        for branch in args.branches:
            runtime = Runtime(args.out / "branches" / branch / task.id, limits)
            backend.runtime = runtime
            hop.runtime = runtime
            record = hop.second(task, plan, checkpoint, initial_memory, updated_memory,
                                candidates, scores, first, branch)
            (runtime.path.parent / "selection.json").write_text(
                json.dumps({key: value for key, value in record.items() if key != "result"},
                           ensure_ascii=False, indent=2), encoding="utf-8")
            (runtime.path.parent / "reader_context.json").write_text(
                json.dumps(record["bundle"], ensure_ascii=False, indent=2), encoding="utf-8")
            (runtime.path.parent / "result.json").write_text(
                json.dumps(record["result"], ensure_ascii=False, indent=2), encoding="utf-8")
            ledgers[branch].append(record["result"])
            print(json.dumps({"task_id": task.id, "branch": branch, "second_doc_id": record["second_doc_id"],
                              "status": record["result"]["status"]}, ensure_ascii=False))
    root = Path(__file__).resolve().parents[1]
    observer_files = sorted((args.observer_cache).glob("*.json")) if args.observer_cache.exists() else []
    observer_ledger = digest({path.stem: sha(path) for path in observer_files})
    shared_first = digest(prefixes)
    sources = ("oec/phase4_shared.py", "oec/phase4_single.py", "oec/frozen_pool.py",
               "scripts/run_phase4_shared_first.py", "oec/phase3_observer.py", "oec/local_searchjev.py")
    for branch in args.branches:
        (args.out / "branches" / branch / "predictions.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ledgers[branch]), encoding="utf-8")
    manifest = {
        "protocol": "phase4-shared-first-selection-v1", "completed": True, "final_answers_deferred": True,
        "prediction_file_is_evaluable": False, "read_k": 2, "branches": list(args.branches),
        "task_ids": [task.id for task in sorted(tasks, key=lambda row: row.id)],
        "tasks_sha256": file_hash(args.tasks), "corpus_sha256": file_hash(args.corpus),
        "plans_sha256": file_hash(args.plans),
        "pool_manifest_sha256": sha(args.pool / "manifest.json"),
        "pool_files_sha256": pool_manifest["pool_files_sha256"],
        "shared_first_ledger_sha256": shared_first,
        "observer_ledger_sha256": observer_ledger,
        "model_identity_sha256": identity_hash,
        "observer_ledger": str(args.observer_cache.resolve()),
        "weights": [0, 0], "scorer_predicate_hash": digest(predicate),
        "native_example_profile": "one noul per candidate; first hop scored once; initial reuses those scores; "
                                  "current rescores remaining only if the observed state changed",
        "online_gold_access": False, "offline_oracle": False,
        "source_sha256": {name: file_hash(root / name) for name in sources},
        "llm_protocol": SharedObserver(Runtime(args.out / "protocol", limits), args.observer_cache, identity_hash).signature(),
        "selection_family": "phase4-shared-first-selection-v1",
        "prefixes": prefixes,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n": len(tasks), "shared_first_ledger_sha256": shared_first,
                      "observation_failures": sum(row["observation_status"] != "success" for row in prefixes)}, indent=2))


if __name__ == "__main__":
    main()
