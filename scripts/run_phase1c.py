"""Separate stage-1 protocol. Defaults to the user's local SearchJev route."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pydantic import ValidationError
from oec.phase1c import (ProbePlan, RoleJSONClient, SelectionOnly, Selector, solve_documents)
from oec.retrieval import LocalBM25
from oec.runtime import BudgetExceeded, ProtocolError, Runtime, digest
from oec.types import Limits, Result, Task


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prepare_plans", "selection", "all_context", "oracle_context"], default="selection")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plans", type=Path)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--backend", choices=["searchjev", "typesafe"], default="searchjev")
    parser.add_argument("--local-model-run")
    parser.add_argument("--method", choices=["rank", "relevance", "utility_static", "utility_dynamic"], default="rank")
    parser.add_argument("--read-k", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.read_k <= 10:
        parser.error("read-k must be 1..10 for this provided-context protocol")
    if args.mode == "selection" and args.plans is None:
        parser.error("Generate a shared --plans file first")
    if args.mode == "selection" and args.backend == "searchjev" and not args.local_model_run:
        parser.error("The local SearchJev route requires --local-model-run")
    if args.mode == "oracle_context" and args.gold is None:
        parser.error("oracle_context is an OFFLINE diagnostic and requires --gold")
    if args.mode != "oracle_context" and args.gold is not None:
        parser.error("Gold is forbidden in the online selection and all-context modes")
    tasks = [Task.model_validate_json(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    if not tasks or len({task.id for task in tasks}) != len(tasks):
        parser.error("Tasks must be non-empty with unique ids")
    if any(task.corpus_ids is None or not task.corpus_ids for task in tasks):
        parser.error("Phase1c requires provided-context task scopes; it is not a FullWiki runner")
    args.out.mkdir(parents=True, exist_ok=False)
    read_limit = args.read_k if args.mode == "selection" else max(len(task.corpus_ids) for task in tasks)
    limits = Limits(search_calls=1, read_calls=read_limit, llm_calls=64, decision_calls=200,
                    repair_search_reserve=0, repair_read_reserve=0,
                    terminal_llm_reserve=6, terminal_decision_reserve=1)
    retriever = LocalBM25(args.corpus)
    signature_runtime = Runtime(args.out / "protocol", limits)
    signature = RoleJSONClient(signature_runtime).signature()
    manifest = {"protocol": "phase1c-fixed-selection-v1", "mode": args.mode, "method": args.method,
                "read_k": args.read_k, "tasks_sha256": file_hash(args.tasks),
                "corpus_sha256": file_hash(args.corpus), "limits": limits.model_dump(),
                "llm_protocol": signature, "task_ids": [task.id for task in tasks],
                "source_sha256": file_hash(Path(__file__).resolve().parents[1] / "oec" / "phase1c.py"),
                "retriever_kind": "provided-context-local-bm25",
                "online_gold_access": False, "offline_oracle": args.mode == "oracle_context"}
    if args.mode == "oracle_context":
        manifest["offline_gold_sha256"] = file_hash(args.gold)
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.mode == "prepare_plans":
        rows = {}
        for index, task in enumerate(tasks):
            runtime = Runtime(args.out / f"plan_{index:05d}", limits)
            client = RoleJSONClient(runtime)
            plan = client.ask("phase1_plan", {"question": task.question}, ProbePlan)
            rows[task.id] = {"question_hash": digest(task.question), "plan": plan.model_dump()}
        (args.out / "plans.json").write_text(json.dumps({"llm_protocol": signature, "tasks": rows},
                                                        ensure_ascii=False, indent=2))
        return
    plans = None
    if args.mode == "selection":
        plans = json.loads(args.plans.read_text())
        if plans["llm_protocol"] != signature or set(plans["tasks"]) != {task.id for task in tasks}:
            raise ProtocolError("Shared plan model/prompt settings or task ids do not match")
        manifest["plans_sha256"] = file_hash(args.plans)
        (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    gold = None
    if args.mode == "oracle_context":
        gold = {row["_id"]: row for row in json.loads(args.gold.read_text())}
        if not {task.id for task in tasks} <= set(gold):
            raise ProtocolError("Oracle diagnostics require gold for every task id")
    backend = None
    with (args.out / "predictions.jsonl").open("x", encoding="utf-8") as writer:
        for index, task in enumerate(tasks):
            runtime = Runtime(args.out / f"task_{index:05d}", limits)
            llm = RoleJSONClient(runtime)
            kernel = None
            try:
                if args.mode == "selection":
                    cached = plans["tasks"][task.id]
                    if cached["question_hash"] != digest(task.question):
                        raise ProtocolError("Shared plan question mismatch")
                    plan = ProbePlan.model_validate(cached["plan"])
                    runtime.event("shared_plan_loaded", plans_sha256=manifest["plans_sha256"], plan=plan.model_dump())
                    if backend is None:
                        if args.backend == "searchjev":
                            from oec.local_searchjev import LocalSearchJev
                            backend = LocalSearchJev(runtime, args.local_model_run)
                        else:
                            from oec.jev import TypesafeJev
                            backend = TypesafeJev(runtime)
                    backend.runtime = runtime
                    kernel = SelectionOnly(llm, Selector(backend, runtime), retriever, runtime, args.method, args.read_k)
                    try:
                        result = kernel.run(task, plan)
                    except BudgetExceeded:
                        runtime.event("normal_budget_stop", reason="Closing with reserved answer calls")
                        result = kernel.finalize(task, kernel.memory)
                else:
                    documents = [retriever.documents[key] for key in task.corpus_ids]
                    if args.mode == "oracle_context":
                        needed_titles = {title for title, sentence_id in gold[task.id]["supporting_facts"]}
                        documents = [document for document in documents if document.title in needed_titles]
                        if {document.title for document in documents} != needed_titles:
                            raise ProtocolError("Oracle supporting document missing from public context")
                        runtime.event("OFFLINE_ORACLE_DIAGNOSTIC", supporting_titles=sorted(needed_titles))
                    result = solve_documents(task, documents, llm, runtime)
            except (ProtocolError, ValidationError, OSError, BudgetExceeded) as error:
                runtime.event("invalid_run", error_type=type(error).__name__, error=str(error))
                result = Result(task_id=task.id, answer="", supporting_facts=[], runtime_valid=False,
                                verified=False, status="INVALID_RUNTIME_EVALUATION", repairs=0,
                                usage=dict(runtime.used) | dict(runtime.tokens))
            payload = result.model_dump(mode="json")
            events = [json.loads(line) for line in runtime.path.read_text().splitlines()]
            payload["answer_attempted"] = any(row["event"] == "answer_attempt" for row in events)
            (runtime.path.parent / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            writer.write(json.dumps(payload, ensure_ascii=False) + "\n")
            writer.flush()
            print(json.dumps({"task_id": task.id, "status": result.status, "answer": result.answer}, ensure_ascii=False))


if __name__ == "__main__":
    main()
