import argparse
import json
import os
from pathlib import Path
from pydantic import ValidationError
from .controller import Controller
from .demo import DemoJev, DemoLLM, DemoRetriever
from .jev import Decisions, TypesafeJev
from .llm import ChatLLM
from .retrieval import LocalBM25, Wiki18Retriever
from .runtime import BudgetExceeded, ProtocolError, Runtime
from .types import Result, Settings, Task


def run_one(task, directory, settings, backend, retriever, local_model_run=None, shared_backends=None):
    runtime = Runtime(directory, settings.limits)
    if backend == "demo":
        llm, decision_backend = DemoLLM(runtime), DemoJev(runtime)
    else:
        llm = ChatLLM(runtime)
        if backend == "typesafe":
            decision_backend = TypesafeJev(runtime)
        elif backend == "searchjev":
            from .local_searchjev import LocalSearchJev
            shared_backends = {} if shared_backends is None else shared_backends
            if "searchjev" not in shared_backends:
                shared_backends["searchjev"] = LocalSearchJev(runtime, local_model_run)
            decision_backend = shared_backends["searchjev"]
            decision_backend.runtime = runtime
        else:
            raise ValueError(backend)
    controller = Controller(llm, Decisions(decision_backend, runtime, settings), retriever, runtime, settings)
    try:
        result = controller.run(task)
    except (ProtocolError, ValidationError, OSError) as error:
        runtime.event("invalid_run", error_type=type(error).__name__, error=str(error))
        result = Result(task_id=task.id, answer="", supporting_facts=[], runtime_valid=False,
                        verified=False, status="INVALID_RUNTIME_EVALUATION", repairs=0,
                        usage=dict(runtime.used) | dict(runtime.tokens))
    (directory / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["demo", "typesafe", "searchjev"], default="demo")
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--method", choices=["rank", "relevance", "utility_static", "utility_dynamic", "full"])
    parser.add_argument("--no-gap-priority", action="store_true")
    parser.add_argument("--search-budget", type=int)
    parser.add_argument("--read-budget", type=int)
    parser.add_argument("--local-model-run")
    parser.add_argument("--retriever-url", default=os.environ.get("OEC_RETRIEVER_URL", ""),
                        help="Full Wikipedia E5 endpoint, for example http://127.0.0.1:18080")
    args = parser.parse_args()
    raw_settings = (json.loads(args.settings.read_text(encoding="utf-8"))
                    if args.settings else Settings().model_dump())
    if args.method is not None:
        raw_settings["method"] = args.method
    if args.no_gap_priority:
        raw_settings["gap_priority"] = False
    if args.search_budget is not None:
        raw_settings.setdefault("limits", {})["search_calls"] = args.search_budget
    if args.read_budget is not None:
        raw_settings.setdefault("limits", {})["read_calls"] = args.read_budget
    settings = Settings.model_validate(raw_settings)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "settings.json").write_text(settings.model_dump_json(indent=2), encoding="utf-8")
    if args.backend == "demo":
        if args.retriever_url:
            parser.error("The demo backend uses its scripted retriever")
        tasks = [Task(id="demo", question="Which city was A's doctoral advisor born in?")]
        retriever = DemoRetriever()
    else:
        if args.tasks is None:
            parser.error("A real backend requires --tasks")
        tasks = [Task.model_validate_json(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line.strip()]
        if args.retriever_url:
            retriever = Wiki18Retriever(args.retriever_url)
        elif args.corpus is not None:
            retriever = LocalBM25(args.corpus)
        else:
            parser.error("Provide --corpus for the small BM25 index, or --retriever-url for full Wikipedia")
    retriever_record = (retriever.describe() if isinstance(retriever, Wiki18Retriever)
                        else {"kind": "demo" if args.backend == "demo" else "local-bm25",
                              "corpus": None if args.corpus is None else str(args.corpus)})
    (args.out / "retriever.json").write_text(
        json.dumps(retriever_record, ensure_ascii=False, indent=2), encoding="utf-8")
    if len({task.id for task in tasks}) != len(tasks):
        raise ProtocolError("Duplicate task id")
    results, shared_backends = [], {}
    for index, task in enumerate(tasks):
        if isinstance(retriever, LocalBM25):
            retriever.set_scope(task.corpus_ids)
        result = run_one(task, args.out / f"task_{index:05d}", settings, args.backend, retriever,
                         args.local_model_run, shared_backends)
        results.append(result.model_dump(mode="json"))
        print(json.dumps({"id": task.id, "status": result.status, "answer": result.answer}, ensure_ascii=False))
    (args.out / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")


if __name__ == "__main__":
    main()
