"""Offline integrity, selection coverage, cost and paired mechanism reports."""
import argparse
import hashlib
import importlib.util
import json
import statistics
from collections import Counter
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def unique(values, key):
    result = {row[key]: row for row in values}
    if len(result) != len(values):
        raise ValueError(f"Duplicate ids: {key}")
    return result


def summarize(run, tasks, corpus, gold, official, expected_k):
    manifest = json.loads((run / "manifest.json").read_text())
    predictions = unique(rows(run / "predictions.jsonl"), "task_id")
    if set(predictions) != set(tasks) or set(gold) != set(tasks):
        raise ValueError("Predictions/tasks/gold must have identical ids")
    if manifest["tasks_sha256"] != sha(tasks.path) or manifest["corpus_sha256"] != sha(corpus.path):
        raise ValueError("Input hashes do not match manifest")
    if set(manifest["task_ids"]) != set(tasks):
        raise ValueError("Manifest id mismatch")
    if manifest.get("offline_oracle") or manifest.get("online_gold_access") is not False:
        raise ValueError("Only online gold-free selection runs are accepted")
    if manifest.get("mode") != "selection" or manifest.get("read_k") != expected_k:
        raise ValueError("Not the requested fixed-K selection protocol")
    if manifest.get("retriever_kind") != "provided-context-local-bm25":
        raise ValueError("Unexpected retrieval scope")
    directories = sorted(run.glob("task_*"))
    if len(directories) != len(tasks):
        raise ValueError("Missing or extra task output directories")
    output, seen = {}, set()
    for directory in directories:
        result = json.loads((directory / "result.json").read_text())
        task_id = result["task_id"]
        if task_id in seen or task_id not in tasks or result != predictions[task_id]:
            raise ValueError("Result/prediction mismatch or duplicate task trace")
        seen.add(task_id)
        task, expected = tasks[task_id], gold[task_id]
        if len(task["corpus_ids"]) != 10 or len(set(task["corpus_ids"])) != 10:
            raise ValueError("Expected 10 unique public candidate ids")
        if task["question"] != expected["question"]:
            raise ValueError("Question text mismatch")
        gold_context = {title: sentences for title, sentences in expected["context"]}
        if len(gold_context) != 10 or {corpus[key]["title"] for key in task["corpus_ids"]} != set(gold_context):
            raise ValueError("Public context titles do not match offline source context")
        for key in task["corpus_ids"]:
            document = corpus[key]
            if document["sentences"] != gold_context[document["title"]]:
                raise ValueError("Sentence ids/text differ from original Hotpot source")
        events = rows(directory / "events.jsonl")
        attempts = [event for event in events if event["event"] == "answer_attempt"]
        answer_requests = []
        for event in events:
            if event["event"] == "request" and event.get("kind") == "llm":
                request = event["request"]
                user = json.loads(request["messages"][-1]["content"])
                if user["schema"]["title"] == "ShortAnswer":
                    answer_requests.append(user["context"])
        if result["runtime_valid"] is not True or result.get("answer_attempted") is not True:
            raise ValueError("Invalid runtime or no explicit answer attempt")
        if len(attempts) != 1 or not answer_requests:
            raise ValueError("Expected one finalization and an actual answer HTTP request")
        selected_events = [event for event in events if event["event"] == "phase1c_selected"]
        selected = [event["doc_id"] for event in selected_events]
        if len(selected) != expected_k or len(set(selected)) != expected_k:
            raise ValueError("Wrong unique selected document count")
        if not set(selected) <= set(task["corpus_ids"]):
            raise ValueError("Selected document outside public scope")
        if result["usage"].get("read") != expected_k:
            raise ValueError("Read budget differs from K")
        if set(attempts[0]["selected_doc_ids"]) != set(selected):
            raise ValueError("Reader selected set differs from selector")
        ordered_docs = [corpus[key] for key in sorted(selected)]
        context = {"question": task["question"], "documents": [
            {"document_ref": f"d{index+1}", "title": document["title"],
             "sentences": [{"sentence_id": sid, "text": sentence}
                           for sid, sentence in enumerate(document["sentences"])]}
            for index, document in enumerate(ordered_docs)]}
        context_hash = canonical_hash(context)
        if attempts[0]["context_hash"] != context_hash or any(value != context for value in answer_requests):
            raise ValueError("Reader saw changed or unseen context")
        selected_titles = {document["title"] for document in ordered_docs}
        gold_support = {tuple(item) for item in expected["supporting_facts"]}
        support_titles = {title for title, _ in gold_support}
        if not support_titles <= set(gold_context):
            raise ValueError("Offline supporting title missing from public source")
        for title, sid in gold_support | {tuple(item) for item in result["supporting_facts"]}:
            if title not in gold_context or not isinstance(sid, int) or not 0 <= sid < len(gold_context[title]):
                raise ValueError("Illegal supporting sentence id")
        if any(title not in selected_titles for title, _ in result["supporting_facts"]):
            raise ValueError("Prediction cites an unread document")
        truncated = sum(bool(value.get("truncated")) for event in events
            if event["event"] == "local_typed_response" for value in event["raw_results"].values())
        if truncated:
            raise ValueError("Native decision input was truncated")
        native_requests = [event["request"] for event in events
            if event["event"] == "request" and event.get("kind") == "decision" and
            "lengths" in event["request"]]
        llm_responses = [event["response"] for event in events
            if event["event"] == "response" and event.get("kind") == "llm"]
        token_complete = bool(llm_responses) and all(
            ("prompt_tokens" in value.get("usage", {}) or "input_tokens" in value.get("usage", {})) and
            ("completion_tokens" in value.get("usage", {}) or "output_tokens" in value.get("usage", {}))
            for value in llm_responses)
        token_value = lambda value, key, alternative: value.get("usage", {}).get(key,
            value.get("usage", {}).get(alternative, 0))
        em = int(official.exact_match_score(result["answer"], expected["answer"]))
        f1 = float(official.f1_score(result["answer"], expected["answer"])[0])
        sp_metrics = Counter()
        sp_em, _, _ = official.update_sp(sp_metrics, result["supporting_facts"], expected["supporting_facts"])
        all_support = support_titles <= selected_titles
        output[task_id] = {"task_id": task_id, "type": expected.get("type", "unknown"),
            "answer": result["answer"], "em": em, "f1": f1, "sp_em": sp_em,
            "sp_f1": sp_metrics["sp_f1"], "joint_em": em * sp_em,
            "selected_sequence": selected, "selected_set": sorted(selected),
            "reader_context_hash": context_hash,
            "support_doc_recall": len(support_titles & selected_titles) / max(1, len(support_titles)),
            "all_support_docs": all_support,
            "available_gold_sentence_recall": sum(title in selected_titles for title, _ in gold_support) /
                                               max(1, len(gold_support)),
            "error_bucket": ("correct_with_full_support" if all_support else "correct_without_full_support")
                            if em else ("reader_or_reasoning_error" if all_support else "missing_gold_support"),
            "abstained": result["status"] == "ANSWER_ATTEMPTED_ABSTAINED",
            "observation_skipped": sum(event["event"] == "observation_skipped" for event in events),
            "latency_s": max(event["elapsed_s"] for event in events),
            "jev_forwards": result["usage"].get("decision", 0),
            "jev_examples": sum(request["examples"] for request in native_requests),
            "jev_input_tokens": sum(sum(request["lengths"]) for request in native_requests),
            "llm_input_tokens": sum(token_value(value, "prompt_tokens", "input_tokens") for value in llm_responses)
                                if token_complete else None,
            "llm_output_tokens": sum(token_value(value, "completion_tokens", "output_tokens") for value in llm_responses)
                                 if token_complete else None,
            "native_cost_instrumented": bool(native_requests), "usage": result["usage"]}
    def mean(key):
        values = [value[key] for value in output.values()]
        return statistics.mean(values) if all(value is not None for value in values) else None
    summary = {"n": len(output), "valid": True, "answer_attempt_rate": 1.0,
        **{key: mean(key) for key in ("em", "f1", "sp_em", "sp_f1", "joint_em", "support_doc_recall", "all_support_docs",
            "available_gold_sentence_recall", "abstained", "latency_s", "jev_forwards", "jev_examples",
            "jev_input_tokens", "llm_input_tokens", "llm_output_tokens")},
        "error_buckets": dict(Counter(value["error_bucket"] for value in output.values())),
        "native_cost_instrumented_all": all(value["native_cost_instrumented"] for value in output.values()),
        "observation_skipped_total": sum(value["observation_skipped"] for value in output.values())}
    return {"manifest": manifest, "summary": summary, "tasks": output}


class InputMap(dict):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--official-evaluator", type=Path, required=True)
    parser.add_argument("--read-k", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tasks, corpus = InputMap(unique(rows(args.tasks), "id")), InputMap(unique(rows(args.corpus), "id"))
    tasks.path, corpus.path = args.tasks, args.corpus
    gold = unique(json.loads(args.gold.read_text()), "_id")
    specification = importlib.util.spec_from_file_location("official_hotpot_report", args.official_evaluator)
    official = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(official)
    if len({str(run) for run in args.runs}) != len(args.runs):
        raise ValueError("Duplicate run paths")
    reports = {str(run): summarize(run, tasks, corpus, gold, official, args.read_k) for run in args.runs}
    values = list(reports.values())
    shared_keys = ("protocol", "tasks_sha256", "corpus_sha256", "plans_sha256", "llm_protocol",
                   "read_k", "limits", "source_sha256", "mechanism_source_sha256")
    for value in values[1:]:
        for key in shared_keys:
            if value["manifest"].get(key) != values[0]["manifest"].get(key):
                raise ValueError(f"Mixed protocols or inputs: {key}")
    comparisons = []
    paths = list(reports)
    for i, left in enumerate(paths):
        for right in paths[i+1:]:
            before, after = reports[left]["tasks"], reports[right]["tasks"]
            same_set = [key for key in tasks if before[key]["selected_set"] == after[key]["selected_set"]]
            if any(before[key]["reader_context_hash"] != after[key]["reader_context_hash"] for key in same_set):
                raise ValueError("Same selected set has a different reader context")
            comparisons.append({"baseline": left, "method": right,
                "delta_em": statistics.mean(after[key]["em"] - before[key]["em"] for key in tasks),
                "wrong_to_right": sum(before[key]["em"] == 0 and after[key]["em"] == 1 for key in tasks),
                "right_to_wrong": sum(before[key]["em"] == 1 and after[key]["em"] == 0 for key in tasks),
                "same_first_doc_rate": statistics.mean(before[key]["selected_sequence"][0] ==
                                                        after[key]["selected_sequence"][0] for key in tasks),
                "same_set_n": len(same_set),
                "same_set_answer_disagreement_n": sum(before[key]["answer"] != after[key]["answer"] for key in same_set),
                "same_set_em_disagreement_n": sum(before[key]["em"] != after[key]["em"] for key in same_set)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"official_evaluator_sha256": sha(args.official_evaluator), "runs": reports,
               "pair_descriptives": comparisons,
               "warning": "Coverage uses offline gold and does not prove answer sufficiency. "
                          "This report does not implement multiple-testing correction or paired inference."}
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: value["summary"] for key, value in reports.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
