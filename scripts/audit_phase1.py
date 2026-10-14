"""Read old/new local traces without rerunning any model or writing to those runs."""
import argparse
import json
from collections import Counter
from pathlib import Path


def role_of(request):
    if "role" in request:
        return request["role"]
    try:
        user = json.loads(request["messages"][-1]["content"])
        title = user["schema"]["title"]
        return {"Plan": "plan", "Query": "query", "Extraction": "extract", "Draft": "answer",
                "ProbePlan": "phase1_plan", "Progress": "phase1_observe", "ShortAnswer": "phase1_answer"}.get(title, title)
    except (KeyError, TypeError, ValueError, IndexError):
        return "unknown"


def decode_content(response):
    if "choices" not in response:  # Demo fixture uses direct schema objects.
        return response, None, False
    choice = response["choices"][0]
    content = choice.get("message", {}).get("content")
    reasoning = bool(choice.get("message", {}).get("reasoning_content") or choice.get("message", {}).get("reasoning"))
    if not isinstance(content, str):
        return None, choice.get("finish_reason"), reasoning
    text = content.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[len("```json\n"):-len("\n```")]
    try:
        return json.loads(text), choice.get("finish_reason"), reasoning
    except ValueError:
        return None, choice.get("finish_reason"), reasoning


def audit_task(directory):
    if not (directory / "result.json").is_file() or not (directory / "events.jsonl").is_file():
        return {"directory": str(directory), "runtime_valid": False, "status": "MISSING_OUTPUT"}
    result = json.loads((directory / "result.json").read_text())
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines() if line.strip()]
    request_roles, rejects, supported_values = Counter(), Counter(), set()
    pending, pool, read_docs, full_quotes = {}, set(), set(), {}
    lengths, raw_claims, raw_incomplete, reasoning_messages = 0, 0, 0, 0
    explicit_thinking, support_probabilities = set(), []
    manual_quote_cuts = 0
    limits = next((event["limits"] for event in events if event["event"] == "runtime_start"), {})
    for event in events:
        if event["event"] == "request":
            kind, request = event["kind"], event["request"]
            pending[kind] = request
            if kind == "llm":
                request_roles[role_of(request)] += 1
                if "chat_template_kwargs" in request:
                    explicit_thinking.add(str(request["chat_template_kwargs"].get("enable_thinking")))
                else:
                    explicit_thinking.add("not_explicit_in_request")
        elif event["event"] == "response":
            kind, response = event["kind"], event["response"]
            request = pending.pop(kind, {})
            if kind == "search":
                pool.update(item["doc_id"] for item in response)
            elif kind == "read":
                read_docs.add(response["id"])
                full_quotes.update({f"{response['id']}:s{sid}": sentence
                                    for sid, sentence in enumerate(response["sentences"])})
            elif kind == "llm":
                parsed, finish, reasoning = decode_content(response)
                lengths += int(finish == "length")
                reasoning_messages += int(reasoning)
                if role_of(request) == "extract" and isinstance(parsed, dict):
                    rows = parsed.get("claims", [])
                    if isinstance(rows, list):
                        raw_claims += len(rows)
                        raw_incomplete += sum(not isinstance(row, dict) or not row.get("text")
                            or not row.get("value") or not row.get("sentence_ids") for row in rows)
        elif event["event"] == "reject_claim":
            rejects[event["reason"]] += 1
        elif event["event"] == "state":
            state = event["state"]
            supported_values.update((claim["goal_id"], claim["value"])
                                    for claim in state.get("claims", {}).values() if claim["status"] == "SUPPORTED")
        elif event["event"] == "local_typed_request":
            state = event["request"]["state"]
            pending["typed_phase"] = state.get("phase")
            if state.get("phase") in ("extract", "final"):
                for row in state.get("claim_rows", []):
                    for evidence in row.get("evidence", []):
                        original = full_quotes.get(evidence["id"])
                        if original is not None and evidence["quote"] != original:
                            manual_quote_cuts += 1
        elif event["event"] == "local_typed_response" and pending.get("typed_phase") == "extract":
            support_probabilities.extend(value["probabilities"]["true"] for value in event["raw_results"].values()
                                         if "true" in value.get("probabilities", {}))
    answer_calls = request_roles["answer"] + request_roles["phase1_answer"]
    answer_attempted = bool(answer_calls or result.get("answer_attempted") or
                            any(row["event"] == "answer_attempt" for row in events))
    usage = result.get("usage", {})
    return {"directory": str(directory), "task_id": result.get("task_id"), "status": result["status"],
            "runtime_valid": result["runtime_valid"], "answer_empty": not bool(result["answer"].strip()),
            "answer_attempted": answer_attempted, "answer_request_count": answer_calls,
            "request_roles": dict(request_roles), "read_documents": len(read_docs),
            "retrieved_unread_documents": len(pool - read_docs),
            "search_budget_left": limits.get("search_calls", 0)-usage.get("search", 0),
            "read_budget_left": limits.get("read_calls", 0)-usage.get("read", 0),
            "retrieved_doc_ids": sorted(pool),
            "raw_extraction_claims": raw_claims, "raw_incomplete_claims": raw_incomplete,
            "reject_claim_reasons": dict(rejects), "distinct_supported_goal_values": len(supported_values),
            "truncated_llm_responses": lengths, "reasoning_messages_observed": reasoning_messages,
            "client_thinking_flags": sorted(explicit_thinking), "manual_verification_quote_cuts": manual_quote_cuts,
            "extract_support_probabilities": support_probabilities}


def audit_run(run_directory, task_scopes=None):
    rows = [audit_task(path) for path in sorted(run_directory.glob("task_*")) if path.is_dir()]
    manifest_file = run_directory / "manifest.json"
    expected_ids = None
    manifest = json.loads(manifest_file.read_text()) if manifest_file.is_file() else {}
    expected_ids = manifest.get("task_ids")
    if expected_ids is None and task_scopes is not None:
        expected_ids = list(task_scopes)
    valid_rows = [row for row in rows if "task_id" in row]
    if task_scopes is not None:
        for row in valid_rows:
            scope = task_scopes.get(row["task_id"])
            if scope is not None:
                row["out_of_scope_retrieved"] = sorted(set(row["retrieved_doc_ids"]) - set(scope))
    actual_ids = {row["task_id"] for row in valid_rows}
    missing = sorted(set(expected_ids or []) - actual_ids)
    extra = sorted(actual_ids - set(expected_ids)) if expected_ids is not None else []
    statuses = Counter(row["status"] for row in rows)
    attempted = sum(row.get("answer_attempted", False) for row in rows)
    not_answered_repeat = [row for row in rows if row["status"] == "REPEATED_QUERY_UNVERIFIED"
                           and not row.get("answer_attempted", False)]
    retriever_file = run_directory / "retriever.json"
    retriever_kind = (json.loads(retriever_file.read_text()).get("kind") if retriever_file.is_file()
                      else manifest.get("retriever_kind", "not_recorded"))
    return {"run": str(run_directory), "retriever_kind": retriever_kind,
            "verdict": ("OUTPUTS_MISSING_OR_PROTOCOL_INVALID" if not rows or missing or extra
                        or len(actual_ids) != len(valid_rows) or any(not row["runtime_valid"] for row in rows)
                        else "OBSERVED_OUTPUTS_VALID_CHECK_ANSWER_ATTEMPTS"),
            "n_output_rows": len(rows), "statuses": dict(statuses),
            "invalid_runs": sum(not row["runtime_valid"] for row in rows), "missing_task_ids": missing,
            "unexpected_task_ids": extra,
            "answer_attempt_rate": attempted / len(rows) if rows else None,
            "answer_nonempty_rate": sum(not row.get("answer_empty", True) for row in rows)/len(rows) if rows else None,
            "repeat_query_without_answer_attempt": len(not_answered_repeat),
            "repeat_empty_with_unread_candidates": sum(row.get("retrieved_unread_documents", 0)>0 for row in not_answered_repeat),
            "repeat_empty_with_read_budget": sum(row.get("read_budget_left", 0)>0 for row in not_answered_repeat),
            "schema_incomplete_claims_before_sanitization": sum(row.get("raw_incomplete_claims", 0) for row in rows),
            "total_dependency_rejections": sum(row.get("reject_claim_reasons", {}).get("dependent slot lacks its parent binding", 0)
                                               + row.get("reject_claim_reasons", {}).get("depends on an unsupported claim", 0) for row in rows),
            "total_manual_quote_cuts": sum(row.get("manual_verification_quote_cuts", 0) for row in rows),
            "questions_with_out_of_scope_candidates": sum(bool(row.get("out_of_scope_retrieved")) for row in rows),
            "tasks": rows,
            "note": "These are trace diagnostics, not official answer correctness. Missing original manifests "
                    "prevent this script from independently detecting every missing expected question."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, help="Optional original task file for public-scope checks")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scopes = None if args.tasks is None else {row["id"]: row.get("corpus_ids")
        for row in (json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip())}
    reports = [audit_run(directory, scopes) for directory in args.runs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    for report in reports:
        print(json.dumps({key: value for key, value in report.items() if key not in ("tasks", "note")}, ensure_ascii=False))
