"""One shared verifier, then one extra document for every requested policy."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oec.phase1c import Selector
from oec.phase2b import (POLICIES, VerifierProtocolFailure, choose_document, policy_state,
                         tentative_draft, third_context, verify_once)
from oec.repair_probe import repair_plan
from oec.runtime import ProtocolError, Runtime, digest
from oec.types import Limits
from scripts.replay_phase1e_reader import indexed, lines, sha


def load_backend(kind, model_run, runtime):
    if kind == "searchjev":
        from oec.local_searchjev import LocalSearchJev
        return LocalSearchJev(runtime, model_run)
    if kind == "fixture":
        return FixtureBackend()
    raise ProtocolError("Unknown backend")


class FixtureBackend:
    """Deterministic CPU stand-in. It is not a model-quality claim."""

    def decide(self, state, questions, critical=False):
        if state.get("phase") == "phase2b_goal_verification":
            labels = list(next(iter(questions.values()))["criteria"])
            choice = "UNRESOLVED" if "advisor" in json.dumps(state.get("goals", [])) else "SOURCE_SUPPORTED"
            return {key: {"type": "choice", "choice": choice,
                          "probabilities": {label: 0.7 if label == choice else 0.1 for label in labels},
                          "confidence": 0.6} for key in questions}
        return {key: {"type": "noul", "noul": 0.2 + (0.5 if key.endswith("_gap") else 0.0),
                      "probabilities": {"true": 0.7, "false": 0.3}, "confidence": 0.6} for key in questions}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--local-model-run", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", required=True)
    parser.add_argument("--backend", default="searchjev")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or any(name not in POLICIES for name in args.policies):
        parser.error("Use a new output directory and known policies")
    if len(set(args.policies)) != len(args.policies):
        parser.error("Duplicate policies")
    identity = json.loads(args.identity.read_text())
    expected = identity.get("llm_served_model") or identity.get("served", {}).get("model")
    if args.backend == "searchjev" and expected and expected != os.environ.get("OEC_LLM_MODEL"):
        raise ProtocolError("Served model does not match the frozen identity")
    tasks, corpus = indexed(lines(args.tasks), "id"), indexed(lines(args.corpus), "id")
    ledger = json.loads((args.checkpoints / "ledger.json").read_text())
    if ledger["tasks_sha256"] != sha(args.tasks) or ledger["corpus_sha256"] != sha(args.corpus):
        raise ProtocolError("Checkpoint inputs do not match the requested tasks or corpus")
    checkpoints = []
    for row in ledger["checkpoints"]:
        path = args.checkpoints / f"{row['task_id']}.json"
        if not path.is_file():
            raise ProtocolError("Missing checkpoint; refusing to rerun a lost question")
        checkpoint = json.loads(path.read_text())
        body = {key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"}
        if digest(body) != checkpoint["checkpoint_sha256"] or digest(body) != row["checkpoint_sha256"]:
            raise ProtocolError("Checkpoint hash mismatch")
        if checkpoint["task_id"] not in tasks:
            raise ProtocolError("Checkpoint task is outside the assigned set")
        checkpoints.append(checkpoint)
    if [row["task_id"] for row in checkpoints] != sorted(tasks):
        raise ProtocolError("Checkpoint task set differs from the assigned tasks")
    args.out.mkdir(parents=True)
    backend = load_backend(args.backend, args.local_model_run,
                           Runtime(args.out / "model_load", Limits(
                               search_calls=1, read_calls=8, llm_calls=8, decision_calls=20,
                               repair_search_reserve=0, repair_read_reserve=0,
                               terminal_llm_reserve=2, terminal_decision_reserve=4)))
    limits = Limits(search_calls=1, read_calls=8, llm_calls=8, decision_calls=400,
                    repair_search_reserve=0, repair_read_reserve=0,
                    terminal_llm_reserve=2, terminal_decision_reserve=4)
    verifier_rows, selections = [], {name: [] for name in args.policies}
    for checkpoint in checkpoints:
        runtime = Runtime(args.out / "verifier" / checkpoint["task_id"], limits)
        backend.runtime = runtime
        wrapped = _RecordingBackend(backend, runtime)
        verdicts, verifier_valid, failure, omitted = None, True, None, []
        try:
            verdicts, omitted = verify_once(checkpoint, wrapped)
            plan = repair_plan(checkpoint["plan"]["goals"], verdicts, checkpoint["documents_by_id"])
        except VerifierProtocolFailure as exc:
            verifier_valid, failure = False, {"type": "verifier_protocol_failure", "message": str(exc)}
            plan = {"target": None, "affected": [], "failed": [], "reason": "verifier_protocol_failure"}
        labels = {row["goal_id"]: row["label"] for row in verdicts or []}
        if verdicts and all(label == "DEFER" for label in labels.values()):
            plan = {"target": None, "affected": [], "failed": [], "reason": "all_defer"}
        verifier_rows.append({"task_id": checkpoint["task_id"], "draft_runtime_valid": checkpoint["draft"]["runtime_valid"],
                              "verifier_valid": verifier_valid, "failure": failure, "omitted_suffixes": omitted,
                              "verdicts": verdicts,
                              "plan": plan, "tentative_draft": tentative_draft(checkpoint["draft"])})
        selector = Selector(wrapped, runtime, batch_size=8)
        continue_proxy, _, _ = policy_state(checkpoint, "continue_dynamic", {"target": None, "failed": []})
        continue_doc, continue_scores = choose_document(selector, continue_proxy, checkpoint["remaining_candidates"])
        for policy in args.policies:
            fallback = plan["target"] is None or policy == "continue_dynamic"
            if policy == "continue_dynamic" or plan["target"] is None:
                proxy, deactivated_flag, deactivated = continue_proxy, False, []
                doc_id, scores = continue_doc, continue_scores
                fallback = policy != "continue_dynamic"
            else:
                proxy, deactivated_flag, deactivated = policy_state(checkpoint, policy, plan)
                doc_id, scores = choose_document(selector, proxy, checkpoint["remaining_candidates"])
                fallback = False
            context, doc_ids = third_context(checkpoint, corpus, doc_id)
            record = {"task_id": checkpoint["task_id"], "policy": policy, "third_doc_id": doc_id,
                      "selected_doc_ids": doc_ids, "scores": scores, "fallback_to_continue": fallback,
                      "target": None if fallback and policy != "continue_dynamic" else plan["target"],
                      "deactivated_observations": len(deactivated), "deactivated": deactivated_flag,
                      "retained_observations": len(proxy.facts),
                      "unattributed_observations": sum(not fact.get("related_goal_ids") for fact in proxy.facts),
                      "logical_reads": 3, "context": context}
            selections[policy].append(record)
            directory = args.out / "policies" / policy / checkpoint["task_id"]
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "selection.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "verifier.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in verifier_rows), encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    manifest = {"protocol": "phase2b-selection-deferred-v1", "policies": list(args.policies),
                "tasks_sha256": sha(args.tasks), "corpus_sha256": sha(args.corpus),
                "prefix_ledger_sha256": ledger["ledger_sha256"],
                "verifier_ledger_sha256": digest(verifier_rows),
                "identity_sha256": sha(args.identity), "backend": args.backend,
                "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                  for name in ("oec/phase2b.py", "oec/repair_probe.py", "scripts/run_phase2b_probe.py")},
                "final_answers_deferred": True, "completed": True}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n": len(checkpoints), "policies": args.policies,
                      "verifier_failures": sum(not row["verifier_valid"] for row in verifier_rows)}, indent=2))


class _RecordingBackend:
    def __init__(self, backend, runtime):
        self.backend, self.runtime = backend, runtime

    def decide(self, state, questions, critical=False):
        self.runtime.event("phase2b_decision", phase=state.get("phase"), n=len(questions))
        return self.backend.decide(state, questions, critical)


if __name__ == "__main__":
    main()
