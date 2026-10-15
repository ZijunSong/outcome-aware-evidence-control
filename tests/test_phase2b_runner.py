"""CPU checks for the phase2b checkpoint, third-read, and report chain."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from oec.phase2b import VerifierProtocolFailure, third_context, verify_once
from oec.runtime import ProtocolError, digest
from oec.phase1c import ShortAnswer


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = Path(os.environ.get(
    "OEC_TEST_OFFICIAL_EVALUATOR",
    "/data/ppnm/oec_project/external/hotpot/hotpot_evaluate_v1.py"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class Phase2bChainTests(unittest.TestCase):
    def test_truncation_is_not_rewritten_as_defer(self):
        class Backend:
            def decide(self, state, questions, critical=False):
                raise ProtocolError("Native decision context truncated")
        checkpoint = {"question": "q", "plan": {"goals": [
            {"id": "g1", "description": "advisor", "depends_on": []}]},
            "documents_by_id": {"a": {"title": "A", "sentences": ["A text"]}},
            "draft": {"runtime_valid": False, "answer": None}}
        with self.assertRaises(ProtocolError) as caught:
            verify_once(checkpoint, Backend())
        self.assertNotIsInstance(caught.exception, VerifierProtocolFailure)

    def test_protocol_failure_stays_distinct_from_defer(self):
        class Backend:
            def decide(self, state, questions, critical=False):
                raise ProtocolError("malformed choice")
        checkpoint = {"question": "q", "plan": {"goals": [
            {"id": "g1", "description": "advisor", "depends_on": []}]},
            "documents_by_id": {"a": {"title": "A", "sentences": ["A text"]}},
            "draft": {"runtime_valid": True, "answer": {"answer": "", "citations": [], "abstain": True}}}
        with self.assertRaises(VerifierProtocolFailure):
            verify_once(checkpoint, Backend())

    def test_cli_chain_keeps_one_prefix_and_one_extra_read(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            docs = {}
            texts = {"b": "birthplace", "a": "advisorfact", "c": "advisorfact"}
            for name in "abcdefghij":
                docs[name] = {"id": name, "title": name.upper(),
                              "sentences": [texts.get(name, "zzzzzzzzzzz")]}
            # Read order is b then a, so observation d1 is b while sorted reader d1 is a.
            selected, scope = ["b", "a"], list(docs)
            question = "Where was the advisor born?"
            goals = [{"id": "g1", "description": "advisor identity", "depends_on": []},
                     {"id": "g2", "description": "birthplace", "depends_on": ["g1"]},
                     {"id": "g3", "description": "award", "depends_on": []}]
            task = {"id": "t1", "question": question, "corpus_ids": scope}
            write_jsonl(root / "tasks.jsonl", [task])
            write_jsonl(root / "corpus.jsonl", list(docs.values()))
            write_json(root / "gold.json", [{"_id": "t1", "question": question, "answer": "Prague"}])
            write_json(root / "identity.json", {"served": {"model": "fixture"}})
            context_docs = [{"document_ref": f"d{index + 1}", "title": docs[key]["title"],
                             "sentences": [{"sentence_id": 0, "text": docs[key]["sentences"][0]}]}
                            for index, key in enumerate(sorted(selected))]
            context = {"question": question, "documents": context_docs}
            bundle = {"task_id": "t1", "context": context, "context_hash": digest(context),
                      "selected_doc_ids": sorted(selected), "selection_stage_valid": True}
            search = [{"doc_id": key, "title": docs[key]["title"],
                       "snippet": docs[key]["sentences"][0][:240], "retrieval_score": 1.0} for key in scope]
            state = {"question": question, "goals": goals, "known_observations": [
                {"text": "advisorfact", "related_goal_ids": ["g1"], "sentence_ids": [0]}],
                "unresolved_goal_ids": ["g1", "g2"], "answer_hypotheses": ["Prague"],
                "read_document_refs": ["d1", "d2"], "state_revision": 2}
            events = [
                {"event": "shared_plan_loaded", "plan": {"goals": goals}},
                {"event": "response", "kind": "search", "response": search},
                {"event": "phase1c_initial_state", "state": {**state, "known_observations": [],
                 "read_document_refs": [], "answer_hypotheses": []}},
                {"event": "phase1c_selected", "doc_id": "b"},
                {"event": "response", "kind": "read", "response": {"id": "b", "title": "B", "sentences": ["birthplace"]}},
                {"event": "phase1c_selected", "doc_id": "a"},
                {"event": "response", "kind": "read", "response": {"id": "a", "title": "A", "sentences": ["advisorfact"]}},
                {"event": "phase1c_state", "state": state}]
            result = {"task_id": "t1", "answer": "", "supporting_facts": [], "runtime_valid": False,
                      "verified": False, "status": "SELECTION_ONLY_NOT_EVALUATED", "repairs": 0,
                      "usage": {"read": 2}, "answer_attempted": False}
            selection = root / "selection"
            write_json(selection / "task_00000" / "result.json", result)
            write_json(selection / "task_00000" / "reader_context.json", bundle)
            write_jsonl(selection / "task_00000" / "events.jsonl", events)
            write_jsonl(selection / "predictions.jsonl", [result])
            manifest = {"protocol": "phase1e-selection-deferred-v1", "completed": True, "method": "utility_dynamic",
                        "variant": "native", "selection_family": "legacy-native", "read_k": 2,
                        "tasks_sha256": __import__("hashlib").sha256((root / "tasks.jsonl").read_bytes()).hexdigest(),
                        "corpus_sha256": __import__("hashlib").sha256((root / "corpus.jsonl").read_bytes()).hexdigest(),
                        "plans_sha256": "plan", "online_gold_access": False, "offline_oracle": False,
                        "task_ids": ["t1"], "limits": {}, "llm_protocol": {}, "source_sha256": "x",
                        "additional_source_sha256": {}}
            write_json(selection / "manifest.json", manifest)
            draft = root / "drafts"
            key = "draft-key"
            entry = {"key": key, "runtime_valid": True, "context": context, "context_hash": digest(context),
                     "answer": {"answer": "Paris", "citations": [], "abstain": False}, "error": None}
            write_json(draft / "reader_pool" / key / "entry.json", entry)
            write_json(draft / "arms" / "arm_00" / "manifest.json", {
                "selection_run": str(selection), "reader_ledger_sha256": "ledger", "reader_keys": {"t1": key}})
            write_json(draft / "report.json", {"protocol": "phase1e-shared-reader-evaluation-v1",
                      "arms": [{"arm": "only"}], "reader_ledger_sha256": "ledger"})
            checkpoints = root / "checkpoints"
            from scripts.freeze_phase2b_prefix import main as freeze_main
            import sys
            sys.argv = ["freeze", "--selection-run", str(selection), "--draft-reader-root", str(draft),
                        "--tasks", str(root / "tasks.jsonl"), "--corpus", str(root / "corpus.jsonl"),
                        "--out", str(checkpoints)]
            freeze_main()
            saved = json.loads((checkpoints / "t1.json").read_text())
            self.assertEqual(saved["observation_ref_to_doc_id"]["d1"], "b")
            self.assertEqual(saved["reader_ref_to_doc_id"]["d1"], "a")
            self.assertEqual(saved["selected_sequence"], ["b", "a"])
            probe = root / "probe"
            from scripts.run_phase2b_probe import main as probe_main
            sys.argv = ["probe", "--checkpoints", str(checkpoints), "--tasks", str(root / "tasks.jsonl"),
                        "--corpus", str(root / "corpus.jsonl"), "--identity", str(root / "identity.json"),
                        "--local-model-run", str(root), "--backend", "fixture",
                        "--policies", "continue_dynamic", "verify_broad", "verify_local", "target_only",
                        "--out", str(probe)]
            probe_main()
            chosen = {}
            for policy in ("continue_dynamic", "verify_broad", "verify_local", "target_only"):
                row = json.loads((probe / "policies" / policy / "t1" / "selection.json").read_text())
                chosen[policy] = row
                self.assertEqual(row["selected_doc_ids"][:2], ["a", "b"])
                self.assertEqual(len(row["selected_doc_ids"]), 3)
                self.assertNotIn(row["third_doc_id"], {"a", "b"})
                self.assertEqual(set(row["context"]), {"question", "documents"})
            self.assertNotEqual(chosen["verify_local"]["third_doc_id"], chosen["continue_dynamic"]["third_doc_id"])
            self.assertEqual(chosen["target_only"]["retained_observations"], chosen["continue_dynamic"]["retained_observations"])
            self.assertLess(chosen["verify_local"]["retained_observations"], chosen["verify_broad"]["retained_observations"])
            tampered = json.loads((probe / "policies" / "continue_dynamic" / "t1" / "selection.json").read_text())
            tampered["context"]["documents"][0]["sentences"][0]["text"] += " changed"
            write_json(probe / "policies" / "continue_dynamic" / "t1" / "selection.json", tampered)
            import scripts.replay_phase2b_reader as replay
            sys.argv = ["replay", "--probe-run", str(probe), "--tasks", str(root / "tasks.jsonl"),
                        "--corpus", str(root / "corpus.jsonl"), "--identity", str(root / "identity.json"),
                        "--out", str(root / "reader-tampered")]
            with self.assertRaises(ProtocolError):
                replay.main()
            write_json(probe / "policies" / "continue_dynamic" / "t1" / "selection.json", chosen["continue_dynamic"])
            original = replay.ContextReader.read
            replay.ContextReader.read = lambda self, context: ShortAnswer(answer="", citations=[], abstain=True)
            try:
                sys.argv = ["replay", "--probe-run", str(probe), "--tasks", str(root / "tasks.jsonl"),
                            "--corpus", str(root / "corpus.jsonl"), "--identity", str(root / "identity.json"),
                            "--out", str(root / "reader")]
                replay.main()
            finally:
                replay.ContextReader.read = original
            from scripts.report_phase2b import main as report_main
            sys.argv = ["report", "--study", str(root / "reader"), "--gold", str(root / "gold.json"),
                        "--official-evaluator", str(EVALUATOR), "--mode", "development",
                        "--probe-run", str(probe), "--out", str(root / "report.json")]
            report_main()
            decision = json.loads((root / "report.json").read_text())
            self.assertEqual(decision["decision"], "NO-GO")
            self.assertFalse((root / "confirm_800").exists())


if __name__ == "__main__":
    unittest.main()
