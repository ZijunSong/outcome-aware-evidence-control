"""CPU checks for shared first-hop, frozen pools, and confirmation gates."""
import json
import tempfile
import unittest
from pathlib import Path
from oec.frozen_pool import load_manifest
from oec.phase1c import Memory, ProbeGoal, ProbePlan
from oec.phase4_endpoints import paired_points
from oec.phase4_shared import pick, reader_bundle
from oec.phase4_single import SingleSignalSelector
from oec.runtime import ProtocolError, Runtime
from oec.types import Candidate, Limits, Task
from scripts.report_phase4 import summarize


class Phase4RunnerTests(unittest.TestCase):
    def test_pick_is_score_then_stable_id(self):
        self.assertEqual(pick({"b": 0.2, "a": 0.2, "c": 0.1}), "a")

    def test_reader_bundle_sorts_by_stable_id_and_contains_only_sources(self):
        class Document:
            def __init__(self, key):
                self.id, self.title, self.sentences = key, key.upper(), [key]
        task = Task(id="t", question="q", corpus_ids=["b", "a"])
        bundle = reader_bundle(task, [Document("b"), Document("a")])
        self.assertEqual(bundle["selected_doc_ids"], ["a", "b"])
        self.assertEqual(bundle["context"]["documents"][0]["document_ref"], "d1")
        self.assertEqual(set(bundle["context"]), {"question", "documents"})

    def test_unchanged_state_makes_both_branches_choose_the_same_second_document(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            limits = Limits(search_calls=1, read_calls=4, llm_calls=8, decision_calls=20,
                            repair_search_reserve=0, repair_read_reserve=0,
                            terminal_llm_reserve=2, terminal_decision_reserve=4)
            class Backend:
                def decide(self, state, questions, critical=False):
                    return {key: {"type": "noul", "noul": 0.4 if "d9" in str(questions[key]) else 0.8}
                            for key in questions}
            runtime = Runtime(root / "rt", limits)
            plan = ProbePlan(goals=[ProbeGoal(id="g1", description="advisor", depends_on=[])])
            memory = Memory("q", plan, unresolved=["g1"])
            candidates = [Candidate(doc_id=f"d{i}", title=f"D{i}", snippet="x", retrieval_score=1)
                          for i in range(10)]
            current = SingleSignalSelector(Backend(), runtime, "current")
            scores = current.score(memory, candidates)
            first = pick(scores)
            remaining = [row for row in candidates if row.doc_id != first]
            initial = SingleSignalSelector(Backend(), Runtime(root / "init", limits), "initial")
            initial.restore_initial(memory, candidates, scores)
            again = SingleSignalSelector(Backend(), Runtime(root / "cur", limits), "current")
            again.restore_initial(memory, candidates, scores)
            self.assertEqual(pick(initial.score(memory, remaining)), pick(again.score(memory, remaining)))

    def test_public_pool_hashes_bind_files_and_reject_gold(self):
        from scripts.freeze_public_candidate_pools import main as freeze
        import sys
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "tasks.jsonl").write_text(json.dumps({
                "id": "t", "question": "q", "corpus_ids": ["a", "b"]}) + "\n")
            (root / "corpus.jsonl").write_text("".join(
                json.dumps({"id": key, "title": key, "sentences": [key]}) + "\n" for key in "ab"))
            sys.argv = ["freeze", "--tasks", str(root / "tasks.jsonl"), "--corpus", str(root / "corpus.jsonl"),
                        "--out", str(root / "pool")]
            freeze()
            manifest = load_manifest(root / "pool")
            self.assertFalse(manifest["gold_used"])
            self.assertEqual(len(manifest["pool_files"]), 1)

    def test_confirmation_rejects_exposed_overlap(self):
        from scripts.make_phase4_unseen_split import build
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            context = [[f"D{i}", [f"s{i}"]] for i in range(10)]
            source = [{"_id": "t0", "question": "q0", "context": context, "answer": "a",
                       "supporting_facts": [["D0", 0]]}]
            (root / "source.json").write_text(json.dumps(source))
            (root / "seen.jsonl").write_text(json.dumps({"id": "t0"}) + "\n")
            from scripts.make_phase4_unseen_split import sha
            (root / "registry.json").write_text(json.dumps({
                "history_complete_declared": True,
                "files": [{"path": "seen.jsonl", "sha256": sha(root / "seen.jsonl")}]}))
            with self.assertRaises(ValueError):
                build(root / "source.json", root / "registry.json", root / "out", 1, 17)

    def test_primary_requires_positive_delta_and_significance(self):
        result = summarize([(0, 0)] * 10)
        self.assertEqual(result["delta"], 0)
        self.assertGreaterEqual(result["mcnemar_exact_p"], 0.05)


if __name__ == "__main__":
    unittest.main()
