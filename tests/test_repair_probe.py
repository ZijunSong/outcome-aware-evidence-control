import copy
import importlib.util
import unittest
from pathlib import Path
from oec.repair_probe import repair_plan, local_view, verify_goals
from oec.runtime import ProtocolError

GOALS = [
    {"id": "g1", "description": "advisor", "depends_on": []},
    {"id": "g2", "description": "birthplace", "depends_on": ["g1"]},
    {"id": "g3", "description": "award", "depends_on": []}]
DOCS = {"doc-a": {"sentences": ["A's advisor is C.", "B was born in Prague."]}}


def verdicts(first="CONTRADICTED", second="SOURCE_SUPPORTED", third="SOURCE_SUPPORTED"):
    return [{"goal_id": key, "label": label, "checked_source_refs":
             [{"doc_id": "doc-a", "sentence_id": 0}] if label in
             {"CONTRADICTED", "SOURCE_SUPPORTED"} else []}
            for key, label in zip(("g1", "g2", "g3"), (first, second, third))]


class RepairProbeTests(unittest.TestCase):
    def test_native_verifier_is_bounded_and_uses_read_sources_without_gold(self):
        class Backend:
            def decide(self, state, questions, critical=False):
                self.state, self.questions = state, questions
                labels = list(next(iter(questions.values()))["criteria"])
                return {key: {"type": "choice", "choice": "UNRESOLVED",
                    "probabilities": {label: .7 if label == "UNRESOLVED" else .1 for label in labels},
                    "confidence": .6} for key in questions}
        backend = Backend()
        docs = {"doc-a": {"title": "A", "sentences": ["A text"]}}
        values = verify_goals("q", GOALS, docs, {"answer": "Prague", "runtime_valid": True}, backend)
        self.assertEqual(len(values), 3)
        self.assertEqual(set(backend.questions), {"g1", "g2", "g3"})
        self.assertEqual(values[0]["checked_source_refs"], [{"doc_id": "doc-a", "sentence_id": 0}])
        self.assertEqual(set(backend.state), {"phase", "question", "goals", "read_documents", "tentative_draft", "instruction"})

    def test_wrong_parent_affects_downstream_but_not_independent_goal(self):
        result = repair_plan(GOALS, verdicts(), DOCS)
        self.assertEqual(result["target"], "g1")
        self.assertEqual(result["affected"], ["g1", "g2"])

    def test_earliest_failure_precedes_failed_child(self):
        result = repair_plan(GOALS, verdicts("UNRESOLVED", "UNRESOLVED"), DOCS)
        self.assertEqual(result["target"], "g1")

    def test_resolved_parent_allows_targeting_child(self):
        result = repair_plan(GOALS, verdicts("SOURCE_SUPPORTED", "UNRESOLVED"), DOCS)
        self.assertEqual(result["target"], "g2")

    def test_deferred_parent_does_not_establish_child_binding(self):
        result = repair_plan(GOALS, verdicts("DEFER", "UNRESOLVED"), DOCS)
        self.assertIsNone(result["target"])

    def test_no_failure_uses_original_policy(self):
        self.assertIsNone(repair_plan(GOALS, verdicts("SOURCE_SUPPORTED"), DOCS)["target"])

    def test_illegal_or_unread_sources_are_rejected(self):
        for change in ({"doc_id": "unread", "sentence_id": 0},
                       {"doc_id": "doc-a", "sentence_id": True},
                       {"doc_id": "doc-a", "sentence_id": 99}):
            rows = verdicts()
            rows[0]["checked_source_refs"] = [change]
            with self.assertRaises(ProtocolError):
                repair_plan(GOALS, rows, DOCS)

    def test_source_supported_requires_legal_source_but_does_not_prove_entailment(self):
        rows = verdicts()
        rows[1]["checked_source_refs"] = []
        with self.assertRaises(ProtocolError):
            repair_plan(GOALS, rows, DOCS)

    def test_cycle_unknown_and_duplicate_goals_are_rejected(self):
        for goals in (GOALS + [GOALS[0]],
                      [{"id": "g1", "depends_on": ["g1"]}],
                      [{"id": "g1", "depends_on": ["other"]}]):
            with self.assertRaises(ProtocolError):
                repair_plan(goals, verdicts(), DOCS)

    def test_local_view_is_immutable_and_retains_independent_observations(self):
        memory = {"goals": GOALS, "known_observations": [
            {"text": "A's advisor is B", "related_goal_ids": ["g1"]},
            {"text": "B birthplace", "related_goal_ids": ["g2"]},
            {"text": "A won award", "related_goal_ids": ["g3"]}],
            "unresolved_goal_ids": [], "answer_hypotheses": ["Prague"], "read_document_refs": ["d1"]}
        frozen = copy.deepcopy(memory)
        view = local_view(memory, repair_plan(GOALS, verdicts(), DOCS))
        self.assertEqual(memory, frozen)
        self.assertEqual([fact["text"] for fact in view["known_observations"]], ["A won award"])
        self.assertEqual(view["unresolved_goal_ids"], ["g1", "g2"])
        self.assertEqual(view["answer_hypotheses"], [])
        self.assertEqual(view["read_document_refs"], ["d1"])


class HeadroomTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1]/"scripts/audit_phase2b_headroom.py"
        spec = importlib.util.spec_from_file_location("audit", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.audit = module.audit
        self.corpus = [{"id": f"d{i}", "title": f"D{i}", "sentences": [f"sentence {i}"]} for i in range(10)]
        self.tasks = [{"id": "t", "question": "q", "corpus_ids": [doc["id"] for doc in self.corpus]}]
        self.gold = [{"_id": "t", "question": "q", "answer": "C", "context":
                      [[doc["title"], doc["sentences"]] for doc in self.corpus],
                      "supporting_facts": [["D0", 0], ["D2", 0]]}]
        self.pred = [{"task_id": "t", "runtime_valid": True, "answer": "B", "selected_doc_ids": ["d0", "d1"]}]

    def run_audit(self):
        return self.audit(self.tasks, self.pred, self.corpus, self.gold, lambda a, b: a == b)

    def test_missing_support_is_proxy_and_denominator_is_all_tasks(self):
        result = self.run_audit()
        self.assertEqual(result["n"], 1)
        self.assertEqual(result["counts"]["one_extra_doc_support_proxy"], 1)
        self.assertIn("Forbidden as online input", result["warning"])

    def test_reader_failure_is_not_labeled_as_evidence_repair_opportunity(self):
        self.pred[0]["runtime_valid"] = False
        self.pred[0]["failure_stage"] = "bounded_reader_output"
        result = self.run_audit()
        self.assertEqual(result["counts"]["bounded_reader_failure"], 1)
        self.assertEqual(result["counts"]["one_extra_doc_support_proxy"], 0)

    def test_all_support_read_is_separate_from_missing_support(self):
        self.pred[0]["selected_doc_ids"] = ["d0", "d2"]
        self.assertEqual(self.run_audit()["counts"]["wrong_with_all_support_titles_read"], 1)

    def test_duplicate_prediction_missing_question_and_changed_source_rejected(self):
        self.pred.append(copy.deepcopy(self.pred[0]))
        with self.assertRaises(ValueError):
            self.run_audit()
        self.pred.pop()
        self.gold[0]["question"] = "another"
        with self.assertRaises(ValueError):
            self.run_audit()
        self.gold[0]["question"] = "q"
        self.corpus[0]["sentences"] = ["changed"]
        with self.assertRaises(ValueError):
            self.run_audit()


if __name__ == "__main__":
    unittest.main()
