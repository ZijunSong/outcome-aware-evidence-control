import json
import tempfile
import unittest
from pathlib import Path
from oec.controller import Controller
from oec.demo import DemoJev, DemoLLM, DemoRetriever
from oec.graph import (add_extraction, apply_support, failed_roots, install_plan,
                       invalidate_goal, resolved)
from oec.jev import Decisions, validate_answers
from oec.retrieval import split_sentences
from oec.runtime import BudgetExceeded, ProtocolError, Runtime
from oec.types import (Candidate, Claim, Document, Extraction, Goal, Limits, Plan,
                       ProposedClaim, Settings, State, Status, Task)


class CoreTests(unittest.TestCase):
    def runtime(self, root, settings=None):
        return Runtime(Path(root) / "run", (settings or Settings()).limits)

    def claim(self, claim_id, goal_id, value, parents=()):
        return Claim(id=claim_id, goal_id=goal_id, text=value, value=value, evidence_ids=[],
                     depends_on=list(parents), mode="DIRECT", status=Status.SUPPORTED)

    def test_plan_rejects_cycles_and_unknown_dependencies(self):
        state = State(task_id="t", question="q")
        for plan in (Plan(goals=[Goal(id="g1", description="x", depends_on=["g1"])]),
                     Plan(goals=[Goal(id="g1", description="x", depends_on=["missing"])])):
            with self.assertRaises(ProtocolError):
                install_plan(state, plan)

    def test_provenance_rejects_fabricated_sentence(self):
        state = State(task_id="t", question="q")
        install_plan(state, Plan(goals=[Goal(id="g1", description="advisor")]))
        proposal = ProposedClaim(goal_id="g1", text="advisor B", value="B", sentence_ids=[99])
        with self.assertRaises(ProtocolError):
            add_extraction(state, Document(id="d", title="record", sentences=["B"]), Extraction(claims=[proposal]))

    def test_conflicting_values_block_descendant_bindings(self):
        state = State(task_id="t", question="q", goals={"g1": Goal(id="g1", description="advisor"),
                      "g2": Goal(id="g2", description="birthplace", depends_on=["g1"])})
        state.claims = {"c1": self.claim("c1", "g1", "B"), "c2": self.claim("c2", "g1", "C"),
                        "c3": self.claim("c3", "g2", "Prague", ["c1"])}
        apply_support(state, ["c1", "c2"], {"c1": .99, "c2": .99}, .85)
        self.assertFalse(resolved(state, "g1"))
        self.assertNotEqual(state.claims["c3"].status, Status.SUPPORTED)

    def test_local_repair_preserves_independent_facts(self):
        state = State(task_id="t", question="q")
        state.claims = {"c1": self.claim("c1", "g1", "B"),
                        "c2": self.claim("c2", "g2", "Prague", ["c1"]),
                        "c3": self.claim("c3", "g3", "B was born in Prague")}
        self.assertEqual(failed_roots(state, {"c1", "c2"}), ["c1"])
        self.assertEqual(invalidate_goal(state, "g1"), ["c1", "c2"])
        self.assertEqual(state.claims["c3"].status, Status.SUPPORTED)

    def test_repair_reserve_is_not_available_to_normal_search(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = self.runtime(root)
            for _ in range(4):
                runtime.invoke("search", {}, lambda: [])
            with self.assertRaises(BudgetExceeded):
                runtime.invoke("search", {}, lambda: [])
            runtime.invoke("search", {}, lambda: [], critical=True)
            runtime.invoke("search", {}, lambda: [], critical=True)
            self.assertEqual(runtime.used["search"], 6)
            with self.assertRaises(BudgetExceeded):
                runtime.invoke("search", {}, lambda: [], critical=True)

    def test_typed_api_rejects_illegal_probability_map(self):
        payload = {"questions": {"priority": {"type": "choice", "criteria": {"g1": "x", "g2": "y"}}}}
        with self.assertRaises(ProtocolError):
            validate_answers(payload, {"answers": {"priority": {"type": "choice", "choice": "g1",
                             "probabilities": {"g1": .9, "illegal": .1}, "confidence": .8}}})

    def test_utility_changes_when_target_gap_changes(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = self.runtime(root)
            settings = Settings()
            decisions = Decisions(DemoJev(runtime), runtime, settings)
            state = State(task_id="t", question="q")
            candidates = [Candidate(doc_id="d1", title="advisor", snippet="advisor record", retrieval_score=1),
                          Candidate(doc_id="d2", title="birth", snippet="birth city", retrieval_score=1)]
            first = decisions.utility(state, candidates, "g1")
            second = decisions.utility(state, candidates, "g2")
            self.assertGreater(first["d1"], first["d2"])
            self.assertGreater(second["d2"], second["d1"])

    def test_full_demo_repairs_once_and_preserves_raw_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            runtime, settings = self.runtime(root), Settings()
            controller = Controller(DemoLLM(runtime), Decisions(DemoJev(runtime), runtime, settings),
                                    DemoRetriever(), runtime, settings)
            result = controller.run(Task(id="demo", question="A's advisor birthplace?"))
            self.assertEqual((result.answer, result.repairs, result.verified), ("Vienna", 1, True))
            self.assertIn("d1:s0", controller.state.evidence)
            self.assertIn("d2:s0", controller.state.evidence)
            self.assertEqual(controller.state.claims["c1"].status, Status.INVALIDATED)
            self.assertEqual(result.usage["search"], 2)

    def test_wikipedia_sentence_ids_are_stable(self):
        text = "First sentence. Second sentence! Third?"
        self.assertEqual(split_sentences(text), ["First sentence.", "Second sentence!", "Third?"])
        self.assertEqual(split_sentences("  First sentence.   Second sentence! Third? "), split_sentences(text))

    def test_baseline_keeps_failed_verification_in_prediction(self):
        with tempfile.TemporaryDirectory() as root:
            settings = Settings(method="utility_dynamic")
            runtime = self.runtime(root, settings)
            result = Controller(DemoLLM(runtime), Decisions(DemoJev(runtime), runtime, settings),
                                DemoRetriever(), runtime, settings).run(Task(id="demo", question="q"))
            self.assertEqual(result.answer, "Prague")
            self.assertFalse(result.verified)
            self.assertTrue(result.runtime_valid)


if __name__ == "__main__":
    unittest.main()
