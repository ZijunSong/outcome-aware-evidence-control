import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from oec.phase1c import (Citation, Memory, ObservedFact, ProbeGoal, ProbePlan, Progress,
    RoleJSONClient, SelectionOnly, Selector, ShortAnswer, fixed_candidates, validate_answer, validate_progress)
from oec.runtime import ProtocolError, Runtime
from oec.types import Candidate, Document, Limits, Task


PLAN = ProbePlan(goals=[ProbeGoal(id="g1", description="Confirm identity", depends_on=[]),
                       ProbeGoal(id="g2", description="Find birthplace", depends_on=["g1"])])


class FixtureRetriever:
    def __init__(self):
        self.documents = {
            "a": Document(id="a", title="Birth", sentences=["B was born in Prague."]),
            "b": Document(id="b", title="Advisor", sentences=["A's doctoral advisor was B."]),
            "c": Document(id="c", title="Other", sentences=["Other additional evidence."])}
        self.scope = None
    def set_scope(self, ids):
        self.scope = ids
    def search(self, question, n):
        # Only one positive BM25 hit. The fixed-scope builder must keep the zero-score docs.
        return [Candidate(doc_id="a", title="Birth", snippet="B born Prague", retrieval_score=1)]
    def fetch(self, key):
        return self.documents[key]


class FixtureLLM:
    def __init__(self, runtime, abstain=False):
        self.runtime, self.abstain = runtime, abstain
        self.calls, self.last_answer_context = [], None
    def ask(self, role, context, schema, critical=False, validator=None):
        self.calls.append(role)
        if role == "phase1_observe":
            raw = Progress(facts=[ObservedFact(text=context["document"]["sentences"][0]["text"],
                sentence_ids=[0], related_goal_ids=["g2"])],
                unresolved_goal_ids=["g1", "g2"], answer_hypotheses=[]).model_dump()
        elif role == "phase1_answer":
            self.last_answer_context = context
            raw = (ShortAnswer(answer="", citations=[], abstain=True) if self.abstain else
                   ShortAnswer(answer="Prague", citations=[Citation(document_ref="d1", sentence_id=0)], abstain=False)).model_dump()
        else:
            raise AssertionError(role)
        response = self.runtime.invoke("llm", {"role": role, "context": context}, lambda: raw, critical, "fixture")
        result = schema.model_validate(response)
        if validator:
            validator(result)
        return result


class FixtureBackend:
    def __init__(self, runtime):
        self.runtime, self.requests = runtime, []
    def decide(self, state, questions, critical=False):
        self.requests.append(state)
        values = {"a": .95, "b": .7, "c": .4} if not state.get("known_observations") else {"a": .95, "b": .01, "c": .99}
        raw = {key: {"type": "noul", "noul": values[question["instructions"]["candidate_doc_id"]]}
               for key, question in questions.items()}
        self.runtime.invoke("decision", {"state": state, "questions": questions}, lambda: raw, critical, "fixture")
        return raw


class Phase1Tests(unittest.TestCase):
    def limits(self):
        return Limits(search_calls=1, read_calls=2, llm_calls=16, decision_calls=64,
                      repair_search_reserve=0, repair_read_reserve=0,
                      terminal_llm_reserve=6, terminal_decision_reserve=1)
    def execute(self, directory, method, abstain=False):
        runtime = Runtime(Path(directory) / "run", self.limits())
        llm, retriever = FixtureLLM(runtime, abstain), FixtureRetriever()
        backend = FixtureBackend(runtime)
        kernel = SelectionOnly(llm, Selector(backend, runtime), retriever, runtime, method, 2)
        result = kernel.run(Task(id="t", question="A's advisor birthplace?", corpus_ids=["a", "b", "c"]), PLAN)
        return result, kernel, backend, llm, runtime

    def test_scope_keeps_zero_bm25_score_documents(self):
        candidates = fixed_candidates(FixtureRetriever(), Task(id="t", question="q", corpus_ids=["a", "b", "c"]))
        self.assertEqual({candidate.doc_id for candidate in candidates}, {"a", "b", "c"})
        self.assertEqual(next(candidate.retrieval_score for candidate in candidates if candidate.doc_id=="b"), 0)

    def test_unresolved_goals_do_not_prevent_an_answer_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            result, kernel, backend, llm, runtime = self.execute(directory, "rank")
            self.assertTrue(kernel.memory.unresolved)
            self.assertEqual(llm.calls.count("phase1_answer"), 1)
            self.assertEqual((result.answer, result.runtime_valid, result.verified), ("Prague", True, False))
            self.assertEqual((runtime.used["search"], runtime.used["read"]), (1, 2))

    def test_early_next_hop_observation_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            result, kernel, backend, llm, runtime = self.execute(directory, "rank")
            self.assertEqual(kernel.memory.facts[0]["related_goal_ids"], ["g2"])
            self.assertIn("Prague", kernel.memory.facts[0]["text"])
            self.assertIn("Prague", llm.last_answer_context["documents"][0]["sentences"][0]["text"])

    def test_static_scores_are_not_recomputed_but_dynamic_scores_are(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            _, static, backend_static, _, _ = self.execute(first, "utility_static")
            _, dynamic, backend_dynamic, _, _ = self.execute(second, "utility_dynamic")
            self.assertEqual(len(backend_static.requests), 1)
            self.assertEqual(len(backend_dynamic.requests), 2)
            self.assertEqual([doc.id for doc in static.memory.documents.values()], ["a", "b"])
            self.assertEqual([doc.id for doc in dynamic.memory.documents.values()], ["a", "c"])

    def test_reader_never_sees_unselected_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            _, kernel, _, llm, _ = self.execute(directory, "rank")
            self.assertEqual(set(llm.last_answer_context), {"question", "documents"})
            self.assertNotIn("Other additional evidence", json.dumps(llm.last_answer_context))

    def test_explicit_abstention_is_a_valid_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            result, kernel, backend, llm, runtime = self.execute(directory, "rank", True)
            self.assertTrue(result.runtime_valid)
            self.assertEqual(result.status, "ANSWER_ATTEMPTED_ABSTAINED")
            self.assertEqual(llm.calls.count("phase1_answer"), 1)

    def test_same_selected_set_has_identical_reader_context_despite_read_order(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            runtime1 = Runtime(Path(first)/"run", self.limits())
            runtime2 = Runtime(Path(second)/"run", self.limits())
            llm1, llm2 = FixtureLLM(runtime1), FixtureLLM(runtime2)
            retriever = FixtureRetriever()
            memory1, memory2 = Memory("q", PLAN), Memory("q", PLAN)
            memory1.documents = {"d1": retriever.documents["a"], "d2": retriever.documents["b"]}
            memory2.documents = {"d1": retriever.documents["b"], "d2": retriever.documents["a"]}
            SelectionOnly(llm1, None, None, runtime1, "rank", 2).finalize(Task(id="t",question="q"), memory1)
            SelectionOnly(llm2, None, None, runtime2, "rank", 2).finalize(Task(id="t",question="q"), memory2)
            self.assertEqual(llm1.last_answer_context, llm2.last_answer_context)

    def test_invalid_citation_is_rejected_without_reassigning_it(self):
        answer = ShortAnswer(answer="X", citations=[Citation(document_ref="d2", sentence_id=0)], abstain=False)
        with self.assertRaises(ProtocolError):
            validate_answer(answer, {"d1": FixtureRetriever().documents["a"]})

    def test_unknown_goals_in_observation_are_rejected(self):
        progress = Progress(facts=[], unresolved_goal_ids=["unknown"], answer_hypotheses=[])
        with self.assertRaises(ProtocolError):
            validate_progress(progress, FixtureRetriever().documents["a"], {"g1", "g2"})

    def test_json_length_failure_can_regenerate_with_counted_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(Path(directory) / "run", self.limits())
            client = RoleJSONClient(runtime)
            replies = iter([{"choices": [{"finish_reason": "length", "message": {"content": '{"answer":'}}]},
                            {"choices": [{"finish_reason": "stop", "message": {"content":
                                '{"answer":"Prague","citations":[{"document_ref":"d1","sentence_id":0}],"abstain":false}'}}]}])
            payloads = []
            def transport(rt, kind, url, payload, key, critical=False, model=""):
                payloads.append(payload)
                return rt.invoke(kind, payload, lambda: next(replies), critical, model)
            with patch("oec.phase1c.retry_json", transport):
                answer = client.ask("phase1_answer", {}, ShortAnswer, critical=True)
            self.assertEqual(answer.answer, "Prague")
            self.assertEqual(runtime.used["llm"], 2)
            self.assertFalse(payloads[0]["chat_template_kwargs"]["enable_thinking"])
            self.assertEqual(payloads[0]["response_format"]["type"], "json_schema")

    def test_final_llm_reserve_is_available_after_normal_calls_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(Path(directory) / "run", self.limits())
            runtime.used["llm"] = self.limits().llm_calls-self.limits().terminal_llm_reserve
            self.assertFalse(runtime.available("llm"))
            self.assertTrue(runtime.available("llm", critical=True))
            answer = FixtureLLM(runtime).ask("phase1_answer", {"documents": []}, ShortAnswer, critical=True)
            self.assertEqual(answer.answer, "Prague")


if __name__ == "__main__":
    unittest.main()
