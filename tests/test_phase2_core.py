import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from oec.cache_core import EvidenceCache, ParentBinding
from oec.frontier_scheduler import choose_target, estimated_frontier
from oec.runtime import ProtocolError, Runtime
from oec.types import Candidate, Document, Limits

ROOT = Path(__file__).resolve().parents[1]
GOALS = [{"id": "g1", "description": "Find A", "depends_on": []},
         {"id": "g2", "description": "Find B", "depends_on": []},
         {"id": "g3", "description": "Compare A and B", "depends_on": ["g1", "g2"]}]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/"scripts"/f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Backend:
    def __init__(self, runtime, model_run=None):
        self.runtime, self.requests, self.defer = runtime, [], False
    def decide(self, state, questions, critical=False):
        self.requests.append((state, questions))
        result = {}
        for key, question in questions.items():
            if question["type"] == "choice":
                labels = list(question["criteria"])
                chosen = "DEFER" if self.defer else ("g2" if "g2" in labels else labels[0])
                probabilities = {label: .8 if label == chosen else .2/(len(labels)-1) for label in labels}
                result[key] = {"type": "choice", "choice": chosen, "probabilities": probabilities, "confidence": .7}
            else:
                doc_id = question["instructions"]["candidate_doc_id"]
                preferred = ("doc0", "doc2") if state["target_goal_id"] == "g1" else ("doc1", "doc3")
                result[key] = {"type": "noul", "noul": .95 if doc_id == preferred[0] else .9 if doc_id == preferred[1] else .1}
        return self.runtime.invoke("decision", {"examples": len(questions), "lengths": [10]*len(questions)},
                                   lambda: result, critical, "scripted_fixture")


class Phase2Tests(unittest.TestCase):
    def runtime(self, root):
        return Runtime(Path(root)/"run", Limits(repair_search_reserve=0, repair_read_reserve=0,
                      terminal_llm_reserve=6, terminal_decision_reserve=1))
    def cache(self):
        return EvidenceCache(["a", "b", "c"], {"g1": (), "g2": ("g1",)}, "fixture-retriever-v1")
    def document(self, key="a"):
        return Document(id=key, title=key, sentences=["B was born in Prague."])

    def test_document_cache_charges_unique_read_once_and_returns_copies(self):
        with tempfile.TemporaryDirectory() as root:
            runtime, cache = self.runtime(root), self.cache()
            first = cache.fetch_once("a", runtime, self.document)
            first.sentences[0] = "locally changed"
            second = cache.fetch_once("a", runtime, self.document)
            self.assertEqual(runtime.used["read"], 1)
            self.assertEqual(second.sentences, ["B was born in Prague."])

    def test_query_cache_keeps_unread_candidates_and_does_not_charge_twice(self):
        with tempfile.TemporaryDirectory() as root:
            runtime, cache = self.runtime(root), self.cache()
            def search(query, top_n):
                return [Candidate(doc_id=key, title=key, snippet=key, retrieval_score=1) for key in ("a", "b")]
            cache.retrieve_once("q", 2, runtime, search)
            cache.fetch_once("a", runtime, self.document)
            cache.retrieve_once("q", 2, runtime, search)
            self.assertEqual(runtime.used["search"], 1)
            self.assertEqual([row.doc_id for row in cache.unread_candidates()], ["b"])

    def test_scope_and_immutable_source_are_enforced(self):
        cache = self.cache()
        cache.put_document(self.document())
        with self.assertRaises(ProtocolError):
            cache.put_document(Document(id="a", title="a", sentences=["changed"]))
        with self.assertRaises(ProtocolError):
            cache.merge_candidates([Candidate(doc_id="foreign", title="x", snippet="x", retrieval_score=1)], "q")

    def test_early_fact_becomes_retryable_without_becoming_supported(self):
        cache = self.cache()
        cache.put_document(self.document())
        cache.add_pending("a", "g2", [0], "Birthplace proposal for B")
        self.assertFalse(cache.ready_attempts({}))
        binding = ParentBinding("g1", "c1", "B", "v1")
        ready = cache.ready_attempts({"g1": [binding]})
        self.assertEqual(len(ready), 1)
        self.assertFalse(hasattr(cache, "claims"))

    def test_parent_change_retries_but_global_unrelated_changes_do_not(self):
        cache = self.cache()
        cache.put_document(self.document())
        cache.add_pending("a", "g2", [0], "Birthplace proposal")
        bindings = {"g1": [ParentBinding("g1", "c1", "B", "v1")]}
        attempt = cache.ready_attempts(bindings)[0]
        self.assertTrue(cache.begin_attempt(attempt))
        self.assertFalse(cache.begin_attempt(attempt))
        cache.finish_attempt(attempt, "REJECTED")
        bindings["unrelated"] = [ParentBinding("unrelated", "c9", "Other", "v9")]
        self.assertFalse(cache.ready_attempts(bindings))
        bindings["g1"] = [ParentBinding("g1", "c2", "C", "v2")]
        new = cache.ready_attempts(bindings)[0]
        self.assertNotEqual(new.key, attempt.key)

    def test_ambiguous_or_unsupported_parent_blocks_pending_retry(self):
        cache = self.cache()
        cache.put_document(self.document())
        cache.add_pending("a", "g2", [0], "Birthplace")
        b = ParentBinding("g1", "c1", "B", "v1")
        c = ParentBinding("g1", "c2", "C", "v1")
        self.assertFalse(cache.ready_attempts({"g1": [b, c]}))
        self.assertFalse(cache.ready_attempts({"g1": [ParentBinding("g1", "c1", "B", "v1", False)]}))

    def test_pending_rejects_fabricated_sentence(self):
        cache = self.cache()
        cache.put_document(self.document())
        with self.assertRaises(ProtocolError):
            cache.add_pending("a", "g2", [99], "bad")
        with self.assertRaises(ProtocolError):
            cache.add_pending("a", "g2", [0, False], "bad coerced id")

    def test_frontier_respects_dependencies_and_detects_invalid_graphs(self):
        self.assertEqual(estimated_frontier(GOALS, ["g1", "g2", "g3"]), ["g1", "g2"])
        self.assertEqual(estimated_frontier(GOALS, ["g3"]), ["g3"])
        with self.assertRaises(ProtocolError):
            estimated_frontier([{"id":"g1", "depends_on":["g1"]}], ["g1"])

    def test_singleton_uses_no_priority_call(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = self.runtime(root)
            backend = Backend(runtime)
            self.assertEqual(choose_target(GOALS, ["g2", "g3"], {}, backend, runtime, "priority"), "g2")
            self.assertEqual(runtime.used["decision"], 0)

    def test_shadow_fixed_and_priority_get_identical_inputs_and_different_targets(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            runtime1, runtime2 = self.runtime(first), self.runtime(second)
            backend1, backend2 = Backend(runtime1), Backend(runtime2)
            semantic = {"question": "q", "budget": {"used": 90}, "known_observations": []}
            a = choose_target(GOALS, ["g1", "g2", "g3"], semantic, backend1, runtime1, "fixed")
            b = choose_target(GOALS, ["g1", "g2", "g3"], semantic, backend2, runtime2, "priority")
            self.assertEqual((a,b), ("g1", "g2"))
            self.assertEqual(backend1.requests, backend2.requests)
            self.assertNotIn("budget", backend1.requests[0][0])

    def test_defer_falls_back_and_estimated_completion_does_not_require_verification(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = self.runtime(root)
            backend = Backend(runtime)
            backend.defer = True
            self.assertEqual(choose_target(GOALS, ["g1", "g2", "g3"], {}, backend, runtime, "priority"), "g1")
            self.assertEqual(choose_target(GOALS, [], {}, backend, runtime, "priority"), "ANSWER")

    def test_real_cli_wiring_uses_targets_and_shared_reader(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                name = payload["response_format"]["json_schema"]["name"]
                if name == "ProbePlan":
                    value = {"goals": GOALS}
                elif name == "Progress":
                    value = {"facts": [], "unresolved_goal_ids": ["g1", "g2", "g3"], "answer_hypotheses": []}
                else:
                    value = {"answer":"Vienna", "citation_ids":["d1:s0"], "abstain":False}
                response = {"choices":[{"finish_reason":"stop","message":{"content":json.dumps(value)}}],
                            "usage":{"prompt_tokens":10,"completion_tokens":10}}
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            def log_message(self,*unused):
                pass
        server = ThreadingHTTPServer(("127.0.0.1",0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                docs = [{"id":f"doc{i}","title":f"Doc{i}","sentences":["A was born in Vienna."]} for i in range(10)]
                tasks = [{"id":"t","question":"Compare the birthplaces?","corpus_ids":[row["id"] for row in docs]}]
                for name, values in (("tasks",tasks),("corpus",docs)):
                    (root/f"{name}.jsonl").write_text("".join(json.dumps(row)+"\n" for row in values))
                (root/"identity.json").write_text('{"fixture":true}')
                base = ["--tasks",str(root/"tasks.jsonl"),"--corpus",str(root/"corpus.jsonl")]
                with patch.dict(os.environ,{"OEC_LLM_BASE_URL":f"http://127.0.0.1:{server.server_port}/v1",
                     "OEC_LLM_MODEL":"fixture","OEC_JSON_MODE":"json_schema"}), patch("oec.local_searchjev.LocalSearchJev",Backend):
                    cli = load("run_phase1c")
                    with patch.object(sys,"argv",["plan",*base,"--mode","prepare_plans","--out",str(root/"plans")]):
                        cli.main()
                    target = load("run_phase2_target")
                    for policy in ("fixed","priority"):
                        with patch.object(sys,"argv",["target","--policy",policy,"--",*base,"--mode","selection",
                            "--method","utility_dynamic","--backend","searchjev","--local-model-run","fixture",
                            "--plans",str(root/"plans/plans.json"),"--read-k","2","--out",str(root/policy)]):
                            target.main()
                    audit = load("audit_scheduling_opportunities")
                    self.assertEqual(audit.audit(root/"fixed")["opportunity_task_rate"], 1)
                    replay = load("replay_phase1e_reader")
                    fixed = replay.lines(root/"fixed/task_00000/events.jsonl")
                    priority = replay.lines(root/"priority/task_00000/events.jsonl")
                    chosen = lambda events: [row["doc_id"] for row in events if row["event"]=="phase1c_selected"]
                    self.assertEqual(chosen(fixed), ["doc0","doc2"])
                    self.assertEqual(chosen(priority), ["doc1","doc3"])
                    with patch.object(sys,"argv",["replay","--runs",str(root/"fixed"),str(root/"priority"),*base,
                        "--identity",str(root/"identity.json"),"--out",str(root/"reader")]):
                        replay.main()
                    self.assertTrue(json.loads((root/"reader/report.json").read_text())["all_runtime_valid"])
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
