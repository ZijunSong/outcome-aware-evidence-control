import copy
import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from oec.phase1c import Memory, ProbePlan, ProbeGoal, Progress, RoleJSONClient
from oec.phase3_observer import SharedObserver
from oec.phase3_selectors import ControlledSelector, POLICIES
from oec.runtime import Runtime, ProtocolError
from oec.types import Limits, Candidate

ROOT = Path(__file__).resolve().parents[1]


def runtime(path):
    return Runtime(path, Limits(search_calls=1, read_calls=4, llm_calls=64, decision_calls=200,
        repair_search_reserve=0, repair_read_reserve=0, terminal_llm_reserve=6, terminal_decision_reserve=1))


class Backend:
    def __init__(self, runtime=None, model_run=None):
        self.runtime, self.requests = runtime, []
    def decide(self, state, questions, critical=False):
        self.requests.append((copy.deepcopy(state), copy.deepcopy(questions)))
        return {key: {"type": "noul", "noul": .8 if value["instructions"]["candidate_doc_id"] == "doc0" else .4}
                for key, value in questions.items()}


class SelectorsTests(unittest.TestCase):
    def memory(self):
        plan = ProbePlan(goals=[ProbeGoal(id="g1", description="Identify advisor", depends_on=[])])
        return Memory("q", plan, facts=[{"text":"known fact", "document_ref":"d1", "related_goal_ids":["g1"]}],
                      unresolved=["g1"], hypotheses=["B"])
    def candidates(self):
        return [Candidate(doc_id=f"doc{i}",title=f"D{i}",snippet="short visible text",retrieval_score=1) for i in range(2)]

    def test_question_only_has_no_state_or_numerical_state_leakage(self):
        with tempfile.TemporaryDirectory() as raw:
            rt, backend = runtime(Path(raw)/"run"), Backend()
            selector = ControlledSelector(backend, rt, "question_relevance")
            memory = self.memory()
            a = selector.score(memory, self.candidates())
            memory.facts = [{"text":"short visible text"}]
            b = selector.score(memory, self.candidates())
            self.assertEqual(a,b)
            state = backend.requests[0][0]
            self.assertNotIn("known_observations", state)
            self.assertNotIn("goals", state)
            self.assertEqual(state["target_goal_description"], "q")

    def test_state_baseline_and_utility_receive_identical_state_and_decision_count(self):
        with tempfile.TemporaryDirectory() as raw:
            rt = runtime(Path(raw)/"run")
            backends = [Backend(), Backend()]
            original = self.memory()
            frozen = copy.deepcopy(original.facts)
            for policy, backend in zip(("utility","state_relevance"), backends):
                ControlledSelector(backend, rt, policy).score(original, self.candidates())
            self.assertEqual(backends[0].requests[0][0], backends[1].requests[0][0])
            self.assertEqual(len(backends[0].requests[0][1]), 6)
            self.assertEqual(len(backends[1].requests[0][1]), 6)
            for key in ("budget","state_revision","read_document_refs"):
                self.assertNotIn(key, backends[0].requests[0][0])
            self.assertEqual(original.facts, frozen)

    def test_typed_backend_invalid_result_is_rejected(self):
        class Bad(Backend):
            def decide(self,*args,**kwargs):
                return {}
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(ProtocolError):
                ControlledSelector(Bad(),runtime(Path(raw)/"run"),"utility").score(self.memory(),self.candidates())


class ObserverTests(unittest.TestCase):
    def test_first_result_reused_and_model_identity_separates_keys(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            def fake(client, role, context, schema, critical=False, validator=None):
                client.runtime.used["llm"] += 1
                client.runtime.tokens["llm.prompt_tokens"] += 10
                return Progress(facts=[], unresolved_goal_ids=["g1"], answer_hypotheses=[])
            with patch.object(RoleJSONClient,"ask",fake) as mocked:
                a = SharedObserver(runtime(root/"a"),root/"cache","identity1")
                b = SharedObserver(runtime(root/"b"),root/"cache","identity1")
                c = SharedObserver(runtime(root/"c"),root/"cache","identity2")
                one = a.ask("phase1_observe", {"question":"q"}, Progress)
                two = b.ask("phase1_observe", {"question":"q"}, Progress)
                c.ask("phase1_observe", {"question":"q"}, Progress)
                self.assertEqual(one,two)
                self.assertEqual(b.runtime.used["llm"],1)
                self.assertEqual(len(list((root/"cache").glob("*.json"))),2)
                self.assertIn("phase3_observer_cache_hit",b.runtime.path.read_text())

    def test_bounded_failure_is_shared_without_becoming_an_abstention_or_success(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            def fail(*args,**kwargs):
                raise ProtocolError("phase1_observe: two output attempts failed")
            with patch.object(RoleJSONClient,"ask",fail):
                for name in ("a","b"):
                    client=SharedObserver(runtime(root/name),root/"cache","identity")
                    with self.assertRaises(ProtocolError):
                        client.ask("phase1_observe",{},Progress)
            entries=[json.loads(path.read_text()) for path in (root/"cache").glob("*.json")]
            self.assertEqual(entries[0]["status"],"bounded_protocol_failure")

    def test_changed_cache_bytes_abort_instead_of_silently_using_them(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw)
            value=Progress(facts=[],unresolved_goal_ids=[],answer_hypotheses=[])
            with patch.object(RoleJSONClient,"ask",return_value=value):
                a=SharedObserver(runtime(root/"a"),root/"cache","id")
                a.ask("phase1_observe",{},Progress)
            path=next((root/"cache").glob("*.json"))
            entry=json.loads(path.read_text());entry["value"]["answer_hypotheses"]=["changed"]
            path.write_text(json.dumps(entry))
            with self.assertRaises(RuntimeError):
                SharedObserver(runtime(root/"b"),root/"cache","id").ask("phase1_observe",{},Progress)


class CostTests(unittest.TestCase):
    def test_aliases_are_not_double_counted_and_cached_cost_is_logical(self):
        path=ROOT/"scripts/summarize_phase3_selection_cost.py"
        spec=importlib.util.spec_from_file_location("cost",path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        result=module.event_cost([
            {"event":"request","kind":"decision","request":{"examples":2,"lengths":[10,20]}},
            {"event":"request","kind":"llm","request":{}},
            {"event":"response","kind":"llm","response":{"usage":{"prompt_tokens":10,"input_tokens":10,"completion_tokens":2}},"latency_s":1},
            {"event":"phase3_observer_cache_hit","logical_used":{"llm":1},"logical_tokens":{"llm.prompt_tokens":20,"llm.completion_tokens":3}}])
        self.assertEqual(result["native_encoded_tokens"],30)
        self.assertEqual(result["physical_llm_input_tokens"],10)
        self.assertEqual(result["logical_llm_input_tokens"],30)
        self.assertEqual(result["logical_llm_requests"],2)


class HTTPChainTests(unittest.TestCase):
    def test_all_four_policies_reuse_original_collector_and_shared_reader(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                name=payload["response_format"]["json_schema"]["name"]
                if name=="ProbePlan":
                    value={"goals":[{"id":"g1","description":"Find a birthplace","depends_on":[]}]}
                elif name=="Progress":
                    value={"facts":[],"unresolved_goal_ids":["g1"],"answer_hypotheses":[]}
                elif name=="Phase3SetwiseChoice":
                    value={"doc_id":payload["response_format"]["json_schema"]["schema"]["properties"]["doc_id"]["enum"][0]}
                else:
                    value={"answer":"Paris","citation_ids":["d1:s0"],"abstain":False}
                result={"choices":[{"finish_reason":"stop","message":{"content":json.dumps(value)}}],"usage":{"prompt_tokens":10,"completion_tokens":10}}
                self.send_response(200);self.end_headers();self.wfile.write(json.dumps(result).encode())
            def log_message(self,*unused):
                pass
        server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as raw:
                root=Path(raw)
                docs=[{"id":f"doc{i}","title":f"D{i}","sentences":["A was born in Paris."]} for i in range(10)]
                tasks=[{"id":"t","question":"Where was A born?","corpus_ids":[row["id"] for row in docs]}]
                for name,rows in (("tasks",tasks),("corpus",docs)):
                    (root/(name+".jsonl")).write_text("".join(json.dumps(row)+"\n" for row in rows))
                (root/"identity.json").write_text('{"fixture":true}')
                common=["--tasks",str(root/"tasks.jsonl"),"--corpus",str(root/"corpus.jsonl")]
                with patch.dict(os.environ,{"OEC_LLM_BASE_URL":f"http://127.0.0.1:{server.server_port}/v1","OEC_LLM_MODEL":"fixture","OEC_JSON_MODE":"json_schema"}),patch("oec.local_searchjev.LocalSearchJev",Backend):
                    from scripts.run_phase1c import main as plan
                    from scripts.run_phase3_selection import main as select
                    from scripts.replay_phase3_reader import main as replay
                    with patch.object(sys,"argv",["plan",*common,"--mode","prepare_plans","--out",str(root/"plans")]):plan()
                    for policy in POLICIES:
                        with patch.object(sys,"argv",["select","--policy",policy,"--observer-cache",str(root/"cache"),"--identity",str(root/"identity.json"),"--",*common,"--mode","selection","--method","utility_dynamic","--backend","searchjev","--local-model-run","fixture","--plans",str(root/"plans/plans.json"),"--read-k","2","--out",str(root/policy)]):select()
                    with patch.object(sys,"argv",["replay","--runs",*[str(root/policy) for policy in POLICIES],*common,"--identity",str(root/"identity.json"),"--read-k","2","--out",str(root/"reader")]):replay()
                    report=json.loads((root/"reader/report.json").read_text())
                    self.assertTrue(report["all_runtime_valid"])
                    self.assertEqual(report["unique_reader_contexts"],1)
                    self.assertEqual(len(list((root/"cache").glob("*.json"))),2)
                    identity = root/"different_identity.json"
                    identity.write_text('{"fixture":false}')
                    from scripts.replay_phase3_reader import preflight
                    with self.assertRaises(ProtocolError):
                        preflight([root/policy for policy in POLICIES], identity)
        finally:
            server.shutdown();server.server_close()


if __name__=="__main__":
    unittest.main()
