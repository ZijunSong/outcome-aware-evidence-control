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
from oec.phase1c import Memory, ProbeGoal, ProbePlan
from oec.runtime import Runtime
from oec.types import Candidate, Limits

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SPLIT = load("make_phase1d_split")
ABLATION = load("run_phase1d_ablation")
REPORT = load("report_phase1d")
PLAN = ProbePlan(goals=[ProbeGoal(id="g1", description="Find person", depends_on=[]),
                       ProbeGoal(id="g2", description="Find place", depends_on=["g1"])])


class Recorder:
    def __init__(self):
        self.requests = []
    def decide(self, state, questions, critical=False):
        self.requests.append((state, questions))
        return {key: {"type": "noul", "noul": .8} for key in questions}


class ToolsTests(unittest.TestCase):
    def source(self, directory):
        rows = [{"_id": f"q{i}", "question": "Where was B born?", "answer": "Vienna",
                 "type": "bridge", "supporting_facts": [["Person", 0], ["Place", 0]],
                 "context": [[title, ["B was born in Vienna."]] for title in
                    ("Person", "Place", *[f"Noise{j}" for j in range(8)])]} for i in range(8)]
        source, exclude = Path(directory) / "source.json", Path(directory) / "excluded.jsonl"
        source.write_text(json.dumps(rows))
        exclude.write_text('{"id":"q0"}\n')
        return source, exclude

    def test_split_is_disjoint_excludes_used_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            source, exclude = self.source(directory)
            first = SPLIT.build(source, [exclude], Path(directory)/"one", 2, 3, 17)
            second = SPLIT.build(source, [exclude], Path(directory)/"two", 2, 3, 17)
            self.assertEqual(first["splits"], second["splits"])
            tune, confirm = [set(first["splits"][key]["task_ids"]) for key in ("tune", "confirm")]
            self.assertFalse(tune & confirm)
            self.assertNotIn("q0", tune | confirm)

    def execute(self, variant):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        runtime = Runtime(Path(temporary.name)/"run", Limits())
        backend = Recorder()
        selector = ABLATION.MechanismSelector(backend, runtime, variant)
        memory = Memory("q", PLAN, unresolved=["g1", "g2"])
        candidate = Candidate(doc_id="a", title="t", snippet="B born Vienna", retrieval_score=1)
        selector.score(memory, [candidate])
        memory.facts = [{"text": "B born Vienna", "related_goal_ids": ["g2"]}]
        memory.unresolved, memory.hypotheses, memory.revision = ["g2"], ["Vienna"], 1
        memory.documents = {"d1": object()}
        runtime.used["decision"] = 3
        scores = selector.score(memory, [candidate])
        return backend, scores

    def test_frozen_refresh_has_identical_semantics_after_observation(self):
        backend, _ = self.execute("frozen_refresh")
        self.assertEqual(backend.requests[0], backend.requests[1])

    def test_full_removes_cost_metadata_but_keeps_new_facts_and_target(self):
        backend, _ = self.execute("full")
        state = backend.requests[1][0]
        self.assertEqual(state["target_goal_id"], "g2")
        self.assertTrue(state["known_observations"])
        self.assertTrue(state["answer_hypotheses"])
        self.assertFalse({"budget", "state_revision", "read_document_refs"} & set(state))

    def test_no_facts_removes_fact_text_only_and_keeps_redundancy_penalty(self):
        backend, scores = self.execute("no_facts")
        self.assertEqual(backend.requests[1][0]["known_observations"], [])
        self.assertEqual(backend.requests[1][0]["target_goal_id"], "g2")
        expected = .8 - .15 - .05 * len("B born Vienna") / 240
        self.assertAlmostEqual(scores["a"], expected)

    def test_relevance_refresh_does_not_receive_evidence_or_current_goal(self):
        backend, _ = self.execute("relevance_refresh")
        state, questions = backend.requests[1]
        self.assertEqual(set(state), {"question", "phase", "candidates", "target_goal_id", "target_goal_description"})
        self.assertEqual(state["target_goal_description"], "q")
        self.assertEqual(len({row["instructions"]["question"] for row in questions.values()}), 1)

    @unittest.skipUnless(os.environ.get("OEC_TEST_OFFICIAL_EVALUATOR"),
                         "Set OEC_TEST_OFFICIAL_EVALUATOR to the pinned official evaluator")
    def test_all_ten_ablation_cli_paths_and_offline_report(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                user = json.loads(request["messages"][-1]["content"])
                name = user["schema"]["title"]
                if name == "ProbePlan":
                    answer = PLAN.model_dump()
                elif name == "Progress":
                    answer = {"facts": [{"text": "B born Vienna", "sentence_ids": [0],
                                          "related_goal_ids": ["g2"]}],
                              "unresolved_goal_ids": ["g2"], "answer_hypotheses": ["Vienna"]}
                else:
                    answer = {"answer": "Vienna", "citations": [{"document_ref": "d1", "sentence_id": 0}],
                              "abstain": False}
                response = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer)}}],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 10}}
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            def log_message(self, *unused):
                pass
        class Backend:
            def __init__(self, runtime, model_run=None):
                self.runtime = runtime
            def decide(self, state, questions, critical=False):
                response = {key: {"type": "noul", "noul": .8} for key in questions}
                self.runtime.invoke("decision", {"examples": len(questions), "lengths": [10]*len(questions)},
                                    lambda: response, critical, "fixture")
                return response
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                source, exclude = self.source(directory)
                data = Path(directory)/"data"
                SPLIT.build(source, [exclude], data, 2, 3, 17)
                base = ["--tasks", str(data/"confirm/tasks.jsonl"), "--corpus", str(data/"corpus.jsonl")]
                with patch.dict(os.environ, {"OEC_LLM_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                                            "OEC_LLM_MODEL": "fixture", "OEC_JSON_MODE": "json_schema"}), \
                     patch("oec.local_searchjev.LocalSearchJev", Backend):
                    cli = load("run_phase1c")
                    with patch.object(sys, "argv", ["cli", *base, "--mode", "prepare_plans",
                                                   "--out", str(Path(directory)/"plans")]):
                        cli.main()
                    runs = []
                    for variant in ABLATION.VARIANTS:
                        output = Path(directory)/variant
                        runs.append(output)
                        argv = ["cli", "--variant", variant, "--", *base, "--mode", "selection",
                                "--method", "utility_dynamic", "--plans", str(Path(directory)/"plans/plans.json"),
                                "--local-model-run", "fixture", "--read-k", "2", "--out", str(output)]
                        with patch.object(sys, "argv", argv):
                            ABLATION.main()
                evaluator = Path(os.environ["OEC_TEST_OFFICIAL_EVALUATOR"])
                report_path = Path(directory)/"report.json"
                with patch.object(sys, "argv", ["report", "--runs", *map(str, runs), *base,
                    "--gold", str(data/"confirm/gold.json"), "--official-evaluator", str(evaluator),
                    "--read-k", "2", "--out", str(report_path)]):
                    REPORT.main()
                report = json.loads(report_path.read_text())
                self.assertEqual(len(report["runs"]), 10)
                self.assertTrue(all(value["summary"]["valid"] for value in report["runs"].values()))
                # Deliberately corrupt one reader hash: this must fail rather than produce scores.
                event_path = next(runs[0].glob("task_*/events.jsonl"))
                events = REPORT.rows(event_path)
                next(row for row in events if row["event"] == "answer_attempt")["context_hash"] = "wrong"
                event_path.write_text("".join(json.dumps(row)+"\n" for row in events))
                with patch.object(sys, "argv", ["report", "--runs", str(runs[0]), *base,
                    "--gold", str(data/"confirm/gold.json"), "--official-evaluator", str(evaluator),
                    "--read-k", "2", "--out", str(Path(directory)/"bad.json")]):
                    with self.assertRaisesRegex(ValueError, "Reader saw"):
                        REPORT.main()
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
