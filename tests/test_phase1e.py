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
from pydantic import ValidationError
from oec.reader_v2 import ContextReader, parse_answer, schema_for
from oec.runtime import ProtocolError, Runtime
from oec.types import Limits

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = {"question": "Where was B born?", "documents": [{"document_ref": "d1", "title": "B",
             "sentences": [{"sentence_id": 0, "text": "B was born in Vienna."}]}]}


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/"scripts"/f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def response(content, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10}}


class ReaderTests(unittest.TestCase):
    def runtime(self, directory):
        return Runtime(Path(directory)/"run", Limits(terminal_llm_reserve=6))

    def test_schema_only_allows_actual_selected_sentence_labels(self):
        self.assertEqual(schema_for(CONTEXT)["properties"]["citation_ids"]["items"]["enum"], ["d1:s0"])
        answer = parse_answer('{"answer":"Vienna","citation_ids":["d1:s0"],"abstain":false}', CONTEXT)
        self.assertEqual(answer.citations[0].sentence_id, 0)
        for label in ("d2:s0", "d1:s9"):
            with self.assertRaises(ProtocolError):
                parse_answer(json.dumps({"answer": "Vienna", "citation_ids": [label], "abstain": False}), CONTEXT)

    def test_no_silent_abstention_or_answer_rewrite(self):
        for value in ({"answer": "Vienna", "citation_ids": [], "abstain": False},
                      {"answer": "Vienna", "citation_ids": [], "abstain": True},
                      {"answer": "", "citation_ids": ["d1:s0"], "abstain": True}):
            with self.assertRaises(ProtocolError):
                parse_answer(json.dumps(value), CONTEXT)
        with self.assertRaises(ValidationError):
            parse_answer('{"answer":"","citation_ids":[],"abstain":"true"}', CONTEXT)

    def test_length_regenerates_whole_object_and_charges_both(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(directory)
            replies = iter([response('{"answer":', "length"), response(
                '{"answer":"Vienna","citation_ids":["d1:s0"],"abstain":false}')])
            requests = []
            def transport(rt, kind, url, payload, key, critical, model):
                requests.append(payload)
                return rt.invoke(kind, payload, lambda: next(replies), critical, model)
            with patch("oec.reader_v2.retry_json", transport):
                answer = ContextReader(runtime, "http://fixture/v1", "fixture", "identity").read(CONTEXT)
            self.assertEqual(answer.answer, "Vienna")
            self.assertEqual(runtime.used["llm"], 2)
            self.assertEqual(requests[0]["max_tokens"], 512)
            self.assertFalse(requests[0]["chat_template_kwargs"]["enable_thinking"])

    def test_persistent_bad_citation_remains_protocol_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(directory)
            def transport(rt, kind, url, payload, key, critical, model):
                return rt.invoke(kind, payload, lambda: response(
                    '{"answer":"Vienna","citation_ids":["d1:s99"],"abstain":false}'), critical, model)
            with patch("oec.reader_v2.retry_json", transport), self.assertRaises(ProtocolError):
                ContextReader(runtime, "http://fixture/v1", "fixture", "identity").read(CONTEXT)
            self.assertEqual(runtime.used["llm"], 2)

    def test_empty_context_requires_true_explicit_abstention(self):
        context = {"question": "q", "documents": []}
        schema = schema_for(context)
        self.assertTrue(schema["properties"]["abstain"]["const"])
        self.assertTrue(parse_answer('{"answer":"","citation_ids":[],"abstain":true}', context).abstain)

    def paired_fixture(self, root):
        gold = [{"_id": "a", "answer": "X", "supporting_facts": [["A", 0]]},
                {"_id": "b", "answer": "Y", "supporting_facts": [["A", 0]]}]
        (root/"gold.json").write_text(json.dumps(gold))
        manifest = {"protocol": "phase1e-shared-reader-evaluation-v1", "task_ids": ["a", "b"],
                    "reader_ledger_sha256": "fixture-ledger",
                    "reader_policy": {"frozen": True}, "selection_manifest": {}}
        for index, name in enumerate(("before", "after")):
            path = root/name
            path.mkdir()
            (path/"manifest.json").write_text(json.dumps(manifest))
            rows = [{"task_id": row["_id"], "answer": row["answer"], "selected_titles": ["A"],
                     "runtime_valid": not (index == 0 and row["_id"] == "b"),
                     "failure_stage": "bounded_reader_output" if index == 0 and row["_id"] == "b" else None}
                    for row in gold]
            (path/"predictions.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))

    @unittest.skipUnless(os.environ.get("OEC_TEST_OFFICIAL_EVALUATOR"), "Pinned official evaluator required")
    def test_prospective_endpoint_scores_reader_failure_zero_without_relabelling_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.paired_fixture(root)
            module = load("paired_phase1e")
            result = module.compare(root/"before", root/"after", root/"gold.json",
                Path(os.environ["OEC_TEST_OFFICIAL_EVALUATOR"]), "answer_em")
            self.assertEqual(result["delta"], .5)
            self.assertEqual(result["reader_invalid_baseline"], 1)
            raw = json.loads((root/"before/predictions.jsonl").read_text().splitlines()[1])
            self.assertFalse(raw["runtime_valid"])
            with self.assertRaises(ValueError):
                module.compare(root/"before", root/"after", root/"gold.json",
                    Path(os.environ["OEC_TEST_OFFICIAL_EVALUATOR"]), "answer_em", True)

    def test_new_endpoint_refuses_old_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.paired_fixture(root)
            (root/"before/manifest.json").write_text('{"protocol":"phase1c-fixed-selection-v1"}')
            with self.assertRaises(ValueError):
                load("paired_phase1e").read_arm(root/"before")

    def test_structural_failures_cannot_be_scored_as_bounded_reader_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.paired_fixture(root)
            path = root/"before/predictions.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[1]["failure_stage"] = "selection_scope_corrupted"
            path.write_text("".join(json.dumps(row)+"\n" for row in rows))
            with self.assertRaises(ValueError):
                load("paired_phase1e").read_arm(root/"before")

    def test_deferred_selection_shared_replay_and_corruption_rejection(self):
        state = {"reader_requests": 0, "fail": False}
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                user = json.loads(payload["messages"][-1]["content"])
                name = payload["response_format"]["json_schema"]["name"]
                if name == "ProbePlan":
                    value = {"goals": [{"id": "g1", "description": "Find birthplace", "depends_on": []}]}
                elif name == "Progress":
                    value = {"facts": [], "unresolved_goal_ids": ["g1"], "answer_hypotheses": []}
                else:
                    state["reader_requests"] += 1
                    value = {"answer": "Vienna", "citation_ids": ["d1:s99" if state["fail"] else "d1:s0"],
                             "abstain": False}
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(response(json.dumps(value))).encode())
            def log_message(self, *unused):
                pass
        class Backend:
            def __init__(self, runtime, model_run=None):
                self.runtime = runtime
            def decide(self, state, questions, critical=False):
                result = {key: {"type": "noul", "noul": .8} for key in questions}
                return self.runtime.invoke("decision", {"examples": len(questions), "lengths": [10]*len(questions)},
                                           lambda: result, critical, "fixture")
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                corpus = [{"id": f"doc{i}", "title": f"Title{i}", "sentences": ["B was born in Vienna."]}
                          for i in range(10)]
                tasks = [{"id": f"t{i}", "question": f"Where was B born? example {i}",
                          "corpus_ids": [row["id"] for row in corpus]} for i in range(2)]
                for name, values in (("tasks", tasks), ("corpus", corpus)):
                    (root/f"{name}.jsonl").write_text("".join(json.dumps(row)+"\n" for row in values))
                identity = root/"identity.json"
                identity.write_text('{"fixture":true,"model":"fixture"}')
                base = ["--tasks", str(root/"tasks.jsonl"), "--corpus", str(root/"corpus.jsonl")]
                with patch.dict(os.environ, {"OEC_LLM_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                        "OEC_LLM_MODEL": "fixture", "OEC_JSON_MODE": "json_schema"}), \
                     patch("oec.local_searchjev.LocalSearchJev", Backend):
                    cli = load("run_phase1c")
                    with patch.object(sys, "argv", ["cli", *base, "--mode", "prepare_plans", "--out", str(root/"plans")]):
                        cli.main()
                    selector, replay = load("run_phase1e_selection"), load("replay_phase1e_reader")
                    runs = []
                    for method in ("rank", "utility_dynamic"):
                        run = root/method
                        runs.append(run)
                        with patch.object(sys, "argv", ["select", "--variant", "native", "--", *base,
                            "--mode", "selection", "--method", method, "--read-k", "2", "--plans", str(root/"plans/plans.json"),
                            "--local-model-run", "fixture", "--out", str(run)]):
                            selector.main()
                        self.assertFalse(any(row["answer_attempted"] for row in replay.lines(run/"predictions.jsonl")))
                    self.assertEqual(state["reader_requests"], 0)
                    argv = ["replay", "--runs", *map(str, runs), *base, "--identity", str(identity), "--read-k", "2"]
                    with patch.object(sys, "argv", [*argv, "--out", str(root/"good")]):
                        replay.main()
                    good = json.loads((root/"good/report.json").read_text())
                    self.assertEqual(good["unique_reader_contexts"], 2)
                    self.assertEqual(state["reader_requests"], 2)
                    predictions = [replay.lines(root/"good/arms"/f"arm_{i:02d}/predictions.jsonl") for i in range(2)]
                    for before, after in zip(*predictions):
                        for key in ("task_id", "answer", "supporting_facts", "runtime_valid", "shared_reader_key"):
                            self.assertEqual(before[key], after[key])
                    self.assertTrue(all(row["runtime_valid"] for row in predictions[0]))
                    state["fail"] = True
                    with patch.object(sys, "argv", [*argv, "--out", str(root/"bad")]), self.assertRaises(SystemExit) as error:
                        replay.main()
                    self.assertEqual(error.exception.code, 2)
                    bad = json.loads((root/"bad/report.json").read_text())
                    self.assertFalse(bad["all_runtime_valid"])
                    self.assertTrue(all(row["invalid_runs"] == 2 for row in bad["arms"]))
                    bundle_path = next(runs[0].glob("task_*/reader_context.json"))
                    bundle = json.loads(bundle_path.read_text())
                    bundle["context"]["question"] = "corrupted"
                    bundle_path.write_text(json.dumps(bundle))
                    calls_before = state["reader_requests"]
                    with patch.object(sys, "argv", [*argv, "--out", str(root/"corrupt")]), self.assertRaises(ProtocolError):
                        replay.main()
                    self.assertEqual(state["reader_requests"], calls_before)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
