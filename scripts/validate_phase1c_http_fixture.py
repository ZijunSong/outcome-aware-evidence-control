"""Local HTTP/CLI contract test. All model answers are scripted, not benchmark evidence."""
import argparse
import importlib.util
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oec.runtime import ProtocolError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    specification = importlib.util.spec_from_file_location("phase1c_cli", root / "scripts/run_phase1c.py")
    cli = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(cli)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            user = json.loads(request["messages"][-1]["content"])
            name, context = user["schema"]["title"], user["context"]
            assert request["chat_template_kwargs"]["enable_thinking"] is False
            assert request["response_format"]["type"] == "json_schema"
            if name == "ProbePlan":
                answer = {"goals": [{"id": "g1", "description": "Find birthplace", "depends_on": []}]}
            elif name == "Progress":
                answer = {"facts": [], "unresolved_goal_ids": ["g1"], "answer_hypotheses": []}
            elif name == "ShortAnswer":
                citations = [{"document_ref": doc["document_ref"], "sentence_id": 0}
                             for doc in context["documents"] if doc["title"] in ("A doctoral record", "C biography")]
                answer = {"answer": "Vienna" if citations else "", "citations": citations,
                          "abstain": not bool(citations)}
            else:
                raise ProtocolError("Unknown fixture schema")
            response = {"model": "scripted_http_fixture", "choices": [{"finish_reason": "stop",
                "message": {"content": json.dumps(answer)}}], "usage": {"prompt_tokens": 10, "completion_tokens": 10}}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        def log_message(self, *unused):
            pass

    class Backend:
        def __init__(self, runtime, model_run=None):
            self.runtime = runtime
        def decide(self, state, questions, critical=False):
            response = {key: {"type": "noul", "noul": .95} for key in questions}
            self.runtime.invoke("decision", {"state": state, "questions": questions},
                                lambda: response, critical, "scripted_typed_fixture")
            return response

    args.out.mkdir(parents=True, exist_ok=False)
    from scripts.prepare_hotpot import prepare
    prepare(root / "fixtures/demo_hotpot.json", args.out / "data", 1, 0, False)
    base = ["run_phase1c.py", "--tasks", str(args.out / "data/tasks.jsonl"),
            "--corpus", str(args.out / "data/corpus.jsonl")]
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with patch.dict(os.environ, {"OEC_LLM_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                                    "OEC_LLM_MODEL": "scripted_http_fixture", "OEC_JSON_MODE": "json_schema"}), \
             patch("oec.local_searchjev.LocalSearchJev", Backend):
            commands = [base + ["--mode", "prepare_plans", "--out", str(args.out / "plans")],
                        base + ["--mode", "all_context", "--out", str(args.out / "all_context")],
                        base + ["--mode", "oracle_context", "--gold", str(args.out / "data/gold.json"),
                                "--out", str(args.out / "oracle")]]
            methods = ("rank", "relevance", "utility_static", "utility_dynamic")
            commands += [base + ["--mode", "selection", "--plans", str(args.out / "plans/plans.json"),
                                "--local-model-run", "scripted_fixture",
                                "--method", method, "--read-k", "2", "--out", str(args.out / method)]
                         for method in methods]
            for argv in commands:
                with patch.object(sys, "argv", argv):
                    cli.main()
            results = {name: json.loads((args.out / name / "predictions.jsonl").read_text())
                       for name in ("all_context", "oracle", *methods)}
            assert all(row["runtime_valid"] and row["answer_attempted"] for row in results.values())
            assert all(results[name]["usage"]["read"] == 2 for name in methods)
            print(json.dumps({"cli_paths_tested": len(commands), "all_runtime_valid": True,
                              "answer_attempted_all": True, "selection_read_budget_equal": 2,
                              "models": "scripted HTTP and typed fixture; no GPU"}, ensure_ascii=False))
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
