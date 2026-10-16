"""Shared reader for frozen BM25 pools. Does not relax the ten-candidate check of the old collector."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oec.runtime import ProtocolError, digest
import scripts.replay_phase1e_reader as old
from scripts.replay_phase3_reader import main as phase3_main


def collect(run, tasks, corpus, tasks_path, corpus_path, k):
    manifest = old.json.loads((run / "manifest.json").read_text())
    if manifest.get("protocol") != "phase1e-selection-deferred-v1" or not manifest.get("completed"):
        raise ProtocolError("Expected a completed deferred-selection run")
    if manifest.get("selection_family") != "phase3-controlled-selection-v1":
        raise ProtocolError("Retrieved reader requires the phase3 family")
    if manifest["tasks_sha256"] != old.sha(tasks_path) or manifest["corpus_sha256"] != old.sha(corpus_path):
        raise ProtocolError("Selection input hash mismatch")
    if manifest.get("online_gold_access") is not False or manifest.get("offline_oracle"):
        raise ProtocolError("Offline/oracle runs forbidden")
    if manifest["read_k"] != k or set(manifest["task_ids"]) != set(tasks):
        raise ProtocolError("Selection K or task ids mismatch")
    directories = list(run.glob("task_*"))
    if len(directories) != len(tasks):
        raise ProtocolError("Missing or extra task outputs")
    predictions = old.indexed(old.lines(run / "predictions.jsonl"), "task_id")
    output = {}
    for directory in directories:
        bundle = old.json.loads((directory / "reader_context.json").read_text())
        task_id = bundle["task_id"]
        task = tasks[task_id]
        pool_size = len(task["corpus_ids"])
        if pool_size < 1 or len(set(task["corpus_ids"])) != pool_size:
            raise ProtocolError("Frozen pool is empty or contains duplicate ids")
        result = old.json.loads((directory / "result.json").read_text())
        if result != predictions[task_id] or result["status"] != "SELECTION_ONLY_NOT_EVALUATED":
            raise ProtocolError("Selection result is not a deferred output")
        events = old.lines(directory / "events.jsonl")
        if any(row["event"] == "invalid_run" for row in events):
            raise ProtocolError("Upstream selection protocol failed")
        selected = [row["doc_id"] for row in events if row["event"] == "phase1c_selected"]
        if not bundle["selection_stage_valid"] or len(set(selected)) != k or len(selected) != k:
            raise ProtocolError("Selection did not read exactly K unique documents")
        if result["usage"].get("read") != k or not set(selected) <= set(task["corpus_ids"]):
            raise ProtocolError("Read count or frozen-pool scope mismatch")
        read_responses = [row["response"] for row in events if row["event"] == "response" and row.get("kind") == "read"]
        if len(read_responses) != k or {row["id"] for row in read_responses} != set(selected):
            raise ProtocolError("Read events differ from selected documents")
        for row in read_responses:
            expected = corpus[row["id"]]
            if row["title"] != expected["title"] or row["sentences"] != expected["sentences"]:
                raise ProtocolError("Read text differs from frozen corpus")
        documents = [corpus[key] for key in sorted(selected)]
        context = {"question": task["question"], "documents": [{"document_ref": f"d{index + 1}",
            "title": document["title"], "sentences": [{"sentence_id": sid, "text": sentence}
            for sid, sentence in enumerate(document["sentences"])]} for index, document in enumerate(documents)]}
        if bundle["context"] != context or bundle["context_hash"] != digest(context):
            raise ProtocolError("Reader context is not the canonical selected source text")
        if bundle["selected_doc_ids"] != sorted(selected):
            raise ProtocolError("Reader selected set mismatch")
        output[task_id] = {"context": context, "selection_result": result,
                           "selected_sequence": selected, "source_directory": str(directory)}
    return manifest, output


def main():
    old.collect = collect
    phase3_main()


if __name__ == "__main__":
    main()
