"""Build immutable K=2 checkpoints from a finished native dynamic selection and its shared draft."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oec.phase2b import checkpoint_from_logs
from oec.runtime import ProtocolError, digest
from scripts.replay_phase1e_reader import collect, indexed, lines, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-run", type=Path, required=True)
    parser.add_argument("--draft-reader-root", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Refusing to overwrite checkpoints")
    tasks, corpus = indexed(lines(args.tasks), "id"), indexed(lines(args.corpus), "id")
    manifest, contexts = collect(args.selection_run, tasks, corpus, args.tasks, args.corpus, 2)
    if manifest.get("method") != "utility_dynamic" or manifest.get("variant") != "native":
        raise ProtocolError("Prefix must be the frozen native dynamic C2 family")
    if manifest.get("selection_family") != "legacy-native":
        raise ProtocolError("Masked mechanism runs cannot supply this prefix")
    report = json.loads((args.draft_reader_root / "report.json").read_text())
    if report.get("protocol") != "phase1e-shared-reader-evaluation-v1" or len(report["arms"]) != 1:
        raise ProtocolError("Draft reader must be one shared arm for this prefix")
    arm = json.loads((args.draft_reader_root / "arms" / "arm_00" / "manifest.json").read_text())
    if Path(arm["selection_run"]).resolve() != args.selection_run.resolve():
        raise ProtocolError("Draft reader was not produced from this selection run")
    if arm["reader_ledger_sha256"] != report["reader_ledger_sha256"]:
        raise ProtocolError("Draft ledger mismatch")
    manifest_sha = hashlib.sha256((args.selection_run / "manifest.json").read_bytes()).hexdigest()
    args.out.mkdir(parents=True)
    rows = []
    for task_id in sorted(tasks):
        directory = Path(contexts[task_id]["source_directory"])
        events = lines(directory / "events.jsonl")
        reader_context = json.loads((directory / "reader_context.json").read_text())
        key = arm["reader_keys"][task_id]
        entry = json.loads((args.draft_reader_root / "reader_pool" / key / "entry.json").read_text())
        if entry["context_hash"] != reader_context["context_hash"]:
            raise ProtocolError("Draft context hash does not match this question")
        checkpoint = checkpoint_from_logs(tasks[task_id], corpus, events, reader_context, entry, key, manifest_sha)
        path = args.out / f"{task_id}.json"
        path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append({"task_id": task_id, "checkpoint_sha256": checkpoint["checkpoint_sha256"]})
    ledger = {"protocol": "phase2b-prefix-ledger-v1", "selection_run": str(args.selection_run),
              "draft_reader_root": str(args.draft_reader_root), "tasks_sha256": sha(args.tasks),
              "corpus_sha256": sha(args.corpus), "plans_sha256": manifest["plans_sha256"],
              "prefix_selection_manifest_sha256": manifest_sha,
              "draft_reader_ledger_sha256": report["reader_ledger_sha256"],
              "checkpoints": rows, "ledger_sha256": digest(rows)}
    (args.out / "ledger.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n": len(rows), "ledger_sha256": ledger["ledger_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
