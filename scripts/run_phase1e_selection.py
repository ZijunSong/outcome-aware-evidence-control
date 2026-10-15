"""Freeze selection with the old observer/selector and defer ALL final answers."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oec.phase1c import SelectionOnly
from oec.types import Result
from oec.runtime import digest


class DeferredSelection(SelectionOnly):
    def finalize(self, task, memory):
        documents = sorted(memory.documents.values(), key=lambda document: document.id)
        context = {"question": task.question, "documents": [{"document_ref": f"d{index+1}",
            "title": document.title, "sentences": [{"sentence_id": sid, "text": sentence}
            for sid, sentence in enumerate(document.sentences)]} for index, document in enumerate(documents)]}
        bundle = {"task_id": task.id, "context": context, "context_hash": digest(context),
                  "selected_doc_ids": [document.id for document in documents],
                  "selection_stage_valid": len(documents) == self.read_k}
        (self.runtime.path.parent / "reader_context.json").write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
        self.runtime.event("reader_deferred", context_hash=bundle["context_hash"],
                           selection_stage_valid=bundle["selection_stage_valid"])
        # This is not a prediction, abstention or valid runtime evaluation.
        return Result(task_id=task.id, answer="", supporting_facts=[], runtime_valid=False,
                      verified=False, status="SELECTION_ONLY_NOT_EVALUATED", repairs=0,
                      usage=dict(self.runtime.used) | dict(self.runtime.tokens))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="native")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    for flag in ("--mode", "--out", "--method"):
        if forwarded.count(flag) != 1 or forwarded.index(flag)+1 == len(forwarded):
            parser.error(f"Pass exactly one {flag} VALUE")
    if forwarded[forwarded.index("--mode")+1] != "selection":
        parser.error("Only selection mode is allowed")
    output = Path(forwarded[forwarded.index("--out")+1])
    root = Path(__file__).resolve().parents[1]
    specification = importlib.util.spec_from_file_location("phase1e_selection_cli", root/"scripts/run_phase1c.py")
    cli = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(cli)
    cli.SelectionOnly = DeferredSelection
    if args.variant != "native":
        from scripts.run_phase1d_ablation import VARIANTS, MechanismSelector
        if args.variant not in VARIANTS or forwarded[forwarded.index("--method")+1] != "utility_dynamic":
            parser.error("Mechanism variants need a valid variant and --method utility_dynamic")
        cli.Selector = lambda backend, runtime: MechanismSelector(backend, runtime, args.variant)
    completed = False
    try:
        sys.argv = [str(root/"scripts/run_phase1c.py"), *forwarded]
        cli.main()
        completed = True
    finally:
        path = output/"manifest.json"
        if path.is_file():
            manifest = json.loads(path.read_text())
            manifest.update(protocol="phase1e-selection-deferred-v1", variant=args.variant,
                selection_family="legacy-native" if args.variant == "native" else "masked-mechanism",
                completed=completed, final_answers_deferred=True,
                prediction_file_is_evaluable=False,
                additional_source_sha256={str(file.relative_to(root)): hashlib.sha256(file.read_bytes()).hexdigest()
                    for file in (Path(__file__), root/"scripts/run_phase1d_ablation.py", root/"scripts/run_phase1c.py")})
            path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
