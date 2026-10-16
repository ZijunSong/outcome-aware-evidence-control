"""Controlled selector policies, original deferred output schema, shared observer ledger."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oec.phase3_selectors import ControlledSelector, POLICIES
from oec.phase3_observer import SharedObserver
from scripts.run_phase1e_selection import DeferredSelection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--observer-cache", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    for flag, value in (("--mode", "selection"), ("--method", "utility_dynamic"), ("--backend", "searchjev")):
        if forwarded.count(flag) != 1 or forwarded.index(flag)+1 >= len(forwarded) or forwarded[forwarded.index(flag)+1] != value:
            parser.error("Pass exactly "+flag+" "+value+" after --")
    if forwarded.count("--out") != 1 or forwarded.index("--out")+1 >= len(forwarded):
        parser.error("Pass exactly --out PATH")
    identity = json.loads(args.identity.read_text())
    if not isinstance(identity, dict) or not identity:
        parser.error("Need a nonempty frozen model identity")
    identity_hash = hashlib.sha256(args.identity.read_bytes()).hexdigest()
    output = Path(forwarded[forwarded.index("--out")+1])
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("phase3_target_cli", root/"scripts/run_phase1c.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    cli.SelectionOnly = DeferredSelection
    cli.RoleJSONClient = lambda runtime: SharedObserver(runtime, args.observer_cache, identity_hash)
    cli.Selector = lambda backend, runtime: ControlledSelector(backend, runtime, args.policy)
    import oec.local_searchjev as local_backend
    original_backend = local_backend.LocalSearchJev
    if args.policy == "setwise_llm":
        class NoNativeBackend:
            def __init__(self, runtime, model_run):
                self.runtime = runtime
            def decide(self, *unused, **keywords):
                raise RuntimeError("The setwise arm must not invoke SearchJev")
        local_backend.LocalSearchJev = NoNativeBackend
    completed = False
    try:
        sys.argv = [str(root/"scripts/run_phase1c.py"), *forwarded]
        cli.main()
        completed = True
    finally:
        local_backend.LocalSearchJev = original_backend
        path = output/"manifest.json"
        if path.is_file():
            manifest = json.loads(path.read_text())
            names = ("oec/phase3_selectors.py", "oec/phase3_observer.py", "scripts/run_phase3_selection.py",
                     "scripts/run_phase1c.py", "scripts/run_phase1e_selection.py", "oec/local_searchjev.py")
            manifest.update(protocol="phase1e-selection-deferred-v1", completed=completed,
                selection_family="phase3-controlled-selection-v1", phase3_policy=args.policy,
                scorer_backend="qwen-generator" if args.policy == "setwise_llm" else "local-searchjev",
                phase3_model_identity_sha256=identity_hash,
                observer_ledger=str(args.observer_cache.resolve()),
                observer_cost="runtime counters include logical cache charges; request events are physical",
                final_answers_deferred=True, prediction_file_is_evaluable=False,
                additional_source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names})
            path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
