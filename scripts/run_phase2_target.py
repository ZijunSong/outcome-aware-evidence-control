"""Runnable controlled C1 experiment. It does not restore the old complete Agent."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oec.phase2_selector import ScheduledSelector
from scripts.run_phase1e_selection import DeferredSelection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["fixed", "priority"], required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    for flag, value in (("--mode", "selection"), ("--method", "utility_dynamic"), ("--backend", "searchjev")):
        if forwarded.count(flag) != 1 or forwarded.index(flag)+1 >= len(forwarded) or forwarded[forwarded.index(flag)+1] != value:
            parser.error(f"Pass exactly {flag} {value} after --")
    if forwarded.count("--out") != 1 or forwarded.index("--out")+1 >= len(forwarded):
        parser.error("Pass --out PATH")
    output = Path(forwarded[forwarded.index("--out")+1])
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("phase2_target_cli", root/"scripts/run_phase1c.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    cli.SelectionOnly = DeferredSelection
    cli.Selector = lambda backend, runtime: ScheduledSelector(backend, runtime, args.policy)
    completed = False
    try:
        sys.argv = [str(root/"scripts/run_phase1c.py"), *forwarded]
        cli.main()
        completed = True
    finally:
        path = output/"manifest.json"
        if path.is_file():
            manifest = json.loads(path.read_text())
            names = ("scripts/run_phase2_target.py", "oec/frontier_scheduler.py", "oec/phase2_selector.py",
                     "scripts/run_phase1e_selection.py", "scripts/run_phase1c.py")
            manifest.update(protocol="phase1e-selection-deferred-v1", completed=completed,
                selection_family="phase2-estimated-frontier-routing-v1", target_policy=args.policy,
                matched_shadow_priority=True, final_answers_deferred=True, prediction_file_is_evaluable=False,
                frontier_semantics="observer-estimated unresolved goals; no factual verification",
                additional_source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names})
            path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
