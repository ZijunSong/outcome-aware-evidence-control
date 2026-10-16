"""Extra phase3 invariants, followed by the unchanged source-validating shared reader."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oec.runtime import ProtocolError, digest
from scripts.replay_phase1e_reader import main as replay, sha


def preflight(runs, identity):
    manifests = [json.loads((run/"manifest.json").read_text()) for run in runs]
    if len({manifest.get("phase3_policy") for manifest in manifests}) != len(manifests):
        raise ProtocolError("Duplicate phase3 policy")
    roots = {manifest.get("observer_ledger") for manifest in manifests}
    if len(roots) != 1 or None in roots:
        raise ProtocolError("Phase3 arms must share one observer ledger")
    for manifest in manifests:
        if manifest.get("selection_family") != "phase3-controlled-selection-v1":
            raise ProtocolError("Not a phase3 controlled run")
        if manifest.get("phase3_model_identity_sha256") != sha(identity):
            raise ProtocolError("Observer and reader model identity mismatch")
    directory = Path(next(iter(roots)))
    for run in runs:
        for path in run.glob("task_*/events.jsonl"):
            for line in path.read_text().splitlines():
                event = json.loads(line)
                if event["event"] not in {"phase3_observer_cache_hit", "phase3_observer_cache_miss"}:
                    continue
                entry = json.loads((directory/(event["key"]+".json")).read_text())
                body = {key: value for key, value in entry.items() if key != "entry_hash"}
                if entry.get("key") != event["key"] or digest(body) != entry.get("entry_hash"):
                    raise ProtocolError("Observer ledger changed")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    args, unused = parser.parse_known_args()
    preflight(args.runs, args.identity)
    replay()  # Same argv; old reader validates all canonical source texts and task hashes.


if __name__ == "__main__":
    main()
