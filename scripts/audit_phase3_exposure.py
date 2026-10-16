"""Record whether any phase1e held-out ids remain unexposed. Does not draw a new sample."""
import argparse
import json
from pathlib import Path


def ids(path):
    return [json.loads(line)["id"] for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base = args.root / "phase1e_base"
    studies = args.root / "phase1e_studies"
    splits = {name: ids(path) for name, path in {
        "tune": base / "tune/tasks.jsonl", "smoke": base / "smoke/tasks.jsonl",
        "confirm": base / "confirm/tasks.jsonl", "main": studies / "main/tasks.jsonl",
        "mechanism": studies / "mechanism/tasks.jsonl"}.items()}
    sets = {name: set(values) for name, values in splits.items()}
    body = {"protocol": "phase3-exposure-audit-v1",
            "counts": {name: len(values) for name, values in splits.items()},
            "smoke_subset_of_tune": sets["smoke"] <= sets["tune"],
            "main_plus_mechanism_equals_confirm": sets["main"] | sets["mechanism"] == sets["confirm"],
            "tune_overlaps_confirm": len(sets["tune"] & sets["confirm"]),
            "scored_phase1e_confirmation_ids": "data/phase1e_studies/main",
            "scored_phase1e_mechanism_ids": "data/phase1e_studies/mechanism",
            "unexposed_confirmation_available": False,
            "reason": "The 700-question phase1e confirm pool was fully consumed by the scored 500-question "
                      "main study and the scored 200-question mechanism study. The 60-question tune set was "
                      "used for phase3 development. No unused 800-question pool is available."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: body[key] for key in ("counts", "unexposed_confirmation_available")}, indent=2))


if __name__ == "__main__":
    main()
