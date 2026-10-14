import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_predictions(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    predictions = {row["task_id"]: row for row in rows}
    if len(rows) != len(predictions):
        raise ValueError("Duplicate prediction ids")
    return predictions


def evaluate(predictions_path: Path, gold_path: Path, evaluator: Path, output: Path):
    predictions = load_predictions(predictions_path)
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    if set(predictions) != {row["_id"] for row in gold}:
        raise ValueError("Prediction ids must exactly match the evaluated gold subset")
    for prediction in predictions.values():
        if not isinstance(prediction["answer"], str):
            raise ValueError("Prediction answer must be the saved raw answer string")
    output.mkdir(parents=True, exist_ok=False)
    official = {"answer": {key: value["answer"] for key, value in predictions.items()},
                "sp": {key: value["supporting_facts"] for key, value in predictions.items()}}
    official_path = output / "official_predictions.json"
    official_path.write_text(json.dumps(official, ensure_ascii=False), encoding="utf-8")
    process = subprocess.run([sys.executable, str(evaluator.resolve()),
                              str(official_path.resolve()), str(gold_path.resolve())],
                             text=True, capture_output=True, check=False)
    invalid = sum(not row["runtime_valid"] for row in predictions.values())
    report = {"n": len(predictions), "invalid_runs": invalid,
              "status": ("OFFICIAL_EVALUATOR_FAILED" if process.returncode
                         else "INVALID_RUNTIME_EVALUATION" if invalid else "VALID_RUNTIME"),
              "official_evaluator_returncode": process.returncode,
              "official_metrics_stdout": process.stdout,
              "official_metrics_stderr": process.stderr}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if process.returncode:
        raise RuntimeError("Official evaluation failed; inspect report.json for its exact stderr")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--official-evaluator", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.predictions, args.gold, args.official_evaluator, args.out)
