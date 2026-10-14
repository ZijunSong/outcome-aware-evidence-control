import argparse
import importlib.util
import json
import math
import random
from pathlib import Path
from evaluate_hotpot import load_predictions


def compare(baseline_path, method_path, gold_path, evaluator, seed=17, samples=10000):
    baseline, method = load_predictions(baseline_path), load_predictions(method_path)
    gold = {row["_id"]: row for row in json.loads(gold_path.read_text(encoding="utf-8"))}
    if set(baseline) != set(method) or set(baseline) != set(gold):
        raise ValueError("Paired comparison requires identical question ids")
    if any(not row["runtime_valid"] for row in [*baseline.values(), *method.values()]):
        raise ValueError("Fix invalid runtime rows before interpreting method effectiveness")
    specification = importlib.util.spec_from_file_location("official_hotpot", evaluator)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    if not hasattr(module, "exact_match_score"):
        raise ValueError("Pinned official evaluator has no exact_match_score; inspect its interface")
    differences, wrong_to_right, right_to_wrong = [], 0, 0
    for task_id in sorted(gold):
        expected = gold[task_id]["answer"]
        before = int(module.exact_match_score(baseline[task_id]["answer"], expected))
        after = int(module.exact_match_score(method[task_id]["answer"], expected))
        differences.append(after - before)
        wrong_to_right += int(before == 0 and after == 1)
        right_to_wrong += int(before == 1 and after == 0)
    n = len(differences)
    if not n:
        raise ValueError("No questions to compare")
    rng = random.Random(seed)
    draws = sorted(sum(rng.choice(differences) for _ in range(n)) / n for _ in range(samples))
    discordant = wrong_to_right + right_to_wrong
    # Exact two-sided binomial McNemar test; avoid degenerate normal-variance claims.
    lower_tail = sum(math.comb(discordant, k) for k in range(min(wrong_to_right, right_to_wrong) + 1))
    p_value = min(1.0, 2 * lower_tail / 2 ** discordant) if discordant else 1.0
    return {"n": n, "delta_em": sum(differences) / n,
            "paired_bootstrap_95ci": [draws[int(0.025*samples)], draws[int(0.975*samples)]],
            "wrong_to_right": wrong_to_right, "right_to_wrong": right_to_wrong,
            "mcnemar_exact_p": p_value,
            "interpretation": "NO_DISCORDANCE_OBSERVED" if not discordant else "ESTIMATED_DIFFERENCE",
            "warning": "An interval containing zero is not evidence of equivalence; a zero-width "
                       "empirical interval with no observed discordance does not prove zero population noise."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--official-evaluator", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(compare(args.baseline, args.method, args.gold, args.official_evaluator),
                                    ensure_ascii=False, indent=2), encoding="utf-8")
