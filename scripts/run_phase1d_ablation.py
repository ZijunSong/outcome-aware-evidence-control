"""Mechanism-only runner. Reuses e89af8d CLI/reader; no online gold access."""
import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oec.phase1c import Memory, Selector
from oec.runtime import digest

VARIANTS = ("full", "frozen_refresh", "relevance_refresh", "no_facts",
            "no_goal_update", "no_hypotheses", "no_redundancy",
            "no_gap", "no_bridge", "no_disamb")


class MechanismSelector(Selector):
    def __init__(self, backend, runtime, variant="full"):
        super().__init__(backend, runtime)
        self.variant, self.initial = variant, None

    def score(self, memory, candidates, relevance=False):
        if relevance:
            raise ValueError("Mechanism arms all use repeated utility scoring")
        if self.initial is None:
            self.initial = Memory(memory.question, memory.plan,
                unresolved=[goal.id for goal in memory.plan.goals])
        initial = self.initial
        proxy = Memory(memory.question, memory.plan, facts=copy.deepcopy(memory.facts),
                       unresolved=list(memory.unresolved), hypotheses=list(memory.hypotheses))
        if self.variant == "frozen_refresh":
            proxy.facts, proxy.unresolved, proxy.hypotheses = [], list(initial.unresolved), []
        elif self.variant == "no_facts":
            proxy.facts = []
        elif self.variant == "no_goal_update":
            proxy.unresolved = list(initial.unresolved)
        elif self.variant == "no_hypotheses":
            proxy.hypotheses = []
        relevance_arm = self.variant == "relevance_refresh"
        view = proxy.view(self.runtime)
        # Remove cost counters, revision and read-id metadata from EVERY mechanism arm.
        # This changes the mechanism protocol; compare these arms only with each other.
        for key in ("budget", "state_revision", "read_document_refs"):
            view.pop(key, None)
        target_id, target_description = proxy.target()
        if relevance_arm:
            view = {"question": memory.question}
            target_id, target_description = "QUESTION", memory.question
        original_backend = self.backend
        original_weights = self.weights
        variant = self.variant
        runtime = self.runtime

        class Adapter:
            def decide(self, state, questions, critical=False):
                state = view | {key: copy.deepcopy(value) for key, value in state.items()
                                if key in ("phase", "candidates")}
                state |= {"target_goal_id": target_id, "target_goal_description": target_description}
                questions = copy.deepcopy(questions)
                if relevance_arm:
                    # Three identical relevance decisions per candidate match the decision count.
                    for question in questions.values():
                        question["instructions"]["question"] = (
                            "The visible candidate is relevant to the original question; "
                            "use the original question and visible snippet only.")
                runtime.event("mechanism_score_context", variant=variant,
                              revision=memory.revision, semantic_state_hash=digest(view),
                              target=target_id, candidate_count=len(state["candidates"]))
                return original_backend.decide(state, questions, critical=critical)

        wg, wb, wh, wr, wc = original_weights
        if variant == "no_gap":
            wg = 0
        elif variant == "no_bridge":
            wb = 0
        elif variant == "no_disamb":
            wh = 0
        if variant == "no_redundancy" or relevance_arm:
            wr = 0
        if relevance_arm:
            wg, wb, wh, wc = 1/3, 1/3, 1/3, 0
        self.backend, self.weights = Adapter(), (wg, wb, wh, wr, wc)
        # Freeze only the semantic scorer fields. Keep the actual facts for the numerical
        # redundancy term, except in frozen_refresh, where the entire state is frozen.
        scoring_memory = copy.copy(proxy)
        if variant != "frozen_refresh":
            scoring_memory.facts = memory.facts
        try:
            return super().score(scoring_memory, candidates, False)
        finally:
            self.backend, self.weights = original_backend, original_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("phase1c_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = args.phase1c_args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    for flag, required in (("--method", "utility_dynamic"), ("--mode", "selection")):
        if flag not in forwarded or forwarded[forwarded.index(flag)+1] != required:
            parser.error(f"Pass {flag} {required} after --")
    if "--out" not in forwarded:
        parser.error("Missing --out")
    output = Path(forwarded[forwarded.index("--out")+1])
    source = Path(__file__).resolve().parents[1] / "scripts/run_phase1c.py"
    specification = importlib.util.spec_from_file_location("phase1c_mechanism_cli", source)
    cli = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(cli)
    cli.Selector = lambda backend, runtime: MechanismSelector(backend, runtime, args.variant)
    sys.argv = [str(source), *forwarded]
    cli.main()
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(protocol="phase1d-mechanism-v1", variant=args.variant,
        mechanism_source_sha256=__import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest(),
        scorer_metadata="budget, state_revision, read_document_refs removed in all mechanism arms",
        note="Compare mechanism arms internally; not identical to e89af8d scorer context")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
