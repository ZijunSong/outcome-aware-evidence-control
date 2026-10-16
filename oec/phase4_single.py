"""One native Noul per candidate; no numerical penalties; fixed initial-state control."""
import copy
import math
from .phase3_selectors import ControlledSelector, semantic_view
from .runtime import ProtocolError, digest


class OnePredicate:
    def __init__(self, backend):
        self.backend = backend

    def decide(self, state, questions, critical=False):
        unique, owners = {}, {}
        for key, row in questions.items():
            if row["type"] != "noul":
                raise ProtocolError("Single-signal adapter only accepts Noul")
            signature = digest(row)
            if signature not in owners:
                owners[signature] = key
                unique[key] = row
        response = self.backend.decide(state, unique, critical=critical)
        if set(response) != set(unique):
            raise ProtocolError("Single-signal backend dropped a decision")
        return {key: copy.deepcopy(response[owners[digest(row)]]) for key, row in questions.items()}


class SingleSignalSelector(ControlledSelector):
    """Use mode=current for state adaptation; initial reuses the original pointwise scores.

    This core is not a complete shared-first-hop runner. It must get a new instance per task.
    LocalSearchJev's per-candidate state isolation is part of this protocol.
    """
    def __init__(self, backend, runtime, mode):
        if mode not in ("current", "initial"):
            raise ValueError("mode must be current or initial")
        super().__init__(OnePredicate(backend), runtime, "state_relevance")
        self.weights = (.5, .3, .2, 0., 0.)
        self.mode, self.initial_candidates, self.initial_scores, self.task_key = mode, None, None, None
        self.initial_semantic = None

    def restore_initial(self, memory, candidates, scores):
        """Caller must first verify the checkpoint's model/source/input fingerprints."""
        if self.initial_scores is not None:
            raise ProtocolError("Initial scores already installed")
        if len({row.doc_id for row in candidates})!=len(candidates) or set(scores)!={row.doc_id for row in candidates}:
            raise ProtocolError("Initial candidate/score ids differ")
        if any(type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1 for value in scores.values()):
            raise ProtocolError("Invalid initial pointwise scores")
        self.task_key=digest({"question":memory.question,"plan":memory.plan.model_dump()})
        self.initial_semantic=digest({"view":semantic_view(memory,self.runtime),"target":memory.target()})
        self.initial_candidates={row.doc_id:row.model_dump() for row in candidates}
        self.initial_scores=dict(scores)

    def score(self, memory, candidates, relevance=False):
        key = digest({"question": memory.question, "plan": memory.plan.model_dump()})
        if self.task_key is not None and key != self.task_key:
            raise ProtocolError("Do not reuse a selector across different tasks")
        self.task_key = key
        semantic = digest({"view":semantic_view(memory,self.runtime),"target":memory.target()})
        if len({row.doc_id for row in candidates}) != len(candidates):
            raise ProtocolError("Duplicate candidates")
        if self.initial_scores is not None and (self.mode == "initial" or semantic == self.initial_semantic):
            for row in candidates:
                if self.initial_candidates.get(row.doc_id) != row.model_dump():
                    raise ProtocolError("Frozen pointwise candidate changed")
            self.runtime.event("phase4_initial_scores_reused", candidate_n=len(candidates),
                               extra_native_examples=0, reason="initial_control" if self.mode=="initial" else "unchanged_semantic_state")
            return {row.doc_id: self.initial_scores[row.doc_id] for row in candidates}
        scores = super().score(memory, candidates, relevance)
        if self.initial_scores is None:
            self.initial_candidates = {row.doc_id: row.model_dump() for row in candidates}
            self.initial_scores = dict(scores)
            self.initial_semantic = semantic
        return scores
