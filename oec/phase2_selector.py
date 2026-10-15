"""C1 target choice on top of the fixed-candidate dynamic selector."""
import copy
from .phase1c import Selector
from .frontier_scheduler import choose_target


class ScheduledSelector(Selector):
    def __init__(self, backend, runtime, policy):
        super().__init__(backend, runtime)
        self.policy = policy

    def score(self, memory, candidates, relevance=False):
        if relevance:
            raise ValueError("This experiment fixes the selector to utility_dynamic")
        semantic = memory.view(self.runtime)
        for key in ("budget", "state_revision", "read_document_refs", "policy", "method"):
            semantic.pop(key, None)
        goals = [goal.model_dump() for goal in memory.plan.goals]
        selected = choose_target(goals, memory.unresolved, semantic,
                                 self.backend, self.runtime, self.policy, matched_shadow=True)
        descriptions = {goal["id"]: goal["description"] for goal in goals}
        description = descriptions.get(selected, "Resolve remaining answer ambiguity with useful evidence")
        proxy = copy.copy(memory)
        proxy.target = lambda: (selected, description)
        proxy.view = lambda runtime: copy.deepcopy(semantic)
        return super().score(proxy, candidates, False)
