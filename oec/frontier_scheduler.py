"""Scheduling over estimated goals. This is not factual support verification."""
from .jev import validate_answers
from .runtime import ProtocolError
from .cache_core import check_dependencies


def estimated_frontier(goals, unresolved):
    ids = [goal["id"] for goal in goals]
    dependencies = {goal["id"]: tuple(goal["depends_on"]) for goal in goals}
    if len(set(ids)) != len(ids) or not set(unresolved) <= set(ids):
        raise ProtocolError("Duplicate goal or unknown unresolved goal")
    check_dependencies(dependencies)
    return [goal_id for goal_id in ids if goal_id in unresolved and
            not set(dependencies[goal_id]) & set(unresolved)]


def choose_target(goals, unresolved, semantic_state, backend, runtime, policy, matched_shadow=True):
    if policy not in ("fixed", "priority"):
        raise ValueError("policy must be fixed or priority")
    options = estimated_frontier(goals, unresolved)
    if not options:
        if unresolved:
            raise ProtocolError("Unresolved goals have no estimated actionable frontier")
        runtime.event("schedule", policy=policy, options=[], selected="ANSWER", reason="estimated_complete")
        return "ANSWER"
    selected, reason = options[0], "singleton" if len(options) == 1 else "fixed_order"
    if len(options) > 1 and (policy == "priority" or matched_shadow):
        descriptions = {goal["id"]: goal["description"] for goal in goals}
        criteria = {key: descriptions[key] for key in options} | {"DEFER": "No reliable priority choice"}
        questions = {"priority": {"type": "choice", "criteria": criteria,
            "instructions": "Select the actionable evidence goal that is most useful for answering the original question. "
                            "Use only observed information; DEFER if unsure."}}
        view = {key: value for key, value in semantic_state.items()
                if key not in ("budget", "state_revision", "read_document_refs", "policy", "method")}
        view |= {"phase": "frontier_priority", "frontier_ids": options}
        answers = backend.decide(view, questions, critical=False)
        answer = validate_answers({"questions": questions}, {"answers": answers})["priority"]
        proposed = answer["choice"]
        runtime.event("priority_proposal", proposed=proposed, fixed_fallback=options[0],
                      applied=policy == "priority" and proposed != "DEFER")
        if policy == "priority" and proposed != "DEFER":
            selected, reason = proposed, "priority"
        elif policy == "priority":
            reason = "defer_fixed_fallback"
    runtime.event("schedule", policy=policy, options=options, selected=selected, reason=reason,
                  frontier_is_estimated=True)
    return selected
