import re
from .runtime import ProtocolError
from .types import Claim, Document, Evidence, Extraction, Goal, Plan, State, Status


def normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def canonicalize_plan(plan: Plan) -> Plan:
    """Rewrite illegal or over-long goal ids to g1..g8. Descriptions stay unchanged."""
    selected = list(plan.goals[:8])
    mapping, used = {}, set()
    for goal in selected:
        new_id = goal.id
        legal = new_id != "DEFER" and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", new_id)
        if not legal or new_id in used:
            number = 1
            while f"g{number}" in used:
                number += 1
            new_id = f"g{number}"
        used.add(new_id)
        mapping[goal.id] = new_id
    goals = []
    for goal in selected:
        missing = [parent for parent in goal.depends_on if parent not in mapping]
        if missing:
            raise ProtocolError("Unknown goal dependency: " + missing[0])
        goals.append(Goal(id=mapping[goal.id], description=goal.description,
                          depends_on=[mapping[parent] for parent in goal.depends_on],
                          required=goal.required))
    return Plan(goals=goals)


def install_plan(state: State, plan: Plan):
    plan = canonicalize_plan(plan)
    goals = {goal.id: goal for goal in plan.goals}
    if len(goals) != len(plan.goals):
        raise ProtocolError("Duplicate goal id")
    if not any(goal.required for goal in plan.goals):
        raise ProtocolError("Plan has no required answer goal")
    if any(goal.id == "DEFER" or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", goal.id)
           for goal in plan.goals):
        raise ProtocolError("Invalid or reserved goal id")
    active, done = set(), set()

    def visit(goal_id):
        if goal_id not in goals:
            raise ProtocolError("Unknown goal dependency: " + goal_id)
        if goal_id in active:
            raise ProtocolError("Goal dependency cycle")
        if goal_id in done:
            return
        active.add(goal_id)
        for parent in goals[goal_id].depends_on:
            visit(parent)
        active.remove(goal_id)
        done.add(goal_id)

    for goal_id in goals:
        visit(goal_id)
    state.goals = goals
    state.revision += 1


def supported(state: State, goal_id: str):
    return [claim for claim in state.claims.values()
            if claim.goal_id == goal_id and claim.status == Status.SUPPORTED]


def resolved(state: State, goal_id: str) -> bool:
    return bool(supported(state, goal_id))


def frontier(state: State):
    return [goal for goal in state.goals.values()
            if not resolved(state, goal.id)
            and all(resolved(state, parent) for parent in goal.depends_on)]


def complete(state: State) -> bool:
    return all(resolved(state, goal.id) for goal in state.goals.values() if goal.required)


def closure(state: State, claim_ids: list[str]) -> list[str]:
    seen, order = set(), []

    def visit(claim_id):
        if claim_id not in state.claims:
            raise ProtocolError("Unknown claim id: " + claim_id)
        if claim_id in seen:
            return
        seen.add(claim_id)
        for parent in state.claims[claim_id].depends_on:
            visit(parent)
        order.append(claim_id)

    for claim_id in claim_ids:
        visit(claim_id)
    return order


def add_extraction(state: State, document: Document, extraction: Extraction) -> list[str]:
    staged = state.model_copy(deep=True)
    additions = _stage_extraction(staged, document, extraction)
    state.claims = staged.claims
    state.evidence = staged.evidence
    state.revision = staged.revision
    return additions


def _stage_extraction(state: State, document: Document, extraction: Extraction) -> list[str]:
    additions = []
    for proposal in extraction.claims:
        if proposal.goal_id not in state.goals:
            raise ProtocolError("Extraction references unknown goal")
        parent_claims = []
        for claim_id in proposal.depends_on:
            if claim_id not in state.claims or state.claims[claim_id].status != Status.SUPPORTED:
                raise ProtocolError("Extraction depends on an unsupported claim")
            parent_claims.append(state.claims[claim_id])
        parent_goals = {claim.goal_id for claim in parent_claims}
        if not set(state.goals[proposal.goal_id].depends_on) <= parent_goals:
            raise ProtocolError("A dependent slot lacks its parent binding")
        evidence_ids = []
        for sentence_id in sorted(set(proposal.sentence_ids)):
            if not 0 <= sentence_id < len(document.sentences):
                raise ProtocolError("Sentence id outside the fetched document")
            evidence_id = f"{document.id}:s{sentence_id}"
            state.evidence[evidence_id] = Evidence(
                id=evidence_id, doc_id=document.id, title=document.title,
                sentence_id=sentence_id, quote=document.sentences[sentence_id],
                source=document.source)
            evidence_ids.append(evidence_id)
        claim_id = f"c{len(state.claims) + 1}"
        state.claims[claim_id] = Claim(
            id=claim_id, goal_id=proposal.goal_id, text=proposal.text,
            value=proposal.value, evidence_ids=evidence_ids,
            depends_on=proposal.depends_on, mode=proposal.mode)
        additions.append(claim_id)
    state.revision += 1
    return additions


def apply_support(state: State, claim_ids: list[str], probabilities: dict[str, float], threshold: float):
    for claim_id in claim_ids:
        claim = state.claims[claim_id]
        claim.support_probability = probabilities[claim_id]
        claim.status = Status.SUPPORTED if probabilities[claim_id] >= threshold else Status.CANDIDATE
    for goal_id in state.goals:
        active = [claim for claim in state.claims.values()
                  if claim.goal_id == goal_id and claim.status in (Status.SUPPORTED, Status.CONFLICTED)]
        if len({normalized(claim.value) for claim in active}) > 1:
            for claim in active:
                claim.status = Status.CONFLICTED
    # A conflict or loss of parent support also blocks any dependent binding.
    changed = True
    while changed:
        changed = False
        for claim in state.claims.values():
            if claim.status == Status.SUPPORTED and any(
                    state.claims[parent].status != Status.SUPPORTED for parent in claim.depends_on):
                claim.status = Status.CANDIDATE
                changed = True
    state.revision += 1


def invalidate_goal(state: State, goal_id: str) -> list[str]:
    affected = {claim.id for claim in state.claims.values() if claim.goal_id == goal_id}
    changed = True
    while changed:
        old = len(affected)
        affected.update(claim.id for claim in state.claims.values()
                        if set(claim.depends_on) & affected)
        changed = len(affected) != old
    for claim_id in affected:
        state.claims[claim_id].status = Status.INVALIDATED
    state.revision += 1
    return sorted(affected)


def failed_roots(state: State, failed: set[str]) -> list[str]:
    # Remove a failing descendant if some failing ancestor already explains it.
    return [claim_id for claim_id in sorted(failed)
            if not (set(closure(state, state.claims[claim_id].depends_on)) & failed)]


def compact(state: State) -> dict:
    return {"question": state.question, "revision": state.revision,
            "goals": [goal.model_dump() | {"resolved": resolved(state, goal.id)}
                      for goal in state.goals.values()],
            "hypotheses": [claim.model_dump(mode="json") for claim in state.claims.values()
                           if claim.status != Status.INVALIDATED],
            "repair_records": [record.model_dump() for record in state.repair_records],
            "evidence": [item.model_dump() for item in state.evidence.values()]}


def terms(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))
