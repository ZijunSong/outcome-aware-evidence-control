import math
import os
from typing import Protocol
from .graph import closure, failed_roots, frontier, resolved, terms
from .runtime import ProtocolError, retry_json
from .types import Candidate, Draft, State, Status


def validate_answers(payload: dict, response: dict) -> dict:
    answers = response.get("answers")
    questions = payload["questions"]
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ProtocolError("Typed response question ids do not match the request")
    for key, question in questions.items():
        answer = answers[key]
        if not isinstance(answer, dict) or answer.get("type") != question["type"]:
            raise ProtocolError("Typed response has the wrong answer type")
        if question["type"] == "noul":
            value = answer.get("noul")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ProtocolError("Noul must be a finite probability in [0, 1]")
        elif question["type"] == "choice":
            probabilities = answer.get("probabilities")
            if not isinstance(probabilities, dict) or set(probabilities) != set(question["criteria"]):
                raise ProtocolError("Choice probabilities contain illegal options")
            if any(isinstance(p, bool) or not isinstance(p, (int, float))
                   or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
                raise ProtocolError("Invalid choice probability")
            if abs(sum(probabilities.values()) - 1) > 1e-3:
                raise ProtocolError("Choice probabilities do not sum to one")
            if answer.get("choice") not in probabilities:
                raise ProtocolError("Illegal selected choice")
            if probabilities[answer["choice"]] < max(probabilities.values()) - 1e-6:
                raise ProtocolError("Choice is not a maximum-probability option")
            confidence = answer.get("confidence")
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                raise ProtocolError("Invalid choice confidence")
        else:
            raise ProtocolError("Reference client supports Choice and Noul only")
    return answers


class DecisionBackend(Protocol):
    def decide(self, state: dict, questions: dict, critical: bool = False) -> dict: ...


class TypesafeJev:
    def __init__(self, runtime, model="jev-latest"):
        self.runtime = runtime
        self.model = model
        self.api_key = os.environ["TYPESAFE_API_KEY"]

    def decide(self, state, questions, critical=False):
        payload = {"model": self.model, "state": state, "questions": questions}
        response = retry_json(self.runtime, "decision",
                              "https://api.typesafe.ai/v1/systemone",
                              payload, self.api_key, critical, self.model)
        return validate_answers(payload, response)


class Decisions:
    def __init__(self, backend: DecisionBackend, runtime, settings):
        self.backend, self.runtime, self.settings = backend, runtime, settings

    def control_view(self, state):
        # SearchJev was trained with a 2048-token prompt. This view stays short so the
        # state itself is not truncated. Verification below still receives source quotes.
        return {
            "question": state.question,
            "revision": state.revision,
            "goals": [{"id": goal.id, "description": goal.description[:180],
                       "depends_on": goal.depends_on, "required": goal.required,
                       "resolved": resolved(state, goal.id)}
                      for goal in state.goals.values()],
            "hypotheses": [{"id": claim.id, "goal_id": claim.goal_id, "value": claim.value[:80],
                            "status": claim.status.value}
                           for claim in state.claims.values() if claim.status != Status.INVALIDATED],
            "repair_records": [{"target_goal_id": record.target_goal_id, "reason": record.reason}
                               for record in state.repair_records],
            "evidence_ids": [evidence.id for evidence in state.evidence.values()],
            "budget": {"used": dict(self.runtime.used)},
        }

    def priority(self, state: State, critical=False) -> tuple[str, float]:
        options = frontier(state)
        if not options:
            raise ProtocolError("Unresolved graph has no actionable frontier")
        criteria = {goal.id: goal.description[:180] for goal in options}
        criteria["DEFER"] = "No listed gap has enough information for a reliable priority decision"
        answer = self.backend.decide(
            self.control_view(state) | {"phase": "priority"},
            {"priority": {"type": "choice", "criteria": criteria,
             "instructions": "Which currently actionable evidence gap is most important to resolve "
                             "for answering the original question? Choose DEFER if unsure."}}, critical)["priority"]
        return answer["choice"], answer["confidence"]

    def utility(self, state: State, candidates: list[Candidate], goal_id: str,
                relevance_only=False, critical=False) -> dict[str, float]:
        scores = {}
        for offset in range(0, len(candidates), self.settings.candidate_batch):
            batch = candidates[offset:offset + self.settings.candidate_batch]
            questions = {}
            for index, candidate in enumerate(batch):
                signals = {"rel": "The candidate is relevant to the original question"} if relevance_only else {
                    "gap": "Reading the candidate is likely to add a fact needed to resolve the target gap",
                    "bridge": "Reading the candidate is likely to reveal a NEW entity, alias, relation or source "
                              "that enables a useful next search for an unresolved goal",
                    "disamb": "Reading the candidate is likely to distinguish existing competing hypotheses "
                              "or resolve a source conflict"}
                for signal, question in signals.items():
                    # Question ids themselves are not seen by Jev: reference doc_id in instructions.
                    questions[f"d{index}_{signal}"] = {
                        "type": "noul",
                        "instructions": {"question": question + "; judge from the visible snippet only, "
                                          "without assuming unseen full-text facts.",
                                         "candidate_doc_id": candidate.doc_id}}
            view = {"question": state.question} if relevance_only else self.control_view(state)
            visible = []
            for item in batch:
                row = item.model_dump()
                row["snippet"] = row["snippet"][:240]
                visible.append(row)
            view |= {"phase": "utility", "target_goal_id": goal_id, "candidates": visible}
            answers = self.backend.decide(view, questions, critical)
            for index, candidate in enumerate(batch):
                if relevance_only:
                    score = answers[f"d{index}_rel"]["noul"]
                else:
                    pg, pb, ph = [answers[f"d{index}_{name}"]["noul"]
                                  for name in ("gap", "bridge", "disamb")]
                    candidate_terms = terms(candidate.snippet)
                    confirmed_evidence = {eid for claim in state.claims.values()
                                          if claim.status == Status.SUPPORTED for eid in claim.evidence_ids}
                    redundancy = max((len(candidate_terms & terms(e.quote)) /
                                      max(1, len(candidate_terms | terms(e.quote)))
                                      for e in state.evidence.values() if e.id in confirmed_evidence), default=0)
                    # Snippet length is a visible cost proxy, not the full document's token cost.
                    length_proxy = min(1, len(candidate.snippet) / 1200)
                    score = (self.settings.weight_gap * pg + self.settings.weight_bridge * pb
                             + self.settings.weight_disambiguate * ph
                             - self.settings.weight_redundancy * redundancy
                             - self.settings.weight_length * length_proxy)
                scores[candidate.doc_id] = score
                self.runtime.event("utility_score", state_revision=state.revision,
                                   doc_id=candidate.doc_id, goal_id=goal_id,
                                   score=score, relevance_only=relevance_only,
                                   raw_signals={k: v for k, v in answers.items() if k.startswith(f"d{index}_")})
        return scores

    def verify_claims(self, state: State, claim_ids: list[str], phase="extract", critical=False):
        probabilities = {}
        for offset in range(0, len(claim_ids), 16):
            batch = claim_ids[offset:offset + 16]
            rows, questions = [], {}
            for claim_id in batch:
                needed = closure(state, [claim_id])
                evidence_ids = {eid for cid in needed for eid in state.claims[cid].evidence_ids}
                rows.append({"claim": {"id": claim_id, "goal_id": state.claims[claim_id].goal_id,
                                       "value": state.claims[claim_id].value[:120],
                                       "text": state.claims[claim_id].text[:240]},
                             "parents": [{"id": cid, "value": state.claims[cid].value[:120]}
                                         for cid in needed if cid != claim_id],
                             "evidence": [{"id": eid, "quote": state.evidence[eid].quote[:280]}
                                          for eid in sorted(evidence_ids)]})
                questions[claim_id] = {"type": "noul", "instructions": {
                    "claim_id": claim_id,
                    "question": "Do the provided source sentences and supported parent claims justify "
                                "this claim, including the exact entity and relationship? Mere name mentions "
                                "or uncertain source statements are insufficient. Check the quoted text "
                                "rather than trusting previous status labels."}}
            answers = self.backend.decide({"question": state.question, "phase": phase,
                                           "claim_rows": rows}, questions, critical)
            probabilities.update({cid: answers[cid]["noul"] for cid in batch})
        return probabilities

    def audit(self, state: State, draft: Draft):
        claim_ids = closure(state, draft.claim_ids)
        probabilities = self.verify_claims(state, claim_ids, "final", True)
        cited = []
        for claim_id in draft.claim_ids:
            claim = state.claims[claim_id]
            cited.append({"id": claim_id, "value": claim.value[:120],
                          "quotes": [state.evidence[eid].quote[:400] for eid in claim.evidence_ids]})
        consistency = self.backend.decide(
            {"question": state.question, "phase": "answer_consistency",
             "draft_text": draft.text[:400], "cited": cited},
            {"consistent": {"type": "noul", "instructions":
             "Does the exact answer text answer the original question and follow from its cited claims, "
             "without adding unsupported facts or changing any entity, relation, date or value?"}}, True)
        failed = {cid for cid, probability in probabilities.items()
                  if probability < self.settings.support_threshold}
        return probabilities, failed, consistency["consistent"]["noul"]

    def repair_target(self, state: State, failed: set[str]):
        roots = failed_roots(state, failed)
        criteria = {cid: state.claims[cid].text[:180] for cid in roots}
        if not criteria:
            raise ProtocolError("No failing root for evidence repair")
        criteria["DEFER"] = "No root can be reliably chosen for repair"
        answer = self.backend.decide(self.control_view(state) | {"phase": "repair", "failed_roots": roots},
            {"repair": {"type": "choice", "criteria": criteria,
                        "instructions": "Choose the earliest unsupported root whose verification is most "
                                        "important to correct the answer; choose only one listed root."}}, True)["repair"]
        return answer["choice"]
