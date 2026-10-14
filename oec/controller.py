from pydantic import Field
from .graph import (add_extraction, apply_support, canonicalize_plan, closure, compact, complete,
                    frontier, install_plan, invalidate_goal, resolved)
from .runtime import BudgetExceeded, ProtocolError
from .types import Draft, Extraction, Plan, Query, RepairRecord, Result, State, Status, Strict


class Priority(Strict):
    goal_id: str = Field(min_length=1)


class Controller:
    def __init__(self, llm, decisions, retriever, runtime, settings):
        self.llm, self.decisions, self.retriever = llm, decisions, retriever
        self.runtime, self.settings = runtime, settings

    def choose_goal(self, state):
        options = frontier(state)
        if not options:
            raise ProtocolError("No frontier in an incomplete graph")
        if not self.settings.gap_priority:
            return options[0].id
        goal_id, confidence = self.decisions.priority(state)
        if goal_id == "DEFER" or confidence < 0.35:
            result = self.llm.ask("priority", compact(state) |
                                  {"frontier_ids": [goal.id for goal in options]}, Priority)
            goal_id = result.goal_id
            self.runtime.event("priority_fallback", selected_goal=goal_id, confidence=confidence)
        if goal_id not in {goal.id for goal in options}:
            raise ProtocolError("Priority selected a blocked or unknown goal")
        return goal_id

    def draft(self, state):
        for attempt in range(2):
            context = compact(state)
            if attempt:
                context = context | {"supported_claim_ids": self.supported_claim_ids(state),
                                     "required_goal_ids": sorted(goal.id for goal in state.goals.values()
                                                                 if goal.required and resolved(state, goal.id)),
                                     "instruction": "Use only these supported claim ids. Cover every required goal."}
            result = self.llm.ask("answer", context, Draft, critical=True)
            normalized = self.normalize_draft(state, result)
            if normalized is not None:
                return normalized
        self.runtime.event("answer_abstain", reason="could not form a supported answer")
        return Draft(text="", claim_ids=[], citation_ids=[], abstain=True)

    def supported_claim_ids(self, state):
        return [claim.id for claim in state.claims.values() if claim.status == Status.SUPPORTED]

    def normalize_draft(self, state, result):
        if result.abstain:
            if result.text:
                raise ProtocolError("An abstention must have empty answer text")
            return result
        supported = [claim_id for claim_id in result.claim_ids
                     if claim_id in state.claims and state.claims[claim_id].status == Status.SUPPORTED]
        dropped = [claim_id for claim_id in result.claim_ids if claim_id not in supported]
        if dropped:
            self.runtime.event("drop_answer_claims", dropped=dropped)
        if not result.text or not supported:
            return None
        closed = closure(state, supported)
        needed_goals = {goal.id for goal in state.goals.values() if goal.required and resolved(state, goal.id)}
        if not needed_goals <= {state.claims[claim_id].goal_id for claim_id in closed}:
            return None
        citations = sorted({evidence_id for claim_id in closed
                            for evidence_id in state.claims[claim_id].evidence_ids})
        if set(result.citation_ids) != set(citations):
            self.runtime.event("fill_answer_citations", model_citations=result.citation_ids,
                               closure_citations=citations)
        return Draft(text=result.text, claim_ids=supported, citation_ids=citations, abstain=False)

    def result(self, state, draft, status, repairs, verified=False):
        supporting_facts = sorted({(state.evidence[eid].title, state.evidence[eid].sentence_id)
                                   for eid in draft.citation_ids if eid in state.evidence})
        result = Result(task_id=state.task_id, answer=draft.text,
                        supporting_facts=supporting_facts, runtime_valid=True,
                        verified=verified, status=status, repairs=repairs,
                        usage=dict(self.runtime.used) | dict(self.runtime.tokens))
        self.runtime.event("finish", result=result.model_dump())
        self.runtime.snapshot(state)
        return result

    def accepted_extraction(self, state, document, extraction):
        """Keep claims the graph can commit. Illegal rows are logged and skipped."""
        kept = []
        for proposal in extraction.claims:
            reason = self.claim_problem(state, document, proposal)
            if reason:
                self.runtime.event("reject_claim", reason=reason, goal_id=proposal.goal_id,
                                   depends_on=proposal.depends_on, sentence_ids=proposal.sentence_ids,
                                   value=proposal.value)
                continue
            kept.append(proposal)
        return Extraction(claims=kept)

    def claim_problem(self, state, document, proposal):
        if proposal.goal_id not in state.goals:
            return "unknown goal"
        if any(not 0 <= sentence_id < len(document.sentences) for sentence_id in proposal.sentence_ids):
            return "sentence id outside the fetched document"
        for claim_id in proposal.depends_on:
            claim = state.claims.get(claim_id)
            if claim is None or claim.status != Status.SUPPORTED:
                return "depends on an unsupported claim"
        parent_goals = {state.claims[claim_id].goal_id for claim_id in proposal.depends_on}
        if not set(state.goals[proposal.goal_id].depends_on) <= parent_goals:
            return "dependent slot lacks its parent binding"
        return None

    def begin_repair(self, state, failed, reason):
        target = self.decisions.repair_target(state, failed)
        if target == "DEFER":
            return False
        if target not in failed:
            raise ProtocolError("Repair selected a non-failing claim")
        goal_id = state.claims[target].goal_id
        disputed = {cid: claim.text for cid, claim in state.claims.items() if claim.goal_id == goal_id}
        affected = invalidate_goal(state, goal_id)
        state.repair_records.append(RepairRecord(target_claim_id=target, target_goal_id=goal_id,
            invalidated_ids=affected, disputed_claims=disputed, reason=reason))
        self.runtime.event("repair", target_claim=target, target_goal=goal_id,
                           invalidated_claims=affected, retained_evidence=sorted(state.evidence), reason=reason)
        self.runtime.snapshot(state)
        return True

    def run(self, task):
        state = State(task_id=task.id, question=task.question)
        self.state = state
        repairs, in_repair, rewrite_attempted = 0, False, False
        pool, last_draft = {}, Draft(text="", claim_ids=[], citation_ids=[], abstain=True)
        plan = self.llm.ask("plan", {"question": task.question}, Plan)
        canonical = canonicalize_plan(plan)
        if [goal.id for goal in plan.goals] != [goal.id for goal in canonical.goals]:
            self.runtime.event("canonicalize_plan",
                               before=[goal.id for goal in plan.goals],
                               after=[goal.id for goal in canonical.goals])
        install_plan(state, canonical)
        self.runtime.snapshot(state)
        try:
            while True:
                if complete(state):
                    last_draft = self.draft(state)
                    if last_draft.abstain:
                        return self.result(state, last_draft, "ABSTAINED", repairs)
                    probabilities, failed, consistency = self.decisions.audit(state, last_draft)
                    self.runtime.event("audit", probabilities=probabilities, failed=sorted(failed),
                                       answer_consistency=consistency, draft=last_draft.model_dump())
                    if not failed and consistency >= self.settings.answer_threshold:
                        return self.result(state, last_draft, "VERIFIED", repairs, True)
                    if self.settings.method != "full":
                        return self.result(state, last_draft, "ANSWER_UNVERIFIED", repairs)
                    if not failed:
                        # Correct evidence with a miswritten answer needs rewriting, not new search.
                        if rewrite_attempted:
                            return self.result(state, last_draft, "REWRITE_FAILED", repairs)
                        rewrite_attempted = True
                        self.runtime.event("rewrite_answer", reason="Evidence passed; answer text failed")
                        continue
                    if repairs >= self.settings.max_repairs:
                        return self.result(state, last_draft, "REPAIR_LIMIT", repairs)
                    if not self.begin_repair(state, failed, "final_claim_verification_failed"):
                        return self.result(state, last_draft, "REPAIR_DEFERRED", repairs)
                    repairs += 1
                    in_repair, rewrite_attempted = True, False
                if not self.runtime.available("search", in_repair) or not self.runtime.available("read", in_repair):
                    if not last_draft.text:
                        last_draft = self.draft(state)
                    return self.result(state, last_draft, "BUDGET_EXHAUSTED_UNVERIFIED", repairs)
                goal_id = self.choose_goal(state)
                query = self.llm.ask("query", compact(state) |
                    {"target_goal_id": goal_id, "past_queries": state.queries,
                     "repair": in_repair}, Query).query.strip()
                if not query or "<query>" in query or "</query>" in query:
                    raise ProtocolError("Query contains an empty string or protocol tags")
                if query in state.queries:
                    self.runtime.event("duplicate_query", query=query)
                    return self.result(state, last_draft, "REPEATED_QUERY_UNVERIFIED", repairs)
                state.queries.append(query)
                hits = self.runtime.invoke("search", {"query": query, "top_n": self.settings.top_n},
                    lambda: [candidate.model_dump() for candidate in self.retriever.search(query, self.settings.top_n)],
                    critical=in_repair)
                from .types import Candidate, Document
                pool = {}  # Active Top-N for this query; prior retrieval sets remain in the event log.
                for raw in hits:
                    candidate = Candidate.model_validate(raw)
                    if candidate.doc_id not in state.shown_doc_ids:
                        pool[candidate.doc_id] = candidate
                pool = {doc_id: candidate for doc_id, candidate in pool.items()
                        if doc_id not in state.shown_doc_ids}
                static_scores = None
                for _ in range(self.settings.reads_per_search):
                    if complete(state) or not pool or not self.runtime.available("read", in_repair):
                        break
                    goal_id = self.choose_goal(state)
                    candidates = list(pool.values())
                    if self.settings.method == "rank":
                        scores = {item.doc_id: item.retrieval_score for item in candidates}
                    elif static_scores is not None:
                        scores = static_scores
                    else:
                        scores = self.decisions.utility(state, candidates, goal_id,
                            relevance_only=self.settings.method == "relevance")
                        if self.settings.method in ("utility_static", "relevance"):
                            static_scores = scores
                    selected_id = min(pool, key=lambda doc_id: (-scores[doc_id], doc_id))
                    self.runtime.event("select_document", doc_id=selected_id,
                                       state_revision=state.revision, target_goal=goal_id, score=scores[selected_id])
                    document = Document.model_validate(self.runtime.invoke("read", {"doc_id": selected_id},
                        lambda: self.retriever.fetch(selected_id).model_dump(), critical=in_repair))
                    if document.id != selected_id:
                        raise ProtocolError("Fetched document id does not match the selected hit")
                    state.shown_doc_ids.append(selected_id)
                    del pool[selected_id]
                    extraction = self.llm.ask("extract", compact(state) |
                        {"document": document.model_dump(), "sentence_index_base": 0,
                         "supported_claim_ids": [claim.id for claim in state.claims.values()
                                                 if claim.status == Status.SUPPORTED]}, Extraction)
                    extraction = self.accepted_extraction(state, document, extraction)
                    claim_ids = add_extraction(state, document, extraction)
                    probabilities = self.decisions.verify_claims(state, claim_ids)
                    apply_support(state, claim_ids, probabilities, self.settings.support_threshold)
                    self.runtime.snapshot(state)
                    conflicts = {cid for cid, claim in state.claims.items() if claim.status == Status.CONFLICTED}
                    if conflicts and self.settings.method == "full":
                        if repairs >= self.settings.max_repairs:
                            return self.result(state, last_draft, "REPAIR_LIMIT", repairs)
                        if not self.begin_repair(state, conflicts, "conflicting_supported_values"):
                            return self.result(state, last_draft, "REPAIR_DEFERRED", repairs)
                        repairs += 1
                        in_repair, rewrite_attempted = True, False
                        break
        except BudgetExceeded as error:
            self.runtime.event("budget_exceeded", stage=str(error))
            return self.result(state, last_draft, "BUDGET_EXHAUSTED_UNVERIFIED", repairs)
