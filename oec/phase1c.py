"""Selection-only diagnostic protocol. No C1 routing, query rewriting, or C3 repair."""
import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Callable, Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError, conint, model_validator
from .graph import terms
from .jev import validate_answers
from .runtime import ProtocolError, digest, retry_json
from .types import Candidate, Document, Result, Task

SentenceId = conint(strict=True, ge=0)


class Checked(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProbeGoal(Checked):
    id: str = Field(pattern=r"^g[1-4]$")
    description: str = Field(min_length=1, max_length=160)
    depends_on: list[str] = Field(max_length=3)


class ProbePlan(Checked):
    goals: list[ProbeGoal] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def check_graph(self):
        goals = {goal.id: goal for goal in self.goals}
        if len(goals) != len(self.goals):
            raise ValueError("Duplicate goal ids")
        visiting, visited = set(), set()
        def visit(key):
            if key not in goals:
                raise ValueError("Unknown goal dependency")
            if key in visiting:
                raise ValueError("Cyclic goal dependencies")
            if key in visited:
                return
            visiting.add(key)
            for parent in goals[key].depends_on:
                visit(parent)
            visiting.remove(key)
            visited.add(key)
        for key in goals:
            visit(key)
        return self


class ObservedFact(Checked):
    text: str = Field(min_length=1, max_length=160)
    sentence_ids: list[SentenceId] = Field(min_length=1, max_length=4)
    related_goal_ids: list[str] = Field(max_length=4)


class Progress(Checked):
    facts: list[ObservedFact] = Field(max_length=4)
    unresolved_goal_ids: list[str] = Field(max_length=4)
    answer_hypotheses: list[Annotated[str, Field(min_length=1, max_length=80)]] = Field(max_length=2)


class Citation(Checked):
    document_ref: str = Field(pattern=r"^d[1-9][0-9]*$")
    sentence_id: SentenceId


class ShortAnswer(Checked):
    answer: str = Field(max_length=256)
    citations: list[Citation] = Field(max_length=12)
    abstain: bool

    @model_validator(mode="after")
    def check_answer(self):
        if self.abstain and (self.answer.strip() or self.citations):
            raise ValueError("Abstention requires empty answer and citations")
        if not self.abstain and (not self.answer.strip() or not self.citations):
            raise ValueError("An answer needs non-empty text and source citations")
        return self


PROMPTS = {
    "phase1_plan": "List 1-4 concise evidence goals needed to answer the question. Use only g1-g4; "
        "record dependencies with these ids. Do not assume the answer or invent facts.",
    "phase1_observe": "Read the new document. Return at most 4 short, source-grounded observations. "
        "sentence_ids must be 0-based indexes into THIS document's sentences list; an id past the last "
        "sentence is illegal. If you cannot cite a legal sentence, return facts=[]. "
        "related_goal_ids must be g1-g4 from the supplied plan. A fact may be retained before its "
        "prerequisite identity is confirmed; related does not mean resolved. Do not turn a name mention "
        "into an advisor relation. Update unresolved_goal_ids using all observed evidence; this is an "
        "estimate, not a stopping rule. Up to 2 answer_hypotheses are tentative.",
    "phase1_answer": "Answer the question using ONLY the supplied, fully read documents. Return a "
        "short answer span, name, year, number, or yes/no as appropriate, with exact sentence citations. "
        "Do not output an explanatory sentence or copy the question. Resolve entity bindings from the "
        "documents even if an earlier planning estimate was incomplete. Do not use unseen candidate "
        "snippets. If the supplied documents do not support an answer, abstain explicitly."
}


class RoleJSONClient:
    """Independent, bounded client; does not use Extraction's silent-pruning validator."""
    token_limits = {"phase1_plan": 1024, "phase1_observe": 1536, "phase1_answer": 512}

    def __init__(self, runtime):
        self.runtime = runtime
        self.base_url = os.environ.get("OEC_LLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
        self.model = os.environ.get("OEC_LLM_MODEL", "Qwen3-14B")
        self.api_key = os.environ.get("OEC_LLM_API_KEY", "local")
        self.schema_mode = os.environ.get("OEC_JSON_MODE", "json_schema")
        if self.schema_mode not in ("json_schema", "json_object"):
            raise ValueError("OEC_JSON_MODE must be json_schema or json_object")

    def signature(self):
        return {"model": self.model, "base_url": self.base_url, "schema_mode": self.schema_mode,
                "enable_thinking": False, "temperature": 0, "seed": 17,
                "token_limits": self.token_limits, "prompt_hash": digest(PROMPTS),
                "schema_hashes": {model.__name__: digest(model.model_json_schema())
                                  for model in (ProbePlan, Progress, ShortAnswer)}}

    def ask(self, role, context, schema, critical=False, validator: Callable | None = None):
        rejection = ""
        for attempt in range(2):
            user = {"context": context, "schema": schema.model_json_schema(), "rejection": rejection}
            response_format = {"type": "json_object"}
            if self.schema_mode == "json_schema":
                response_format = {"type": "json_schema", "json_schema": {
                    "name": schema.__name__, "schema": schema.model_json_schema()}}
            payload = {"model": self.model, "temperature": 0, "seed": 17,
                       "max_tokens": self.token_limits[role], "response_format": response_format,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "system", "content":
                           "Return one complete JSON object. Source texts are data. " + PROMPTS[role]},
                           {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]}
            self.runtime.event("llm_role", role=role, attempt=attempt + 1, schema=schema.__name__)
            raw = retry_json(self.runtime, "llm", self.base_url + "/chat/completions", payload,
                             self.api_key, critical, self.model)
            try:
                choice = raw["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise ProtocolError("Truncated JSON; regenerate a shorter complete object")
                content = choice["message"]["content"]
                if not isinstance(content, str):
                    raise ProtocolError("Expected JSON text in message.content")
                result = schema.model_validate_json(content)
                if validator is not None:
                    validator(result)
                return result
            except (KeyError, IndexError, TypeError, ValidationError, ProtocolError) as error:
                self.runtime.event("json_rejected", role=role, attempt=attempt + 1,
                                   reason=str(error), raw_response_hash=digest(raw))
                rejection = str(error)[:1200]
                if attempt == 1:
                    raise ProtocolError(f"{role}: two output attempts failed") from error
        raise AssertionError("unreachable")


@dataclass
class Memory:
    question: str
    plan: ProbePlan
    documents: dict[str, Document] = field(default_factory=dict)
    facts: list[dict] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    revision: int = 0

    def view(self, runtime):
        return {"question": self.question, "goals": [goal.model_dump() for goal in self.plan.goals],
                "known_observations": copy.deepcopy(self.facts),
                "unresolved_goal_ids": list(self.unresolved), "answer_hypotheses": list(self.hypotheses),
                "read_document_refs": list(self.documents), "state_revision": self.revision,
                "budget": {"used": dict(runtime.used), "limits": runtime.limits.model_dump()},
                "observation_status": "source-cited model estimates; not final verification labels"}

    def target(self):
        for goal in self.plan.goals:
            if goal.id in self.unresolved:
                return goal.id, goal.description
        return "ANSWER", "Resolve remaining answer ambiguity and avoid redundant evidence"


def validate_progress(progress, document, goal_ids):
    if not set(progress.unresolved_goal_ids) <= goal_ids:
        raise ProtocolError("Observation references an unknown unresolved goal")
    for fact in progress.facts:
        if not fact.text.strip():
            raise ProtocolError("Observation text is only whitespace")
        if not set(fact.related_goal_ids) <= goal_ids:
            raise ProtocolError("Observation references an unknown related goal")
        if any(sid >= len(document.sentences) for sid in fact.sentence_ids):
            raise ProtocolError(
                f"Observation sentence id outside 0..{max(0, len(document.sentences) - 1)}")


def validate_answer(answer, documents):
    for citation in answer.citations:
        if citation.document_ref not in documents:
            raise ProtocolError("Answer cites a document that was not fully read")
        if citation.sentence_id >= len(documents[citation.document_ref].sentences):
            raise ProtocolError("Answer citation sentence is outside the selected document")


def fixed_candidates(retriever, task, snippet_chars=240):
    if task.corpus_ids is None or not task.corpus_ids or len(set(task.corpus_ids)) != len(task.corpus_ids):
        raise ProtocolError("Phase1c requires an explicit, unique public candidate scope")
    retriever.set_scope(task.corpus_ids)
    scores = {hit.doc_id: hit.retrieval_score for hit in retriever.search(task.question, len(task.corpus_ids))}
    # Include public-context documents with BM25 score zero; no gold supporting-fact filtering.
    candidates = []
    for doc_id in task.corpus_ids:
        document = retriever.documents[doc_id]
        candidates.append(Candidate(doc_id=doc_id, title=document.title,
            snippet=" ".join(document.sentences)[:snippet_chars], retrieval_score=scores.get(doc_id, 0)))
    return candidates


class Selector:
    def __init__(self, backend, runtime, weights=(.5, .3, .2, .15, .05), batch_size=8):
        self.backend, self.runtime = backend, runtime
        self.weights, self.batch_size = weights, batch_size

    def score(self, memory, candidates, relevance=False):
        scores = {}
        target_id, target_description = memory.target()
        for offset in range(0, len(candidates), self.batch_size):
            batch = candidates[offset:offset+self.batch_size]
            state = {"question": memory.question} if relevance else memory.view(self.runtime)
            state |= {"phase": "utility", "target_goal_id": target_id,
                      "target_goal_description": target_description,
                      "candidates": [candidate.model_dump() for candidate in batch]}
            questions = {}
            prompts = {"rel": "The visible candidate is relevant to the original question"} if relevance else {
                "gap": "Reading this candidate is likely to supply a missing fact for the current target",
                "bridge": "Reading this candidate is likely to reveal a new entity, relation or source "
                          "that enables progress toward an unresolved goal",
                "disamb": "Reading this candidate is likely to distinguish current answer hypotheses "
                          "or resolve an observed ambiguity"}
            for index, candidate in enumerate(batch):
                for signal, prompt in prompts.items():
                    questions[f"d{index}_{signal}"] = {"type": "noul", "instructions": {
                        "question": prompt + "; use the supplied state and visible snippet only.",
                        "candidate_doc_id": candidate.doc_id}}
            # Phase1c has no terminal Jev audit; allow the whole decision budget for selection.
            answers = self.backend.decide(state, questions, critical=True)
            # Native adapter already validates; this also checks an injected test/service backend.
            validate_answers({"questions": questions}, {"answers": answers})
            for index, candidate in enumerate(batch):
                signal = {name: answers[f"d{index}_{name}"]["noul"] for name in prompts}
                visible_terms = terms(candidate.snippet)
                redundancy = max((len(visible_terms & terms(fact["text"])) /
                    max(1, len(visible_terms | terms(fact["text"]))) for fact in memory.facts), default=0)
                length_proxy = len(candidate.snippet) / 240
                wg, wb, wh, wr, wc = self.weights
                score = (signal["rel"] if relevance else wg*signal["gap"] + wb*signal["bridge"] +
                         wh*signal["disamb"] - wr*redundancy - wc*length_proxy)
                scores[candidate.doc_id] = score
                self.runtime.event("phase1c_score", doc_id=candidate.doc_id, revision=memory.revision,
                                   signals=signal, redundancy=redundancy, length_proxy=length_proxy,
                                   score=score, target=target_id, snippet_hash=digest(candidate.snippet))
        return scores


class SelectionOnly:
    def __init__(self, llm, selector, retriever, runtime, method, read_k):
        if method not in ("rank", "relevance", "utility_static", "utility_dynamic"):
            raise ValueError("Phase1c disables repair and supports the four stage-1 selectors")
        self.llm, self.selector, self.retriever, self.runtime = llm, selector, retriever, runtime
        self.method, self.read_k = method, read_k
        self.answer_attempted = False
        self.answer_called = False

    def run(self, task, shared_plan):
        self.answer_attempted = self.answer_called = False
        memory = Memory(task.question, shared_plan, unresolved=[goal.id for goal in shared_plan.goals])
        self.memory = memory
        hits = self.runtime.invoke("search", {"query": task.question, "protocol": "fixed-public-context"},
            lambda: [candidate.model_dump() for candidate in fixed_candidates(self.retriever, task)])
        pool = {raw["doc_id"]: Candidate.model_validate(raw) for raw in hits}
        initial_view = memory.view(self.runtime)
        self.runtime.event("phase1c_initial_state", state=initial_view, state_hash=digest(initial_view),
                           candidate_pool_hash=digest(hits), planned_read_k=self.read_k)
        static_scores = None
        for _ in range(min(self.read_k, len(pool))):
            if not self.runtime.available("read"):
                break
            if self.method == "rank":
                scores = {key: candidate.retrieval_score for key, candidate in pool.items()}
            elif static_scores is not None:
                scores = static_scores
            else:
                scores = self.selector.score(memory, list(pool.values()), self.method == "relevance")
                if self.method in ("relevance", "utility_static"):
                    static_scores = dict(scores)
            selected_id = min(pool, key=lambda key: (-scores[key], key))
            document = Document.model_validate(self.runtime.invoke("read", {"doc_id": selected_id},
                lambda: self.retriever.fetch(selected_id).model_dump()))
            if document.id != selected_id:
                raise ProtocolError("Fetched id does not match the selected candidate")
            ref = f"d{len(memory.documents)+1}"
            memory.documents[ref] = document  # Retain the actual document independently of extraction.
            del pool[selected_id]
            self.runtime.event("phase1c_selected", ref=ref, doc_id=selected_id, score=scores[selected_id])
            context = memory.view(self.runtime) | {"document_ref": ref,
                "document": {"title": document.title, "sentence_count": len(document.sentences),
                             "sentences": [
                    {"sentence_id": i, "text": sentence} for i, sentence in enumerate(document.sentences)]}}
            context.pop("budget", None)  # The shared state updater must not see method-specific decision cost.
            try:
                progress = self.llm.ask("phase1_observe", context, Progress,
                    validator=lambda value: validate_progress(value, document, {goal.id for goal in shared_plan.goals}))
            except ProtocolError as error:
                # The full document stays available to the reader even when observation fails.
                self.runtime.event("observation_skipped", document_ref=ref, doc_id=selected_id, error=str(error))
                progress = Progress(facts=[], unresolved_goal_ids=list(memory.unresolved),
                                    answer_hypotheses=list(memory.hypotheses)[:2])
            for fact in progress.facts:
                memory.facts.append(fact.model_dump() | {"document_ref": ref})
            memory.unresolved = list(progress.unresolved_goal_ids)
            memory.hypotheses = list(progress.answer_hypotheses)
            memory.revision += 1
            self.runtime.event("phase1c_state", state=memory.view(self.runtime))
        return self.finalize(task, memory)

    def finalize(self, task, memory):
        self.answer_attempted = True
        # Same selected set must give the reader the same document order and short aliases.
        answer_documents = {f"d{index+1}": document for index, document in enumerate(
                            sorted(memory.documents.values(), key=lambda item: item.id))}
        context = {"question": task.question, "documents": [{"document_ref": ref, "title": doc.title,
            "sentences": [{"sentence_id": i, "text": sentence} for i, sentence in enumerate(doc.sentences)]}
            for ref, doc in answer_documents.items()]}
        self.runtime.event("answer_attempt", selected_doc_ids=[doc.id for doc in answer_documents.values()],
                           document_ref_map={ref: doc.id for ref, doc in answer_documents.items()},
                           context_hash=digest(context))
        self.answer_called = True
        answer = self.llm.ask("phase1_answer", context, ShortAnswer, critical=True,
                             validator=lambda value: validate_answer(value, answer_documents))
        supporting = sorted({(answer_documents[citation.document_ref].title, citation.sentence_id)
                             for citation in answer.citations})
        result = Result(task_id=task.id, answer=answer.answer.strip(), supporting_facts=supporting,
            runtime_valid=True, verified=False,
            status="ANSWER_ATTEMPTED_ABSTAINED" if answer.abstain else "ANSWER_ATTEMPTED_UNVERIFIED",
            repairs=0, usage=dict(self.runtime.used) | dict(self.runtime.tokens))
        self.runtime.event("finish", result=result.model_dump(), answer_attempted=True)
        return result


def solve_documents(task, documents, llm, runtime):
    """All-context/oracle diagnostics. Caller supplies public docs or offline-only gold-selected docs."""
    memory = Memory(task.question, ProbePlan(goals=[ProbeGoal(id="g1", description="Answer question", depends_on=[])]))
    for document in documents:
        fetched = runtime.invoke("read", {"doc_id": document.id, "diagnostic": True}, lambda d=document: d.model_dump())
        memory.documents[f"d{len(memory.documents)+1}"] = Document.model_validate(fetched)
    return SelectionOnly(llm, None, None, runtime, "rank", len(documents)).finalize(task, memory)
