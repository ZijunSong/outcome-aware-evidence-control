"""Provided-context cache and pending retry bookkeeping; never promotes a fact to support."""
from dataclasses import dataclass
from typing import Callable
from .runtime import ProtocolError, digest
from .types import Candidate, Document


@dataclass(frozen=True)
class ParentBinding:
    goal_id: str
    claim_id: str
    value: str
    evidence_version: str
    supported: bool = True


@dataclass(frozen=True)
class PendingFact:
    id: str
    doc_id: str
    goal_id: str
    sentence_ids: tuple[int, ...]
    proposal_text: str


@dataclass(frozen=True)
class RetryAttempt:
    pending_id: str
    parent_bindings: tuple[ParentBinding, ...]

    @property
    def key(self):
        return digest({"pending_id": self.pending_id, "parents": [vars(value) for value in self.parent_bindings]})


def check_dependencies(dependencies):
    active, done = set(), set()
    def visit(key):
        if key not in dependencies:
            raise ProtocolError("Unknown parent goal")
        if key in active:
            raise ProtocolError("Cyclic goal graph")
        if key in done:
            return
        active.add(key)
        for parent in dependencies[key]:
            visit(parent)
        active.remove(key)
        done.add(key)
    for key in dependencies:
        visit(key)


class EvidenceCache:
    def __init__(self, scope_ids, dependencies, retriever_fingerprint):
        if not scope_ids or len(set(scope_ids)) != len(scope_ids):
            raise ProtocolError("A cache needs a unique explicit public document scope")
        self.scope = frozenset(scope_ids)
        self.dependencies = {key: tuple(value) for key, value in dependencies.items()}
        check_dependencies(self.dependencies)
        self.retriever_fingerprint = retriever_fingerprint
        self._documents, self._queries, self._candidates, self._ranks = {}, {}, {}, {}
        self.pending, self.attempts = {}, {}

    def put_document(self, document):
        document = Document.model_validate(document)
        if document.id not in self.scope:
            raise ProtocolError("Document outside the task's public scope")
        payload = document.model_dump()
        if document.id in self._documents and self._documents[document.id] != payload:
            raise ProtocolError("Immutable document id changed content")
        self._documents[document.id] = payload

    def fetch_once(self, doc_id, runtime, fetch: Callable, critical=False):
        if doc_id not in self.scope:
            raise ProtocolError("Fetch outside public scope")
        if doc_id not in self._documents:
            raw = runtime.invoke("read", {"doc_id": doc_id}, lambda: fetch(doc_id).model_dump(), critical=critical)
            document = Document.model_validate(raw)
            if document.id != doc_id:
                raise ProtocolError("Fetch returned the wrong document id")
            self.put_document(document)
        else:
            runtime.event("document_cache_hit", doc_id=doc_id)
        return Document.model_validate(self._documents[doc_id])

    def merge_candidates(self, candidates, source_key):
        values = [Candidate.model_validate(value) for value in candidates]
        if len({value.doc_id for value in values}) != len(values):
            raise ProtocolError("Duplicate id in a retrieved list")
        if any(value.doc_id not in self.scope for value in values):
            raise ProtocolError("Retrieved candidate outside public scope")
        for rank, candidate in enumerate(values, 1):
            self._candidates[candidate.doc_id] = candidate.model_dump()
            self._ranks.setdefault(candidate.doc_id, {})[source_key] = rank

    def retrieve_once(self, query, top_n, runtime, search: Callable, critical=False):
        query = query.strip()
        if not query or top_n < 1:
            raise ProtocolError("Empty query or invalid top_n")
        key = digest({"query": query, "top_n": top_n, "scope": sorted(self.scope),
                      "retriever": self.retriever_fingerprint})
        if key not in self._queries:
            raw = runtime.invoke("search", {"query": query, "top_n": top_n},
                                 lambda: [value.model_dump() for value in search(query, top_n)], critical=critical)
            self.merge_candidates(raw, key)
            self._queries[key] = raw
        else:
            runtime.event("query_cache_hit", query_key=key)
        return [Candidate.model_validate(value) for value in self._queries[key]]

    def unread_candidates(self):
        return [Candidate.model_validate(value) for key, value in self._candidates.items()
                if key not in self._documents]

    def rrf_scores(self, constant=60):
        if constant <= 0:
            raise ValueError("RRF constant must be positive")
        return {key: sum(1/(constant+rank) for rank in ranks.values())
                for key, ranks in self._ranks.items() if key not in self._documents}

    def add_pending(self, doc_id, goal_id, sentence_ids, proposal_text):
        if doc_id not in self._documents or goal_id not in self.dependencies or not proposal_text.strip():
            raise ProtocolError("Pending fact requires a read source, known goal and nonempty proposal")
        if any(type(sid) is not int for sid in sentence_ids):
            raise ProtocolError("Sentence ids must be integers, not coerced values")
        ids = tuple(sorted(set(sentence_ids)))
        if not ids or any(type(sid) is not int or not 0 <= sid < len(self._documents[doc_id]["sentences"]) for sid in ids):
            raise ProtocolError("Illegal pending sentence ids")
        key = digest({"doc_id": doc_id, "goal_id": goal_id, "sentence_ids": ids, "text": proposal_text})
        self.pending[key] = PendingFact(key, doc_id, goal_id, ids, proposal_text)
        return key

    def ready_attempts(self, bindings):
        ready = []
        for record in self.pending.values():
            parents = []
            for goal_id in sorted(set(self.dependencies[record.goal_id])):
                options = bindings.get(goal_id, ())
                if len(options) != 1 or not options[0].supported or options[0].goal_id != goal_id:
                    break
                parents.append(options[0])
            else:
                attempt = RetryAttempt(record.id, tuple(parents))
                if attempt.key not in self.attempts:
                    ready.append(attempt)
        return ready

    def begin_attempt(self, attempt):
        if attempt.pending_id not in self.pending:
            raise ProtocolError("Unknown pending fact")
        if attempt.key in self.attempts:
            return False
        self.attempts[attempt.key] = "STARTED"
        return True

    def finish_attempt(self, attempt, outcome):
        if self.attempts.get(attempt.key) != "STARTED" or outcome not in ("ACCEPTED", "REJECTED", "DEFERRED", "PROTOCOL_FAILED"):
            raise ProtocolError("Illegal pending attempt transition")
        self.attempts[attempt.key] = outcome
