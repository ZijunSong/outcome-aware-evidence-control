"""Shared first-hop kernel. Second read is not followed by an observation."""
import copy
from .frozen_pool import candidates_from_pool
from .phase1c import Memory, Progress, validate_progress
from .phase4_single import SingleSignalSelector
from .runtime import ProtocolError, digest
from .types import Document, Result


def pick(scores):
    return min(scores, key=lambda key: (-scores[key], key))


def apply_progress(memory, progress, ref):
    for fact in progress.facts:
        memory.facts.append(fact.model_dump() | {"document_ref": ref})
    memory.unresolved = list(progress.unresolved_goal_ids)
    memory.hypotheses = list(progress.answer_hypotheses)
    memory.revision += 1


def observe_first(observer, memory, document, ref, plan):
    context = memory.view(observer.runtime) | {"document_ref": ref,
        "document": {"title": document.title, "sentence_count": len(document.sentences),
                     "sentences": [{"sentence_id": index, "text": sentence}
                                   for index, sentence in enumerate(document.sentences)]}}
    context.pop("budget", None)
    try:
        progress = observer.ask("phase1_observe", context, Progress,
            validator=lambda value: validate_progress(value, document, {goal.id for goal in plan.goals}))
        return progress, None
    except ProtocolError as error:
        if "two output attempts failed" not in str(error) and "bounded_protocol_failure" not in str(error):
            raise
        return None, str(error)


def reader_bundle(task, documents):
    ordered = sorted(documents, key=lambda document: document.id)
    context = {"question": task.question, "documents": [{"document_ref": f"d{index + 1}",
        "title": document.title, "sentences": [{"sentence_id": sid, "text": sentence}
            for sid, sentence in enumerate(document.sentences)]} for index, document in enumerate(ordered)]}
    return {"task_id": task.id, "context": context, "context_hash": digest(context),
            "selected_doc_ids": [document.id for document in ordered],
            "selection_stage_valid": len(documents) == 2}


class SharedFirstHop:
    def __init__(self, backend, observer, retriever, runtime, pool_root, manifest, identity_hash):
        self.backend, self.observer, self.retriever, self.runtime = backend, observer, retriever, runtime
        self.pool_root, self.manifest, self.identity_hash = pool_root, manifest, identity_hash

    def prefix(self, task, plan):
        memory = Memory(task.question, plan, unresolved=[goal.id for goal in plan.goals])
        candidates = candidates_from_pool(self.retriever, task, self.pool_root, self.manifest)
        selector = SingleSignalSelector(self.backend, self.runtime, "current")
        scores = selector.score(memory, candidates)
        first_id = pick(scores)
        document = Document.model_validate(self.runtime.invoke("read", {"doc_id": first_id},
            lambda: self.retriever.fetch(first_id).model_dump()))
        if document.id != first_id:
            raise ProtocolError("Fetched id does not match the selected candidate")
        ref = "d1"
        memory.documents[ref] = document
        self.runtime.event("phase1c_selected", ref=ref, doc_id=first_id, score=scores[first_id])
        updated = copy.deepcopy(memory)
        progress, error = observe_first(self.observer, updated, document, ref, plan)
        if progress is None:
            self.runtime.event("observation_skipped", document_ref=ref, doc_id=first_id, error=error)
            status = "bounded_protocol_failure"
        else:
            apply_progress(updated, progress, ref)
            status = "success"
        self.runtime.event("phase1c_state", state=updated.view(self.runtime))
        checkpoint = {
            "task_id": task.id, "question": task.question, "plan": plan.model_dump(),
            "first_doc_id": first_id, "first_document": document.model_dump(),
            "initial_scores": scores,
            "initial_candidates": [row.model_dump() for row in candidates],
            "initial_view": memory.view(self.runtime),
            "updated_view": updated.view(self.runtime),
            "observation_status": status, "observation_error": error,
            "identity_sha256": self.identity_hash,
            "pool_manifest_sha256": digest({key: self.manifest[key] for key in sorted(self.manifest)
                                            if key != "checkpoint_sha256"}),
        }
        return checkpoint, memory, updated, candidates, scores, document

    def second(self, task, plan, checkpoint, initial_memory, updated_memory, candidates, scores, first, branch):
        mode = "current" if branch == "current_once" else "initial"
        memory = copy.deepcopy(updated_memory if mode == "current" else initial_memory)
        remaining = [row for row in candidates if row.doc_id != first.id]
        selector = SingleSignalSelector(self.backend, self.runtime, mode)
        selector.restore_initial(initial_memory, candidates, scores)
        second_scores = selector.score(memory, remaining)
        second_id = pick(second_scores)
        if second_id == first.id:
            raise ProtocolError("Second document repeats the shared first document")
        document = Document.model_validate(self.runtime.invoke("read", {"doc_id": second_id},
            lambda: self.retriever.fetch(second_id).model_dump()))
        if document.id != second_id:
            raise ProtocolError("Fetched id does not match the selected candidate")
        self.runtime.event("phase1c_selected", ref="d2", doc_id=second_id, score=second_scores[second_id])
        bundle = reader_bundle(task, [first, document])
        result = Result(task_id=task.id, answer="", supporting_facts=[], runtime_valid=False,
                        verified=False, status="SELECTION_ONLY_NOT_EVALUATED", repairs=0,
                        usage=dict(self.runtime.used) | dict(self.runtime.tokens))
        return {"task_id": task.id, "branch": branch, "first_doc_id": first.id, "second_doc_id": second_id,
                "selected_sequence": [first.id, second_id], "scores": second_scores,
                "logical_reads": 2, "observation_after_second": False,
                "observation_status": checkpoint["observation_status"] if branch == "current_once" else "not_consumed",
                "bundle": bundle, "result": result.model_dump(mode="json")}
