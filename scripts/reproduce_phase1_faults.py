"""Deterministic counterexamples for commit dddd3d8; no model or gold calls."""
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oec.controller import Controller
from oec.runtime import Runtime
from oec.types import (Candidate, Claim, Document, Draft, Evidence, Extraction,
                       Goal, Plan, ProposedClaim, Query, Settings, State, Status, Task)


class FixedRetriever:
    def __init__(self):
        self.documents = {
            "birth": Document(id="birth", title="B biography", sentences=["B was born in Prague."]),
            "advisor": Document(id="advisor", title="A doctoral record", sentences=["A's doctoral advisor was B."]),
            "other": Document(id="other", title="Other", sentences=["Other information."])}

    def search(self, query, top_n):
        return [Candidate(doc_id=key, title=doc.title, snippet=doc.sentences[0], retrieval_score=10-index)
                for index, (key, doc) in enumerate(self.documents.items())][:top_n]

    def fetch(self, doc_id):
        return self.documents[doc_id]


class FixedDecisions:
    def verify_claims(self, state, claim_ids):
        return {key: .99 for key in claim_ids}


class RepeatingLLM:
    def __init__(self, runtime, bridge=False):
        self.runtime, self.bridge, self.answers = runtime, bridge, 0

    def ask(self, role, context, schema, critical=False):
        if role == "plan":
            result = Plan(goals=[Goal(id="g1", description="Confirm A's advisor")])
            if self.bridge:
                result.goals.append(Goal(id="g2", description="Find advisor birthplace", depends_on=["g1"]))
        elif role == "query":
            result = Query(query="A advisor birthplace")
        elif role == "extract":
            claims = []
            if self.bridge:
                key = context["document"]["id"]
                if key == "birth":
                    claims = [ProposedClaim(goal_id="g2", text="B was born in Prague", value="Prague", sentence_ids=[0])]
                elif key == "advisor":
                    claims = [ProposedClaim(goal_id="g1", text="A's advisor was B", value="B", sentence_ids=[0])]
            result = Extraction(claims=claims)
        elif role == "answer":
            self.answers += 1
            result = Draft(text="", claim_ids=[], citation_ids=[], abstain=True)
        else:
            raise AssertionError(role)
        raw = self.runtime.invoke("llm", {"role": role, "context": context},
                                  lambda: result.model_dump(), critical, "scripted_counterexample")
        return schema.model_validate(raw)


def reproduce():
    results = {}
    for name, bridge in (("repeated_query_empty_answer", False), ("early_bridge_discarded", True)):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(method="rank", gap_priority=False)
            runtime = Runtime(Path(directory) / "run", settings.limits)
            llm = RepeatingLLM(runtime, bridge)
            controller = Controller(llm, FixedDecisions(), FixedRetriever(), runtime, settings)
            result = controller.run(Task(id="t", question="A's advisor birthplace?"))
            state = controller.state
            results[name] = {"status": result.status, "answer": result.answer,
                "answer_calls": llm.answers, "search_used": runtime.used["search"],
                "read_used": runtime.used["read"], "shown": state.shown_doc_ids,
                "stored_evidence": sorted(state.evidence),
                "search_budget_left": settings.limits.search_calls-runtime.used["search"],
                "read_budget_left": settings.limits.read_calls-runtime.used["read"]}
    incomplete = {"claims": [{"goal_id": "g1", "text": "", "value": "", "sentence_ids": []}]}
    results["silent_schema_pruning"] = {"raw_claims": 1,
        "parsed_claims": len(Extraction.model_validate(incomplete).claims)}
    with tempfile.TemporaryDirectory() as directory:
        settings = Settings(method="rank", gap_priority=False)
        runtime = Runtime(Path(directory) / "run", settings.limits)
        controller = Controller(None, None, None, runtime, settings)
        state = State(task_id="t", question="Who is the advisor?", goals={"g1": Goal(id="g1", description="advisor")})
        state.evidence["e1"] = Evidence(id="e1", doc_id="d", title="Record", sentence_id=0,
                                         quote="The advisor was B.", source="fixture")
        state.claims["c1"] = Claim(id="c1", goal_id="g1", text="The advisor was B", value="B",
            evidence_ids=["e1"], depends_on=[], mode="DIRECT", status=Status.SUPPORTED)
        normalized = controller.normalize_draft(state, Draft(text="Unsupported answer X",
            claim_ids=["c1", "unknown"], citation_ids=["invented"], abstain=False))
        results["metadata_corrected_without_text_rewrite"] = normalized.model_dump()
    return results


if __name__ == "__main__":
    print(json.dumps(reproduce(), ensure_ascii=False, indent=2))
