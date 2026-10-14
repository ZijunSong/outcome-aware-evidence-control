from .graph import complete
from .jev import validate_answers
from .types import (Candidate, Document, Draft, Extraction, Goal, Plan,
                    ProposedClaim, Query, State)


DOCUMENTS = {
    "d1": Document(id="d1", title="Unverified profile",
        sentences=["A worked with B. This directory does not verify who supervised A's doctorate."]),
    "d2": Document(id="d2", title="B biography", sentences=["B was born in Prague."]),
    "d3": Document(id="d3", title="A doctoral record", sentences=["A's doctoral advisor was C."]),
    "d4": Document(id="d4", title="C biography", sentences=["C was born in Vienna."])
}


class DemoRetriever:
    def search(self, query, top_n):
        doc_ids = (["d3", "d4"] if "verify" in query or "C" in query else ["d1", "d2"])
        return [Candidate(doc_id=doc_id, title=DOCUMENTS[doc_id].title,
                          snippet=" ".join(DOCUMENTS[doc_id].sentences), retrieval_score=10-index)
                for index, doc_id in enumerate(doc_ids)][:top_n]

    def fetch(self, doc_id):
        return DOCUMENTS[doc_id]


class DemoLLM:
    """Scripted fixture. The first extraction intentionally makes an advisor-binding error."""
    def __init__(self, runtime):
        self.runtime = runtime

    def ask(self, role, context, schema, critical=False):
        def produce():
            if role == "plan":
                result = Plan(goals=[Goal(id="g1", description="Confirm A's doctoral advisor"),
                    Goal(id="g2", description="Find the confirmed advisor's birthplace", depends_on=["g1"])])
            elif role == "priority":
                result = schema(goal_id=context["frontier_ids"][0])
            elif role == "query":
                values = [claim["value"] for claim in context["hypotheses"] if claim["status"] == "SUPPORTED"]
                query = ("verify A doctoral advisor" if context["repair"] and "C" not in values
                         else "C birthplace" if "C" in values
                         else "B birthplace" if "B" in values else "A doctoral advisor")
                result = Query(query=query)
            elif role == "extract":
                doc_id = context["document"]["id"]
                if doc_id in ("d1", "d3"):
                    value = "B" if doc_id == "d1" else "C"
                    claim = ProposedClaim(goal_id="g1", text=f"A's doctoral advisor was {value}",
                                          value=value, sentence_ids=[0])
                else:
                    advisor = "B" if doc_id == "d2" else "C"
                    parent = next((c for c in context["hypotheses"] if c["goal_id"] == "g1"
                                   and c["value"] == advisor and c["status"] == "SUPPORTED"), None)
                    if parent is None:
                        return Extraction(claims=[]).model_dump()
                    city = "Prague" if doc_id == "d2" else "Vienna"
                    claim = ProposedClaim(goal_id="g2", text=f"A's doctoral advisor {advisor} was born in {city}",
                                          value=city, sentence_ids=[0], depends_on=[parent["id"]], mode="DERIVED")
                result = Extraction(claims=[claim])
            elif role == "answer":
                selected = [c for c in context["hypotheses"] if c["status"] == "SUPPORTED"]
                city = next((c["value"] for c in selected if c["goal_id"] == "g2"), "")
                result = Draft(text=city, claim_ids=[c["id"] for c in selected] if city else [],
                    citation_ids=sorted({eid for c in selected for eid in c["evidence_ids"]}) if city else [],
                    abstain=not bool(city))
            else:
                raise ValueError(role)
            return result.model_dump()
        raw = self.runtime.invoke("llm", {"role": role, "context": context}, produce, critical, "scripted_demo")
        return schema.model_validate(raw)


class DemoJev:
    def __init__(self, runtime):
        self.runtime = runtime

    def decide(self, state, questions, critical=False):
        payload = {"state": state, "questions": questions}
        def produce():
            answers = {}
            for key, question in questions.items():
                if question["type"] == "choice":
                    selected = next(option for option in question["criteria"] if option != "DEFER")
                    answers[key] = {"type": "choice", "choice": selected, "confidence": 1.0,
                                    "probabilities": {option: float(option == selected)
                                                      for option in question["criteria"]}}
                else:
                    probability = 0.97
                    if state["phase"] == "utility":
                        doc_id = question["instructions"]["candidate_doc_id"]
                        target = state["target_goal_id"]
                        probability = 0.95 if (target == "g1" and doc_id in ("d1", "d3")) or (
                            target == "g2" and doc_id in ("d2", "d4")) else 0.05
                    if state["phase"] == "final":
                        claim = next(row["claim"] for row in state["claim_rows"] if row["claim"]["id"] == key)
                        if claim["goal_id"] == "g1" and claim["value"] == "B":
                            probability = 0.1
                    answers[key] = {"type": "noul", "noul": probability}
            return {"model": "scripted_demo", "answers": answers}
        response = self.runtime.invoke("decision", payload, produce, critical, "scripted_demo")
        return validate_answers(payload, response)
