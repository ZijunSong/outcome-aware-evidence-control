from enum import StrEnum
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Status(StrEnum):
    CANDIDATE = "CANDIDATE"
    SUPPORTED = "SUPPORTED"
    CONFLICTED = "CONFLICTED"
    INVALIDATED = "INVALIDATED"


class Goal(Strict):
    id: str
    description: str
    depends_on: list[str] = Field(default_factory=list)
    required: bool = True


class Plan(Strict):
    goals: list[Goal] = Field(min_length=1, max_length=8)


class Document(Strict):
    id: str
    title: str
    sentences: list[str] = Field(min_length=1)
    source: str = "local_corpus"


class Candidate(Strict):
    doc_id: str
    title: str
    snippet: str
    retrieval_score: float


class Evidence(Strict):
    id: str
    doc_id: str
    title: str
    sentence_id: int = Field(ge=0)
    quote: str
    source: str


class ProposedClaim(Strict):
    goal_id: str
    text: str = Field(min_length=1)
    value: str = Field(min_length=1)
    sentence_ids: list[int] = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    mode: Literal["DIRECT", "DERIVED"] = "DIRECT"


class Extraction(Strict):
    claims: list[ProposedClaim] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def drop_incomplete_claims(cls, data):
        if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
            return data
        kept = []
        for row in data["claims"]:
            if not isinstance(row, dict):
                kept.append(row)
                continue
            text = str(row.get("text") or "").strip()
            value = str(row.get("value") or "").strip()
            sentence_ids = row.get("sentence_ids") or []
            if not text or not value or not sentence_ids:
                continue
            mode = str(row.get("mode") or "DIRECT").upper()
            row = dict(row)
            row["text"] = text
            row["value"] = value
            row["mode"] = mode if mode in ("DIRECT", "DERIVED") else "DIRECT"
            row["depends_on"] = row.get("depends_on") or []
            kept.append(row)
        return {**data, "claims": kept}


class Claim(Strict):
    id: str
    goal_id: str
    text: str
    value: str
    evidence_ids: list[str]
    depends_on: list[str]
    mode: Literal["DIRECT", "DERIVED"]
    status: Status = Status.CANDIDATE
    support_probability: float = Field(default=0, ge=0, le=1)


class Query(Strict):
    query: str = Field(min_length=1, max_length=500)


class Draft(Strict):
    text: str
    claim_ids: list[str]
    citation_ids: list[str]
    abstain: bool = False


class Limits(Strict):
    search_calls: int = Field(default=6, ge=1)
    read_calls: int = Field(default=12, ge=1)
    llm_calls: int = Field(default=64, ge=5)
    decision_calls: int = Field(default=200, ge=5)
    repair_search_reserve: int = Field(default=2, ge=0)
    repair_read_reserve: int = Field(default=4, ge=0)
    terminal_llm_reserve: int = Field(default=2, ge=1)
    terminal_decision_reserve: int = Field(default=4, ge=1)

    @model_validator(mode="after")
    def check_reserves(self):
        pairs = [(self.repair_search_reserve, self.search_calls),
                 (self.repair_read_reserve, self.read_calls),
                 (self.terminal_llm_reserve, self.llm_calls),
                 (self.terminal_decision_reserve, self.decision_calls)]
        if any(reserve >= total for reserve, total in pairs):
            raise ValueError("Each reserve must be smaller than its total budget")
        return self


class Settings(Strict):
    method: Literal["rank", "relevance", "utility_static", "utility_dynamic", "full"] = "full"
    gap_priority: bool = True
    top_n: int = Field(default=100, ge=1)
    reads_per_search: int = Field(default=2, ge=1)
    candidate_batch: int = Field(default=8, ge=1)
    support_threshold: float = Field(default=0.85, ge=0, le=1)
    answer_threshold: float = Field(default=0.9, ge=0, le=1)
    max_repairs: int = Field(default=2, ge=0)
    weight_gap: float = 0.5
    weight_bridge: float = 0.3
    weight_disambiguate: float = 0.2
    weight_redundancy: float = 0.15
    weight_length: float = 0.05
    limits: Limits = Field(default_factory=Limits)

    @model_validator(mode="after")
    def reserve_only_for_repair_methods(self):
        if self.method != "full":
            self.limits.repair_search_reserve = 0
            self.limits.repair_read_reserve = 0
        return self


class RepairRecord(Strict):
    target_claim_id: str
    target_goal_id: str
    invalidated_ids: list[str]
    disputed_claims: dict[str, str]
    reason: str


class State(Strict):
    task_id: str
    question: str
    goals: dict[str, Goal] = Field(default_factory=dict)
    claims: dict[str, Claim] = Field(default_factory=dict)
    evidence: dict[str, Evidence] = Field(default_factory=dict)
    shown_doc_ids: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    repair_records: list[RepairRecord] = Field(default_factory=list)
    revision: int = 0


class Task(Strict):
    id: str
    question: str
    corpus_ids: list[str] | None = None  # Public task context ids; never gold support labels.


class Result(Strict):
    task_id: str
    answer: str
    supporting_facts: list[tuple[str, int]]
    runtime_valid: bool
    verified: bool
    status: str
    repairs: int
    usage: dict[str, int]
