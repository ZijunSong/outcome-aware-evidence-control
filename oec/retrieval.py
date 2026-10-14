import json
import math
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from .graph import terms
from .runtime import ProtocolError
from .types import Candidate, Document

# Stable sentence ids for Wikipedia passages. Gold Hotpot sentence ids use a different segmentation.
SENTENCE_SPLITTER = "oec-sent-v1"


class LocalBM25:
    """Small-corpus reference retriever. Replace with an indexed service for full Wikipedia."""
    def __init__(self, path: Path):
        documents = [Document.model_validate(json.loads(line))
                     for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.documents = {item.id: item for item in documents}
        if len(self.documents) != len(documents):
            raise ProtocolError("Duplicate document id in corpus")
        self.counts = {}
        self.df = Counter()
        for document in documents:
            tokens = self.tokenize(document.title + " " + " ".join(document.sentences))
            self.counts[document.id] = Counter(tokens)
            self.df.update(set(tokens))
        self.average_length = sum(sum(c.values()) for c in self.counts.values()) / max(1, len(documents))
        self.scope = None

    def set_scope(self, doc_ids):
        if doc_ids is not None and not set(doc_ids) <= set(self.documents):
            raise ProtocolError("Task context references an unknown corpus document")
        self.scope = None if doc_ids is None else set(doc_ids)

    @staticmethod
    def tokenize(text):
        import re
        return re.findall(r"\w+", text.casefold())

    def search(self, query: str, top_n: int) -> list[Candidate]:
        query_terms = terms(query)
        scores = []
        size = len(self.documents)
        for document_id, counts in self.counts.items():
            if self.scope is not None and document_id not in self.scope:
                continue
            length = sum(counts.values())
            score = 0.0
            for token in query_terms:
                frequency = counts[token]
                if not frequency:
                    continue
                idf = math.log(1 + (size - self.df[token] + 0.5) / (self.df[token] + 0.5))
                denominator = frequency + 1.2 * (0.25 + 0.75 * length / max(1, self.average_length))
                score += idf * frequency * 2.2 / denominator
            if score > 0:
                scores.append((score, document_id))
        scores.sort(key=lambda row: (-row[0], row[1]))
        return [Candidate(doc_id=document_id, title=self.documents[document_id].title,
                          snippet=" ".join(self.documents[document_id].sentences)[:1200],
                          retrieval_score=score)
                for score, document_id in scores[:top_n]]

    def fetch(self, doc_id: str) -> Document:
        if doc_id not in self.documents:
            raise ProtocolError("Retriever returned a non-existent document id")
        return self.documents[doc_id]


def split_sentences(text: str) -> list[str]:
    compact = " ".join(text.split())
    if not compact:
        raise ProtocolError("Retrieved document has no text to split")
    pieces = []
    start = 0
    for index, char in enumerate(compact):
        if char in ".!?" and index + 1 < len(compact) and compact[index + 1] == " ":
            pieces.append(compact[start:index + 1])
            start = index + 2
    if start < len(compact):
        pieces.append(compact[start:])
    return [piece for piece in pieces if piece.strip()]


class Wiki18Retriever:
    """Full Wikipedia 2018 index already served at the local E5 endpoint."""

    def __init__(self, base_url: str = "http://127.0.0.1:18080", timeout: float = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.health = self._request("GET", "/health")
        if not self.health.get("ok"):
            raise ProtocolError("Wikipedia retriever health check failed")
        self._hits: dict[str, dict] = {}

    def describe(self) -> dict:
        return {"kind": "wiki18-e5", "url": self.base_url, "health": self.health,
                "sentence_splitter": SENTENCE_SPLITTER,
                "note": "Search uses the full 2018 Wikipedia E5 index. "
                        "Sentence ids are oec-sent-v1, not HotpotQA gold sentence ids."}

    def search(self, query: str, top_n: int) -> list[Candidate]:
        payload = self._request("POST", "/retrieve",
                                {"queries": [query], "topk": top_n, "return_scores": True})
        groups = payload.get("result") or []
        if len(groups) != 1:
            raise ProtocolError("Wikipedia retriever did not return one result group")
        candidates = []
        for item in groups[0]:
            document = item.get("document") if isinstance(item, dict) else None
            if not isinstance(document, dict):
                raise ProtocolError("Wikipedia hit is missing its document")
            doc_id = str(document.get("document_id") or document.get("id") or "")
            title = str(document.get("title") or "")
            text = str(document.get("text") or "")
            if not doc_id or not title or not text:
                raise ProtocolError("Wikipedia hit is missing id, title, or text")
            self._hits[doc_id] = {"title": title, "text": text}
            candidates.append(Candidate(
                doc_id=doc_id, title=title, snippet=" ".join(text.split())[:1200],
                retrieval_score=float(item.get("score") or 0.0)))
        return candidates

    def fetch(self, doc_id: str) -> Document:
        opened = self._request("POST", "/open", {"document_id": doc_id}).get("document")
        if not isinstance(opened, dict):
            raise ProtocolError("Wikipedia open did not return a document")
        opened_id = str(opened.get("document_id") or opened.get("id") or "")
        title = str(opened.get("title") or "")
        text = str(opened.get("text") or "")
        if opened_id != doc_id:
            raise ProtocolError("Fetched document id does not match the requested id")
        seen = self._hits.get(doc_id)
        if seen is not None and (seen["title"] != title or seen["text"] != text):
            raise ProtocolError("Fetched document does not match the retrieved hit")
        return Document(id=doc_id, title=title, sentences=split_sentences(text),
                        source=f"wiki18-e5/{SENTENCE_SPLITTER}")

    def _request(self, method: str, path: str, payload: dict | None = None):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base_url + path, data=data, method=method,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise ProtocolError(f"Wikipedia retriever HTTP {error.code}: {body}") from error
        except urllib.error.URLError as error:
            raise ProtocolError("Wikipedia retriever is unreachable: " + str(error.reason)) from error
        try:
            return json.loads(raw)
        except json.JSONDecodeError as error:
            raise ProtocolError("Wikipedia retriever returned non-JSON") from error
