import hashlib
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Callable
from .types import Limits


class ProtocolError(RuntimeError):
    pass


class HTTPFailure(ProtocolError):
    def __init__(self, status, body, retry_after=0):
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.retry_after = retry_after


class BudgetExceeded(RuntimeError):
    pass


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class Runtime:
    def __init__(self, directory: Path, limits: Limits):
        directory.mkdir(parents=True, exist_ok=False)
        self.path = directory / "events.jsonl"
        self.limits = limits
        self.used = Counter()
        self.tokens = Counter()
        self.started = time.perf_counter()
        self.event("runtime_start", limits=limits.model_dump())

    def event(self, event_type: str, **values):
        row = {"event": event_type, "elapsed_s": time.perf_counter() - self.started, **values}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical(row) + "\n")

    def ceiling(self, kind: str, critical: bool) -> int:
        total = getattr(self.limits, kind + "_calls")
        reserves = {"search": self.limits.repair_search_reserve,
                    "read": self.limits.repair_read_reserve,
                    "llm": self.limits.terminal_llm_reserve,
                    "decision": self.limits.terminal_decision_reserve}
        return total if critical else total - reserves[kind]

    def available(self, kind: str, critical: bool = False) -> bool:
        return self.used[kind] < self.ceiling(kind, critical)

    def invoke(self, kind: str, request: dict, function: Callable,
               critical: bool = False, model: str = "local"):
        if not self.available(kind, critical):
            raise BudgetExceeded(kind)
        self.used[kind] += 1  # Charge before execution; failed requests also consume budget.
        start = time.perf_counter()
        self.event("request", kind=kind, model=model, request=request,
                   request_hash=digest(request), attempt=self.used[kind])
        try:
            response = function()
        except Exception as error:
            self.event("request_error", kind=kind, error_type=type(error).__name__,
                       error=str(error), latency_s=time.perf_counter() - start)
            raise
        if isinstance(response, dict):
            usage = response.get("usage", {})
            for key in ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens"):
                if key in usage:
                    self.tokens[f"{kind}.{key}"] += int(usage[key])
        self.event("response", kind=kind, model=model, response=response,
                   response_hash=digest(response), latency_s=time.perf_counter() - start)
        return response

    def snapshot(self, state):
        payload = state.model_dump(mode="json")
        self.event("state", revision=state.revision, state=payload, state_hash=digest(payload))


def post_json(url: str, payload: dict, api_key: str, timeout: float = 60):
    request = urllib.request.Request(
        url, data=canonical(payload).encode(), method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        # Body is preserved, but auth headers are never part of the event log.
        try:
            retry_after = float(error.headers.get("Retry-After", "0"))
        except ValueError:
            retry_after = 0
        raise HTTPFailure(error.code, body, retry_after) from error
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise ProtocolError("Non-JSON HTTP response: " + raw) from error


def retry_json(runtime, kind, url, payload, api_key, critical=False, model="unknown"):
    for attempt in range(3):
        try:
            return runtime.invoke(kind, payload,
                                  lambda: post_json(url, payload, api_key),
                                  critical=critical, model=model)
        except HTTPFailure as error:
            if error.status not in (429, 500, 502, 503, 504, 529) or attempt == 2:
                raise
            delay = min(8, max(0.5 * 2 ** attempt, error.retry_after))
            runtime.event("retry", kind=kind, status=error.status, delay_s=delay)
            time.sleep(delay)
    raise AssertionError("unreachable")
