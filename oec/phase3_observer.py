"""Sequential-run observer ledger: preserve the first bounded result for an exact context."""
import copy
import json
from pathlib import Path
from pydantic import ValidationError
from .phase1c import Progress, RoleJSONClient
from .runtime import HTTPFailure, ProtocolError, BudgetExceeded, digest


class SharedObserver(RoleJSONClient):
    def __init__(self, runtime, directory, identity_sha256):
        super().__init__(runtime)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = identity_sha256

    def ask(self, role, context, schema, critical=False, validator=None):
        if role != "phase1_observe":
            return super().ask(role, context, schema, critical, validator)
        if schema is not Progress:
            raise ProtocolError("Unexpected observer schema")
        key = digest({"role": role, "context": context, "signature": self.signature(),
                      "identity": self.identity, "schema": schema.model_json_schema()})
        path = self.directory/(key+".json")
        if path.is_file():
            entry = json.loads(path.read_text())
            body = {key: value for key, value in entry.items() if key != "entry_hash"}
            if entry.get("key") != key or digest(body) != entry.get("entry_hash"):
                raise RuntimeError("Observer ledger corrupted")
            for kind, count in entry["used_delta"].items():
                if self.runtime.used[kind]+count > self.runtime.ceiling(kind, critical):
                    raise BudgetExceeded(kind)
                self.runtime.used[kind] += count  # Logical charge; no physical request occurred.
            self.runtime.tokens.update(entry["token_delta"])
            self.runtime.event("phase3_observer_cache_hit", key=key,
                logical_used=entry["used_delta"], logical_tokens=entry["token_delta"],
                physical_requests=0, first_result_status=entry["status"])
            if entry["status"] == "bounded_protocol_failure":
                raise ProtocolError(entry["error"])
            result = schema.model_validate(entry["value"])
            if validator:
                validator(result)
            return result
        before_used, before_tokens = dict(self.runtime.used), dict(self.runtime.tokens)
        try:
            result = super().ask(role, context, schema, critical, validator)
            status, value, error = "success", result.model_dump(), None
        except HTTPFailure as exc:
            # Old kernel catches ProtocolError; configuration failures must escape that catch.
            raise RuntimeError("Observer HTTP/service failure; abort this new study") from exc
        except ProtocolError as exc:
            if "two output attempts failed" not in str(exc):
                raise
            status, value, error = "bounded_protocol_failure", None, str(exc)
        used = {key: value-before_used.get(key, 0) for key, value in self.runtime.used.items()
                if value != before_used.get(key, 0)}
        tokens = {key: value-before_tokens.get(key, 0) for key, value in self.runtime.tokens.items()
                  if value != before_tokens.get(key, 0)}
        entry = {"key": key, "status": status, "value": value, "error": error,
                 "used_delta": used, "token_delta": tokens}
        entry["entry_hash"] = digest(entry)
        # Intentional exclusive write. Run selector arms sequentially against one ledger.
        with path.open("x", encoding="utf-8") as handle:
            json.dump(entry, handle, ensure_ascii=False, sort_keys=True)
        self.runtime.event("phase3_observer_cache_miss", key=key, status=status)
        if status != "success":
            raise ProtocolError(error)
        return result
