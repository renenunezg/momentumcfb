import json
import logging
import os
import time
from pathlib import Path

import requests

from backend.config import CFBD_BASE_URL


class CFBDError(RuntimeError):
    pass


log = logging.getLogger(__name__)


class CFBDClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        max_calls: int = 100,
        min_remaining: int = 40,
        usage_file: Path | None = None,
    ):
        key = api_key or os.getenv("CFBD_API_KEY")
        if not key:
            raise CFBDError("CFBD_API_KEY is not set")
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {key}"
        if max_calls < 1 or min_remaining < 0:
            raise ValueError(
                "CFBD call budget must be positive and reserve nonnegative"
            )
        self.max_calls = max_calls
        self.min_remaining = min_remaining
        self.usage_file = usage_file or (
            Path(os.environ["CFBD_USAGE_FILE"])
            if os.getenv("CFBD_USAGE_FILE")
            else None
        )
        self.calls_used = 0
        self.remaining: int | None = None
        self._planned_calls = 0
        if self.usage_file and self.usage_file.exists():
            usage = json.loads(self.usage_file.read_text())
            self.calls_used = int(usage["calls_used"])
            self.remaining = usage["remaining"]

    def ensure_budget(self, estimated_calls: int) -> None:
        """Reserve a sequential job before its first useful quota-reading request.

        The first required response supplies the provider quota header; the
        remaining plan is checked against it before any subsequent request.
        Retries count against the same hard session limit as successful calls.
        """
        if estimated_calls < 0:
            raise ValueError("CFBD estimated calls cannot be negative")
        estimated_calls = max(estimated_calls, self._planned_calls)
        if not estimated_calls:
            return
        if self.calls_used + estimated_calls > self.max_calls:
            raise CFBDError(
                f"CFBD job needs {estimated_calls} calls with {self.calls_used} already "
                f"spent; session budget is {self.max_calls}. More than 100 calls "
                "requires explicit approval before increasing --max-calls."
            )
        if self.remaining is None and self.calls_used:
            raise CFBDError("CFBD response omitted X-CallLimit-Remaining; stopping")
        if (
            self.remaining is not None
            and self.remaining - estimated_calls < self.min_remaining
        ):
            raise CFBDError(
                f"CFBD job needs {estimated_calls} calls, provider has {self.remaining}; "
                f"preserving a {self.min_remaining}-call reserve"
            )
        self._planned_calls = estimated_calls
        log.info(
            "CFBD budget: estimated=%s spent=%s limit=%s remaining=%s reserve=%s",
            estimated_calls,
            self.calls_used,
            self.max_calls,
            self.remaining,
            self.min_remaining,
        )

    def _persist_usage(self) -> None:
        if self.usage_file:
            self.usage_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.usage_file.with_suffix(".tmp")
            temporary.write_text(
                json.dumps({"calls_used": self.calls_used, "remaining": self.remaining})
            )
            temporary.replace(self.usage_file)

    def extend_budget(self, additional_calls: int) -> None:
        """Budget a provider row-cap fallback before issuing extra requests."""
        self.ensure_budget(self._planned_calls + additional_calls)

    def get(
        self,
        path: str,
        params: dict | None = None,
        retries: int = 3,
        timeout: float = 60,
    ) -> list:
        if retries < 1:
            raise ValueError("retries must be positive")
        url = f"{CFBD_BASE_URL}{path}"
        for attempt in range(retries):
            self.ensure_budget(max(1, self._planned_calls))
            self.calls_used += 1
            # Persist attempts even if the request fails before a response.
            if self.remaining is not None:
                self.remaining -= 1
            self._persist_usage()
            resp = self.session.get(url, params=params, timeout=timeout)
            remaining = resp.headers.get("X-CallLimit-Remaining")
            try:
                self.remaining = int(remaining) if remaining is not None else None
            except (TypeError, ValueError):
                self.remaining = None
            self._persist_usage()
            if resp.status_code == 429:
                if attempt < retries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise CFBDError(f"GET {path} rate limited after {retries} attempts")
            if resp.status_code >= 500 and attempt < retries - 1:
                # Gateway errors clear within seconds; the call is idempotent.
                time.sleep(10 * (attempt + 1))
                continue
            if resp.status_code != 200:
                raise CFBDError(
                    f"GET {path} returned {resp.status_code}: {resp.text[:200]}"
                )
            data = resp.json()
            if not isinstance(data, list):
                raise CFBDError(f"GET {path} returned a non-list payload")
            self._planned_calls = max(0, self._planned_calls - 1)
            return data
        raise CFBDError(f"GET {path} failed after {retries} attempts")
