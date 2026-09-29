"""Python client for any OpenDecisions server.

    from opendecisions import OpenDecision, image

    client = OpenDecision()
    r = client.decisions.create(
        context=["The customer wrote:", "My card was charged twice.", image("receipt.png")],
        questions=[{"id": "team", "question": "Which team should handle this?", "options": ["billing", "tech", "sales"]}],
    )
    r["team"].answer, r["team"].confidence
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator, Sequence

from .api import file_image


class APIError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


def image(ref: Any) -> dict[str, Any]:
    """An image context part from a URL, a data URI, or a local file."""
    ref = str(ref) if isinstance(ref, Path) else ref
    if isinstance(ref, str) and ref.startswith(("http://", "https://", "data:")):
        return {"type": "image", "url": ref} if ref.startswith("http") else {"type": "image", "data": ref}
    return file_image(ref)


def video(frames: Sequence[Any], fps: float = 2.0) -> dict[str, Any]:
    """A video context part from its frames (URLs, data URIs or local files) and the rate they were sampled at."""
    return {"type": "video", "frames": [f if isinstance(f, str) and f.startswith(("http://", "https://", "data:"))
                                        else image(f)["data"] for f in frames], "fps": fps}


class Result:
    def __init__(self, data: dict[str, Any]) -> None:
        self.raw = data
        self.id: str = data["id"]
        self.answer: str = data["answer"]
        self.confidence: float = data["confidence"]
        self.probabilities: dict[str, float] = data["probabilities"]
        self.expected: float | None = data.get("expected")
        self.provider: str | None = data.get("provider")

    def __repr__(self) -> str:
        return f"Result(id={self.id!r}, answer={self.answer!r}, confidence={self.confidence:.3f}, provider={self.provider!r})"


class Decision:
    def __init__(self, data: dict[str, Any]) -> None:
        self.raw = data
        self.id: str = data.get("id", "")
        self.model: str = data.get("model", "")
        self.results = [Result(r) for r in data.get("results", [])]
        self.usage: dict[str, Any] = data.get("usage", {})
        self.notes: list[str] = data.get("notes") or []

    def __getitem__(self, key: str | int) -> Result:
        if isinstance(key, int):
            return self.results[key]
        for r in self.results:
            if r.id == key:
                return r
        raise KeyError(key)

    def __iter__(self) -> Iterator[Result]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    @property
    def answers(self) -> dict[str, str]:
        return {r.id: r.answer for r in self.results}

    def __repr__(self) -> str:
        return f"Decision(model={self.model!r}, answers={self.answers!r})"


class Decisions:
    def __init__(self, client: "OpenDecision") -> None:
        self._client = client

    def create(self, *, questions: Sequence[dict[str, Any]] | dict[str, dict[str, Any]], context: Any = None,
               model: str = "auto", threshold: float | None = None, **extra: Any) -> Decision:
        body: dict[str, Any] = {"model": model, "context": context, "questions": questions, **extra}
        if threshold is not None:
            body["threshold"] = threshold
        return Decision(self._client._request("POST", "/v1/decisions", body))


class OpenDecision:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: float = 120.0) -> None:
        self.base_url = (base_url or os.environ.get("OPENDECISIONS_BASE_URL") or "http://localhost:8000").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENDECISIONS_API_KEY")
        self.timeout = timeout
        self.decisions = Decisions(self)

    def _request(self, method: str, path: str, body: Any = None) -> dict[str, Any]:
        headers = {"Content-Type": "application/json", "User-Agent": "opendecisions"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", "replace")
            try:
                text = json.loads(text)["error"]["message"]
            except Exception:
                pass
            raise APIError(e.code, text) from None

    def models(self) -> list[str]:
        return [m["id"] for m in self._request("GET", "/v1/models")["data"]]
