"""Where decisions are computed: the local OneJev engine or a remote System One endpoint, and the router between them."""
from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from typing import Any

from .api import DecisionsRequest, DecisionsResponse, Result, Usage, result_from, split_context, systemone_question

REMOTES = {
    "typesafe": ("https://api.typesafe.ai/v1/systemone", "TYPESAFE_API_KEY", "jev-latest"),
    "openrouter": ("https://openrouter.ai/api/alpha/decisions", "OPENROUTER_API_KEY", "typesafe/jev-1.13"),
}


class ProviderError(RuntimeError):
    pass


class UnknownModel(ValueError):
    pass


class Provider:
    name: str
    model: str
    accepts_media: bool
    local: bool = False

    def systemone(self, state: Any, questions: dict[str, Any], media: list[dict]) -> tuple[dict[str, Any], dict[str, Any]]:
        raise NotImplementedError


class LocalProvider(Provider):
    local = True

    def __init__(self, engine: Any, model: str, debias: int = 1) -> None:
        self.engine = engine
        self.model = model
        self.name = "local"
        self.debias = debias
        self.accepts_media = hasattr(engine, "_set_media")
        self._lock = threading.Lock()

    def systemone(self, state, questions, media):
        from .onejev.schema import SystemOneRequest

        if media and not self.accepts_media:
            raise ValueError(f"{self.model} reads text only")
        request = SystemOneRequest.model_validate({"state": state, "questions": questions, "model": self.model})
        with self._lock:
            if self.accepts_media:
                response, _ = self.engine.decide(request, debias=self.debias, media=media)
            else:
                response, _ = self.engine.decide(request, debias=self.debias)
        data = response.model_dump()
        return data["answers"], data["usage"]


class RemoteProvider(Provider):
    def __init__(self, name: str, url: str, model: str, api_key: str | None = None, accepts_media: bool = False,
                 timeout: float = 120.0) -> None:
        if not url.rstrip("/").endswith(("/systemone", "/decisions")):
            url = url.rstrip("/") + "/v1/systemone"
        self.name, self.url, self.model, self.api_key = name, url, model, api_key
        self.accepts_media, self.timeout = accepts_media, timeout

    def with_model(self, model: str) -> "RemoteProvider":
        return RemoteProvider(self.name, self.url, model, self.api_key, self.accepts_media, self.timeout)

    def systemone(self, state, questions, media):
        if media and not self.accepts_media:
            raise ProviderError(f"{self.name} reads text only")
        body: dict[str, Any] = {"model": self.model, "state": state, "questions": questions}
        if media:
            body["media"] = media
        headers = {"Content-Type": "application/json", "User-Agent": "opendecisions"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise ProviderError(f"{self.name} returned HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise ProviderError(f"{self.name} is unreachable: {e}") from e
        if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
            raise ProviderError(f"{self.name} returned no answers")
        return data["answers"], data.get("usage") or {}


def known_remote(name: str, model: str | None = None) -> RemoteProvider | None:
    """A hosted provider whose key is set in the environment."""
    url, env, default = REMOTES[name]
    key = os.environ.get(env)
    return RemoteProvider(name, url, model or default, key) if key else None


def _add(usage: Usage, extra: dict[str, Any]) -> None:
    usage.input_tokens += int(extra.get("input_tokens") or 0)
    usage.output_tokens += int(extra.get("output_tokens") or 0)


class Router:
    """`auto` asks the local model first and sends answers under the confidence threshold to the fallback provider.
    A model name picks one provider: the local model's name, `local`, a provider name, or `provider/model`."""

    def __init__(self, local: LocalProvider | None = None, remotes: dict[str, RemoteProvider] | None = None,
                 fallback: str | None = None, threshold: float = 0.8) -> None:
        self.local = local
        self.remotes = dict(remotes or {})
        if fallback is not None and fallback not in self.remotes:
            raise ValueError(f"fallback {fallback!r} is not a configured provider")
        self.fallback = fallback
        self.threshold = threshold

    def models(self) -> list[str]:
        out = ["auto"]
        if self.local:
            out.append(self.local.model)
        out += [f"{p.name}/{p.model}" for p in self.remotes.values()]
        return out

    def resolve(self, model: str | None) -> tuple[Provider, RemoteProvider | None]:
        name = model or "auto"
        if name in ("auto", "default"):
            if self.local:
                return self.local, self.remotes.get(self.fallback) if self.fallback else None
            if self.fallback:
                return self.remotes[self.fallback], None
            if self.remotes:
                return next(iter(self.remotes.values())), None
            raise UnknownModel("no model is served")
        if self.local and name in ("local", self.local.model, "jev-latest", "jev-preview"):
            return self.local, None
        head, _, rest = name.partition("/")
        if head in self.remotes:
            p = self.remotes[head]
            return (p.with_model(rest) if rest else p), None
        if head in REMOTES:
            raise UnknownModel(f"{head} is not configured; set {REMOTES[head][1]}")
        raise UnknownModel(f"model {name!r} is not served; available: {', '.join(self.models())}")

    def decide(self, req: DecisionsRequest) -> DecisionsResponse:
        state, media = split_context(req.context, (req.model_extra or {}).get("media"))
        provider, fallback = self.resolve(req.model)
        threshold = self.threshold if req.threshold is None else req.threshold
        questions = {q.id: systemone_question(q, remote=not provider.local) for q in req.questions}
        answers, used = provider.systemone(state, questions, media)
        usage = Usage()
        _add(usage, used)
        results: list[Result] = []
        for q in req.questions:
            if q.id not in answers:
                raise ProviderError(f"{provider.name} did not answer {q.id!r}")
            results.append(result_from(q, answers[q.id], provider.name))
        notes = []
        low = [i for i, r in enumerate(results) if r.confidence < threshold]
        if fallback and low:
            if media and not fallback.accepts_media:
                notes.append(f"{len(low)} answers under {threshold} kept: {fallback.name} reads text only")
            else:
                ask = [req.questions[i] for i in low]
                try:
                    more, used = fallback.systemone(state, {q.id: systemone_question(q, remote=True) for q in ask}, media)
                    _add(usage, used)
                    for i, q in zip(low, ask):
                        results[i] = result_from(q, more[q.id], fallback.name)
                except (ProviderError, KeyError, ValueError) as e:
                    notes.append(f"{len(low)} answers under {threshold} kept: {e}")
        return DecisionsResponse(model=provider.model, results=results, usage=usage, notes=notes or None)

    def systemone(self, body: dict[str, Any]) -> dict[str, Any]:
        """A System One request passed through to the provider its model names."""
        provider, _ = self.resolve(body.get("model") or "auto")
        answers, usage = provider.systemone(body.get("state"), body.get("questions") or {}, body.get("media") or [])
        return {"model": provider.model, "answers": answers, "usage": usage}
