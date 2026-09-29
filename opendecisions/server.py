"""HTTP server.

    POST /v1/decisions   the Decisions API
    POST /v1/systemone   TypeSafe's System One API, for the official SDK
    GET  /v1/models
    GET  /health
"""
from __future__ import annotations

import hmac
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from . import __version__
from .api import DecisionsRequest
from .providers import ProviderError, Router, UnknownModel


def _error(status: int, message: str, kind: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": kind}})


def _invalid(exc: ValidationError) -> JSONResponse:
    parts = [f"{'.'.join(str(x) for x in e['loc']) or 'body'}: {e['msg']}" for e in exc.errors()]
    return _error(422, "; ".join(parts), "invalid_request")


def create_app(router: Router, api_key: str | None = None) -> FastAPI:
    app = FastAPI(title="OpenDecisions", version=__version__)
    app.state.router = router

    @app.middleware("http")
    async def frame(request: Request, call_next):
        if api_key and request.url.path.startswith("/v1/"):
            given = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            if not hmac.compare_digest(given, api_key):
                return _error(401, "missing or wrong API key", "authentication_error")
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request.headers.get("x-request-id") or uuid.uuid4().hex
        response.headers["x-processing-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
        return response

    def run(fn, body: Any) -> JSONResponse:
        try:
            return JSONResponse(content=fn(body))
        except ValidationError as e:
            return _invalid(e)
        except UnknownModel as e:
            return _error(404, str(e), "model_not_found")
        except ProviderError as e:
            return _error(502, str(e), "provider_error")
        except ValueError as e:
            return _error(422, str(e), "invalid_request")

    def decisions(body: Any) -> dict[str, Any]:
        return router.decide(DecisionsRequest.model_validate(body)).model_dump(exclude_none=True)

    @app.post("/v1/decisions")
    def post_decisions(body: dict[str, Any]) -> JSONResponse:
        return run(decisions, body)

    @app.post("/v1/systemone")
    def post_systemone(body: dict[str, Any]) -> JSONResponse:
        return run(router.systemone, body)

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": m, "object": "model"} for m in router.models()]}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "models": router.models(),
                "fallback": router.fallback, "threshold": router.threshold}

    @app.get("/")
    def index() -> dict[str, Any]:
        return {"name": "OpenDecisions", "version": __version__, "endpoints": ["/v1/decisions", "/v1/systemone", "/v1/models"],
                "docs": "/docs"}

    return app
