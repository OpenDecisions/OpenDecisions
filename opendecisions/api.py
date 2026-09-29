"""The Decisions API: request and response types, and how they map onto System One questions and answers."""
from __future__ import annotations

import base64
import mimetypes
import time
import uuid
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

MAX_OPTIONS = 255
MAX_LEVELS = 10
TEXT_PARTS = {"text", "input_text"}
IMAGE_PARTS = {"image", "image_url", "input_image"}
VIDEO_PARTS = {"video"}


class Question(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str | None = None
    question: str = Field(validation_alias=AliasChoices("question", "text", "instructions"))
    options: list[str] | dict[str, str | None] = Field(validation_alias=AliasChoices("options", "answers", "choices"))
    ordered: bool = False

    @model_validator(mode="after")
    def _check(self) -> "Question":
        names = list(self.options)
        if len(names) < 2:
            raise ValueError("a question needs at least two options")
        if len(set(names)) != len(names):
            raise ValueError("options must be unique")
        if self.ordered and len(names) > MAX_LEVELS:
            raise ValueError(f"an ordered question takes at most {MAX_LEVELS} options")
        if len(names) > MAX_OPTIONS:
            raise ValueError(f"a question takes at most {MAX_OPTIONS} options")
        return self

    @property
    def names(self) -> list[str]:
        return [str(n) for n in self.options]

    @property
    def descriptions(self) -> dict[str, str]:
        if isinstance(self.options, dict):
            return {str(k): str(v) for k, v in self.options.items() if v}
        return {}

    @property
    def kind(self) -> str:
        if self.ordered:
            return "score"
        if sorted(n.lower() for n in self.names) == ["no", "yes"]:
            return "noul"
        return "choice"


class DecisionsRequest(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    model: str = "auto"
    context: Any = Field(default=None, validation_alias=AliasChoices("context", "state", "input"))
    questions: list[Question]
    threshold: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _keyed_questions(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("questions"), dict):
            data = dict(data)
            data["questions"] = [{"id": k, **v} for k, v in data["questions"].items()]
        return data

    @model_validator(mode="after")
    def _ids(self) -> "DecisionsRequest":
        if not self.questions:
            raise ValueError("at least one question is required")
        seen = set()
        for i, q in enumerate(self.questions):
            q.id = q.id or f"q{i + 1}"
            if q.id in seen:
                raise ValueError(f"question id {q.id!r} is used twice")
            seen.add(q.id)
        return self


class Result(BaseModel):
    id: str
    answer: str
    confidence: float
    probabilities: dict[str, float]
    expected: float | None = None
    provider: str


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class DecisionsResponse(BaseModel):
    id: str = Field(default_factory=lambda: "dec_" + uuid.uuid4().hex[:24])
    object: str = "decision"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    results: list[Result]
    usage: Usage = Field(default_factory=Usage)
    notes: list[str] | None = None


def _image_ref(part: Any) -> Any:
    if isinstance(part, str):
        return part
    if not isinstance(part, dict):
        raise ValueError("an image must be a URL, a data URI, base64 data, a path, or an object holding one")
    ref = part.get("image_url", part.get("url", part.get("data", part.get("path"))))
    if isinstance(ref, dict):
        ref = ref.get("url")
    if not isinstance(ref, str) or not ref:
        raise ValueError("an image part needs image_url, url, data or path")
    return ref


def image_item(ref: str) -> dict[str, Any]:
    if ref.startswith(("http://", "https://")):
        return {"type": "image", "url": ref}
    if ref.startswith("data:") or len(ref) > 512:
        return {"type": "image", "data": ref}
    return {"type": "image", "path": ref}


def _is_part(x: Any) -> bool:
    return isinstance(x, dict) and x.get("type") in TEXT_PARTS | IMAGE_PARTS | VIDEO_PARTS


def split_context(context: Any, media: list[dict] | None = None) -> tuple[Any, list[dict]]:
    """The System One state and media list for a context: a string, any JSON value, or a list of text, image and video
    parts. Parts become one text with <image:N> and <video:N> placeholders in their places."""
    if _is_part(context):
        context = [context]
    if not (isinstance(context, list) and any(_is_part(p) for p in context)
            and all(isinstance(p, str) or _is_part(p) for p in context)):
        return context, list(media or [])
    lines, items = [], []
    for p in context:
        if isinstance(p, str):
            lines.append(p)
        elif p["type"] in TEXT_PARTS:
            lines.append(str(p.get("text", "")))
        elif p["type"] in IMAGE_PARTS:
            items.append(image_item(_image_ref(p)))
            lines.append(f"<image:{len(items)}>")
        else:
            frames = p.get("frames")
            if not isinstance(frames, list) or not frames:
                raise ValueError("a video part needs a non-empty frames list")
            items.append({"type": "video", "frames": [_image_ref(f) for f in frames], "fps": float(p.get("fps") or 2.0)})
            lines.append(f"<video:{len(items)}>")
    return "\n".join(lines), items


def systemone_question(q: Question, remote: bool = False) -> dict[str, Any]:
    """One question in System One form. Remote endpoints get every option described, the local model reads bare names."""
    desc = q.descriptions
    if q.kind == "noul":
        yes = next(n for n in q.names if n.lower() == "yes")
        no = next(n for n in q.names if n.lower() == "no")
        out: dict[str, Any] = {"type": "noul", "instructions": q.question}
        if desc.get(yes) or desc.get(no):
            out["criteria"] = {"true": desc.get(yes) or yes, "false": desc.get(no) or no}
        return out
    if q.kind == "score":
        return {"type": "score", "instructions": q.question,
                "criteria": [f"{n}: {desc[n]}" if n in desc else n for n in q.names]}
    return {"type": "choice", "instructions": q.question,
            "criteria": {n: desc.get(n) or (n if remote else None) for n in q.names}}


def result_from(q: Question, answer: dict[str, Any], provider: str) -> Result:
    """A System One answer as a result over the question's own option names."""
    names = q.names
    if q.kind == "noul":
        p = min(1.0, max(0.0, float(answer["noul"])))
        probs = [p if n.lower() == "yes" else 1.0 - p for n in names]
    elif q.kind == "score":
        raw = answer.get("probabilities") or {}
        keys = sorted(raw, key=float)
        if len(keys) != len(names):
            raise ValueError(f"answer to {q.id!r} has {len(keys)} levels, the question has {len(names)}")
        probs = [float(raw[k]) for k in keys]
    else:
        raw = answer.get("probabilities") or {}
        probs = [float(raw.get(n, 0.0)) for n in names]
        if sum(probs) <= 0 and answer.get("choice") in names:
            probs = [1.0 if n == answer["choice"] else 0.0 for n in names]
    total = sum(probs)
    probs = [x / total for x in probs] if total > 0 else [1.0 / len(names)] * len(names)
    best = max(range(len(names)), key=probs.__getitem__)
    return Result(
        id=q.id, answer=names[best], confidence=round(probs[best], 6),
        probabilities={n: round(x, 6) for n, x in zip(names, probs)},
        expected=round(sum(i * x for i, x in enumerate(probs)), 6) if q.kind == "score" else None,
        provider=provider,
    )


def file_image(path: str) -> dict[str, Any]:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        return {"type": "image", "data": f"data:{mime};base64," + base64.b64encode(f.read()).decode()}
