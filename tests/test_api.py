import pytest
from fastapi.testclient import TestClient

from opendecisions.api import DecisionsRequest, split_context, systemone_question
from opendecisions.onejev.answers import make_choice, make_noul, make_score
from opendecisions.onejev.schema import ChoiceQuestion, NoulQuestion, SystemOneResponse, Usage
from opendecisions.providers import LocalProvider, Provider, ProviderError, Router
from opendecisions.server import create_app


class FakeEngine:
    def __init__(self, sure=0.6):
        self.sure = sure
        self.calls = []

    def _set_media(self, items):
        pass

    def decide(self, request, debias=1, media=None):
        self.calls.append((request, media))
        answers = {}
        for qid, q in request.questions.items():
            if isinstance(q, NoulQuestion):
                answers[qid] = make_noul(0.9 if "urgent" in str(request.state) else 0.2)
            elif isinstance(q, ChoiceQuestion):
                names = list(q.criteria)
                answers[qid] = make_choice(names, [self.sure] + [(1 - self.sure) / (len(names) - 1)] * (len(names) - 1))
            else:
                k = len(q.criteria)
                answers[qid] = make_score(q.criteria, [0.7 if i == 1 else 0.3 / (k - 1) for i in range(k)])
        return SystemOneResponse(model="fake", answers=answers, usage=Usage(input_tokens=10, output_tokens=0)), {}


class FakeRemote(Provider):
    def __init__(self, name="openrouter", accepts_media=False, fail=False):
        self.name, self.model, self.accepts_media, self.fail = name, "typesafe/jev-1.13", accepts_media, False or fail
        self.asked = []

    def with_model(self, model):
        return self

    def systemone(self, state, questions, media):
        if self.fail:
            raise ProviderError("HTTP 402")
        self.asked.append(questions)
        out = {}
        for qid, q in questions.items():
            names = list(q["criteria"]) if q["type"] == "choice" else None
            out[qid] = {"type": "choice", "choice": names[-1], "probabilities": {n: (0.97 if n == names[-1] else 0.03 / (len(names) - 1)) for n in names}}
        return out, {"input_tokens": 5, "output_tokens": 0}


QUESTIONS = [
    {"id": "team", "question": "Which team?", "options": ["billing", "tech", "sales"]},
    {"id": "urgent", "question": "Is it urgent?", "options": ["Yes", "No"]},
    {"id": "mood", "question": "How upset?", "options": ["calm", "annoyed", "angry"], "ordered": True},
]


def test_question_kinds_and_results():
    engine = FakeEngine()
    router = Router(LocalProvider(engine, "onejev-4b"))
    r = router.decide(DecisionsRequest.model_validate({"context": "urgent: charged twice", "questions": QUESTIONS}))
    team, urgent, mood = r.results
    assert (team.answer, team.confidence) == ("billing", 0.6)
    assert urgent.answer == "Yes" and urgent.probabilities == {"Yes": 0.9, "No": 0.1}
    assert mood.answer == "annoyed" and mood.expected == pytest.approx(1.0)
    assert r.model == "onejev-4b" and r.usage.input_tokens == 10
    sent = engine.calls[0][0].questions
    assert sent["team"].type == "choice" and sent["urgent"].type == "noul" and sent["mood"].type == "score"


def test_aliases_and_keyed_questions():
    req = DecisionsRequest.model_validate({"state": "x", "questions": {"a": {"text": "Pick", "answers": {"p": "first", "q": None}}}})
    assert req.context == "x" and req.questions[0].id == "a"
    assert systemone_question(req.questions[0]) == {"type": "choice", "instructions": "Pick", "criteria": {"p": "first", "q": None}}
    assert systemone_question(req.questions[0], remote=True)["criteria"] == {"p": "first", "q": "q"}


def test_bad_questions():
    with pytest.raises(ValueError):
        DecisionsRequest.model_validate({"questions": [{"question": "x", "options": ["only"]}]})
    with pytest.raises(ValueError):
        DecisionsRequest.model_validate({"questions": [{"id": "a", "question": "x", "options": ["p", "q"]},
                                                       {"id": "a", "question": "y", "options": ["p", "q"]}]})


def test_parts_become_placeholders():
    state, media = split_context(["Look:", {"type": "image_url", "image_url": {"url": "https://x/a.png"}},
                                  {"type": "video", "frames": ["https://x/1.jpg", "https://x/2.jpg"], "fps": 1}])
    assert state == "Look:\n<image:1>\n<video:2>"
    assert media == [{"type": "image", "url": "https://x/a.png"},
                     {"type": "video", "frames": ["https://x/1.jpg", "https://x/2.jpg"], "fps": 1.0}]
    assert split_context({"task": "t"}) == ({"task": "t"}, [])
    assert split_context(["a", "b"]) == (["a", "b"], [])


def test_fallback_takes_unsure_answers():
    remote = FakeRemote()
    router = Router(LocalProvider(FakeEngine(sure=0.6), "onejev-4b"), {"openrouter": remote}, fallback="openrouter", threshold=0.8)
    r = router.decide(DecisionsRequest.model_validate({"context": "urgent", "questions": QUESTIONS[:2]}))
    assert [x.provider for x in r.results] == ["openrouter", "local"]
    assert r.results[0].answer == "sales" and list(remote.asked[0]) == ["team"]
    assert r.usage.input_tokens == 15


def test_fallback_skips_media_for_text_only_and_survives_errors():
    router = Router(LocalProvider(FakeEngine(), "onejev-4b"), {"openrouter": FakeRemote()}, fallback="openrouter")
    r = router.decide(DecisionsRequest.model_validate({"context": ["see", {"type": "image", "url": "https://x/a.png"}],
                                                       "questions": QUESTIONS[:1]}))
    assert r.results[0].provider == "local" and "reads text only" in r.notes[0]
    router = Router(LocalProvider(FakeEngine(), "onejev-4b"), {"openrouter": FakeRemote(fail=True)}, fallback="openrouter")
    r = router.decide(DecisionsRequest.model_validate({"context": "x", "questions": QUESTIONS[:1]}))
    assert r.results[0].provider == "local" and "402" in r.notes[0]


def test_model_routing():
    remote = FakeRemote()
    router = Router(LocalProvider(FakeEngine(), "onejev-4b"), {"openrouter": remote})
    assert router.resolve("onejev-4b")[0].local and router.resolve("local")[0].local
    assert router.resolve("openrouter/typesafe/jev-1.13")[0] is remote
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        router.resolve("typesafe")
    with pytest.raises(ValueError):
        router.resolve("gpt-6-luna")


def test_http_endpoints():
    app = create_app(Router(LocalProvider(FakeEngine(), "onejev-4b")), api_key="k")
    c = TestClient(app)
    assert c.post("/v1/decisions", json={"questions": QUESTIONS}).status_code == 401
    h = {"Authorization": "Bearer k"}
    r = c.post("/v1/decisions", json={"context": "urgent", "questions": QUESTIONS}, headers=h)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "decision" and [x["id"] for x in body["results"]] == ["team", "urgent", "mood"]
    assert "expected" not in body["results"][0] and "notes" not in body
    assert c.post("/v1/decisions", json={"questions": []}, headers=h).status_code == 422
    assert c.post("/v1/decisions", json={"model": "nope", "questions": QUESTIONS}, headers=h).status_code == 404
    s = c.post("/v1/systemone", json={"state": "urgent", "questions": {"u": {"type": "noul", "instructions": "Urgent?"}}}, headers=h)
    assert s.status_code == 200 and s.json()["answers"]["u"]["noul"] == 0.9
    assert c.get("/v1/models", headers=h).json()["data"][1]["id"] == "onejev-4b"
