# OpenDecisions

<p>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-D22128?style=flat-square&logo=apache&logoColor=white"></a>
  <a href="https://huggingface.co/collections/OmniJev/onejev"><img alt="Models" src="https://img.shields.io/badge/Models-OneJev-FFD21E?style=flat-square&logo=huggingface&logoColor=black"></a>
  <a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white"></a>
</p>

Open models and one server for the Decisions API. Send text, images or video with a few questions that have fixed
options, and get back the chosen option and a calibrated probability for every option. The default model is OneJev,
which answers all questions of a request in one forward pass and generates no text. Hosted decision models such as Jev
plug in behind the same endpoint.

```bash
pip install opendecisions
opendecisions serve
```

`serve` runs OneJev-4B: on PyTorch when there is a CUDA GPU, otherwise on llama.cpp (on macOS: `brew install llama.cpp`).
The weights download on first start. Then, in another terminal:

```bash
opendecisions ask "I was charged twice for my March invoice and I need the money back today." \
  -q "Which team should handle this?" -o billing,tech,sales \
  -q "Is this urgent?" -o yes,no \
  -q "How upset is the customer?" -o "<calm,annoyed,angry"
```

```text
q1  billing  0.90  (local)
q2  yes  0.73  (local)
q3  angry  0.61  (local)
```

A leading `<` marks options that form an ordered scale.

## Python

```python
from opendecisions import OpenDecision, image

client = OpenDecision()

r = client.decisions.create(
    context=["The agent's task: pay the open invoice from ACME.", image("screen.png")],
    questions=[
        {"id": "done", "question": "Has the invoice been paid?", "options": ["yes", "no"]},
        {"id": "next", "question": "What should the agent do next?", "options": ["click", "type", "scroll", "stop"]},
        {"id": "progress", "question": "How far along is the task?",
         "options": ["not started", "halfway", "almost done", "done"], "ordered": True},
    ],
)

r["next"].answer           # the most likely option
r["next"].confidence       # its probability
r["next"].probabilities    # one probability per option
r["progress"].expected     # expected position on the scale, 0 to 3
```

The client has no dependencies beyond the standard library. `image()` takes a file, a URL or a data URI, and
`video(frames, fps)` takes a list of frames.

## HTTP

`POST /v1/decisions`

```json
{
  "model": "auto",
  "context": "The customer wrote: I was charged twice for my March invoice and I need the money back today.",
  "questions": [
    {"id": "team", "question": "Which team should handle this?", "options": ["billing", "tech", "sales"]},
    {"id": "urgent", "question": "Is this urgent?", "options": ["yes", "no"]},
    {"id": "mood", "question": "How upset is the customer?", "options": ["calm", "annoyed", "angry"], "ordered": true}
  ]
}
```

```json
{
  "id": "dec_bd163e0bc81344e789e31ba7",
  "object": "decision",
  "created": 1790710266,
  "model": "onejev-4b-q8_0",
  "results": [
    {"id": "team", "answer": "billing", "confidence": 0.931785,
     "probabilities": {"billing": 0.931785, "tech": 0.024601, "sales": 0.043614}, "provider": "local"},
    {"id": "urgent", "answer": "yes", "confidence": 0.748659,
     "probabilities": {"yes": 0.748659, "no": 0.251341}, "provider": "local"},
    {"id": "mood", "answer": "angry", "confidence": 0.558194,
     "probabilities": {"calm": 0.09354, "annoyed": 0.348266, "angry": 0.558194}, "expected": 1.464653, "provider": "local"}
  ],
  "usage": {"input_tokens": 223, "output_tokens": 3}
}
```

`context` is a string, any JSON value, or a list of parts. A part is text, an image (`{"type": "image", "url" | "data" |
"path": ...}`, or OpenAI's `image_url` and `input_image` forms) or a video (`{"type": "video", "frames": [...], "fps": 2}`).
`options` is a list of names, or an object that maps each name to a description. Questions whose options are yes and no
are read as yes-or-no questions. `"ordered": true` adds `expected`, the probability-weighted position on the scale.
`questions` may also be an object keyed by id. `state`, `answers` and `text` are accepted for `context`, `options` and
`question`.

The server also answers TypeSafe's System One API on `POST /v1/systemone`, so the official `typesafe-sdk` works with
`TYPESAFE_BASE_URL=http://localhost:8000`.

## Models

| Model | Run it with |
|---|---|
| OneJev-0.8B, 4B, 9B, 27B, 27B-FP8 | `opendecisions serve --model onejev-9b` (PyTorch, CUDA GPU) |
| OneJev GGUF | `opendecisions serve --gguf mradermacher/OneJev-9B-GGUF:Q8_0` (llama.cpp, any machine) |

OneJev reads screenshots, photos, video and text. Video needs the PyTorch backend. On one H200, OneJev-4B answers ten
questions about one screenshot in 104 ms. Accuracy and calibration are on the
[OneJev page](https://github.com/OmniJev/OneJev).

## Hosted providers and fallback

With `OPENROUTER_API_KEY` or `TYPESAFE_API_KEY` set, the server also routes to Jev. A request picks a provider with
`model`:

| `model` | Answered by |
|---|---|
| `auto` | the local model, then the fallback for answers under the threshold |
| `local`, `onejev-4b` | the local model |
| `openrouter`, `openrouter/typesafe/jev-1.13` | Jev through OpenRouter |
| `typesafe`, `typesafe/jev-latest` | Jev through TypeSafe |
| `remote` | another OpenDecisions or System One server given with `--remote URL` |

```bash
opendecisions serve --fallback openrouter --threshold 0.8
```

With a fallback, `auto` sends only the questions the local model is less sure about, and each result names the
provider that answered it. Jev reads text only, so answers about images and video stay local. `--no-local` serves no
model and only routes.

## License

Apache-2.0. The OneJev weights are Apache-2.0 as well.
