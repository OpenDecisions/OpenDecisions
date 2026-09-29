"""Command line: opendecisions serve | ask | models."""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys


def cmd_serve(args: argparse.Namespace) -> None:
    import atexit

    import uvicorn

    from .local import choose, load_gguf, load_torch
    from .providers import REMOTES, LocalProvider, RemoteProvider, Router, known_remote
    from .server import create_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    remotes: dict[str, RemoteProvider] = {}
    for name in REMOTES:
        p = known_remote(name)
        if p:
            remotes[name] = p
    if args.remote:
        remotes["remote"] = RemoteProvider("remote", args.remote, args.remote_model, os.environ.get("REMOTE_API_KEY"),
                                           accepts_media=True)
    if args.fallback and args.fallback not in remotes:
        hint = f"set {REMOTES[args.fallback][1]}" if args.fallback in REMOTES else "pass --remote URL"
        raise SystemExit(f"fallback {args.fallback!r} is not available: {hint}")

    local = None
    if not args.no_local:
        backend, spec = choose(args.model, args.gguf, args.llama_server)
        if backend == "gguf":
            engine, name, server = load_gguf(spec, llama_server=args.llama_server, mmproj=args.mmproj,
                                             tokenizer=args.tokenizer, parallel=args.parallel,
                                             max_branch_tokens=args.max_branch_tokens,
                                             max_request_tokens=args.max_request_tokens)
            atexit.register(server.stop)

            def stop(signum, frame):
                server.stop()
                raise SystemExit(128 + signum)

            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, stop)
        else:
            engine, name = load_torch(spec, device=args.device, dtype=args.dtype, media_root=args.media_root,
                                      max_branch_tokens=args.max_branch_tokens,
                                      max_request_tokens=args.max_request_tokens)
        local = LocalProvider(engine, args.name or name)
    elif not remotes:
        raise SystemExit("--no-local needs a hosted provider: set OPENROUTER_API_KEY or TYPESAFE_API_KEY, or pass --remote")

    router = Router(local, remotes, fallback=args.fallback, threshold=args.threshold)
    app = create_app(router, api_key=args.api_key or os.environ.get("OPENDECISIONS_SERVER_KEY"))
    logging.getLogger("opendecisions").info("serving %s on http://%s:%d", ", ".join(router.models()), args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def _questions(args: argparse.Namespace) -> list[dict]:
    if len(args.question) != len(args.options):
        raise SystemExit("give each --question its own --options")
    out = []
    for i, (q, o) in enumerate(zip(args.question, args.options)):
        ordered = o.startswith("<")
        out.append({"id": f"q{i + 1}", "question": q, "options": [x.strip() for x in o.lstrip("<").split(",") if x.strip()],
                    "ordered": ordered})
    return out


def cmd_ask(args: argparse.Namespace) -> None:
    from .client import APIError, OpenDecision, image

    if args.file:
        body = json.load(sys.stdin if args.file == "-" else open(args.file))
    else:
        parts: list = [args.context] if args.context else []
        parts += [image(p) for p in args.image]
        body = {"context": parts if args.image else args.context, "questions": _questions(args)}
    body["model"] = args.model or body.get("model") or "auto"
    client = OpenDecision(args.url)
    try:
        r = client.decisions.create(**body)
    except APIError as e:
        raise SystemExit(str(e))
    if args.json:
        print(json.dumps(r.raw, ensure_ascii=False, indent=2))
        return
    width = max(len(x.id) for x in r)
    for x in r:
        print(f"{x.id:<{width}}  {x.answer}  {x.confidence:.2f}  ({x.provider})")
    for n in r.notes:
        print("note:", n, file=sys.stderr)


def cmd_models(args: argparse.Namespace) -> None:
    from .client import OpenDecision

    print("\n".join(OpenDecision(args.url).models()))


def main() -> None:
    p = argparse.ArgumentParser(prog="opendecisions", description="Open models and one server for the Decisions API")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start the server")
    s.add_argument("--model", default=None,
                   help="safetensors model on PyTorch: onejev-0.8b | onejev-4b | onejev-9b | onejev-27b | onejev-27b-fp8, "
                        "a Hugging Face id or a local folder (default: onejev-4b on a CUDA GPU)")
    s.add_argument("--gguf", default=None,
                   help="GGUF model on llama.cpp: a local file or repo:quant, e.g. mradermacher/OneJev-4B-GGUF:Q8_0 "
                        "(default without a CUDA GPU)")
    s.add_argument("--no-local", action="store_true", help="serve no local model, only route to hosted providers")
    s.add_argument("--fallback", default=None,
                   help="provider that answers when the local model is less sure than --threshold: openrouter, "
                        "typesafe, or remote")
    s.add_argument("--threshold", type=float, default=0.8, help="confidence under which answers go to the fallback")
    s.add_argument("--remote", default=None, help="URL of another System One or OpenDecisions server, used as provider `remote`")
    s.add_argument("--remote-model", default="auto", help="model name to request from --remote")
    s.add_argument("--name", default=None, help="model name the API reports for the local model")
    s.add_argument("--api-key", default=None, help="require this bearer key on /v1 requests")
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--dtype", default="bfloat16")
    s.add_argument("--media-root", default=None, help="also accept image paths inside this folder (URLs and data always work)")
    s.add_argument("--llama-server", default="llama-server", help="llama-server binary for GGUF models")
    s.add_argument("--mmproj", default=None, help="vision projector GGUF (found next to a local file or in the repo)")
    s.add_argument("--tokenizer", default=None, help="OmniJev/OneJev-<size> tokenizer for a GGUF file with an unusual name")
    s.add_argument("--parallel", type=int, default=2, help="llama-server slots")
    s.add_argument("--max-branch-tokens", type=int, default=32768, help="longest context plus one question")
    s.add_argument("--max-request-tokens", type=int, default=65536, help="longest whole request")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    a = sub.add_parser("ask", help="ask a running server",
                       description='opendecisions ask "My card was charged twice." -q "Which team?" -o billing,tech,sales '
                                   '-q "How upset is the customer?" -o "<calm,annoyed,angry"   (a leading < marks an '
                                   'ordered scale)')
    a.add_argument("context", nargs="?", default=None, help="context text")
    a.add_argument("-q", "--question", action="append", default=[], help="a question (repeatable)")
    a.add_argument("-o", "--options", action="append", default=[], help="its options, comma separated (repeatable)")
    a.add_argument("-i", "--image", action="append", default=[], help="an image file or URL to add to the context")
    a.add_argument("-f", "--file", default=None, help="a request JSON file, or - for stdin")
    a.add_argument("-m", "--model", default=None)
    a.add_argument("--json", action="store_true", help="print the whole response")
    a.add_argument("--url", default=os.environ.get("OPENDECISIONS_BASE_URL", "http://localhost:8000"))
    a.set_defaults(func=cmd_ask)

    m = sub.add_parser("models", help="list the models of a running server")
    m.add_argument("--url", default=os.environ.get("OPENDECISIONS_BASE_URL", "http://localhost:8000"))
    m.set_defaults(func=cmd_models)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
