"""Load a OneJev model for the local provider: safetensors weights on PyTorch, or a GGUF file on llama.cpp."""
from __future__ import annotations

import glob
import importlib.util
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

SIZES = ("0.8B", "4B", "9B", "27B", "27B-FP8")
ALIASES = {f"onejev-{s.lower()}": f"OmniJev/OneJev-{s}" for s in SIZES}
DEFAULT_MODEL = "OmniJev/OneJev-4B"
DEFAULT_GGUF = "mradermacher/OneJev-4B-GGUF:Q8_0"

log = logging.getLogger("opendecisions.local")


def model_name(spec: str) -> str:
    """The name the API reports for a model: onejev-4b, onejev-4b-q8_0, or the file or folder name."""
    base, _, quant = spec.partition(":")
    stem = Path(base).name.removesuffix(".gguf")
    for size in sorted(SIZES, key=len, reverse=True):
        if stem.lower().startswith(f"onejev-{size.lower()}"):
            name = f"onejev-{size.lower()}"
            return f"{name}-{quant.lower()}" if quant else name
    return stem.lower() + (f"-{quant.lower()}" if quant else "")


def has_cuda() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return torch.cuda.is_available()


def choose(model: str | None, gguf: str | None, llama_server: str) -> tuple[str, str]:
    """("torch", model) or ("gguf", spec): what `serve` runs when nothing or only part is specified."""
    if gguf:
        return "gguf", gguf
    if model:
        return "torch", ALIASES.get(model, model)
    if has_cuda():
        return "torch", DEFAULT_MODEL
    if shutil.which(llama_server) or Path(llama_server).exists():
        return "gguf", DEFAULT_GGUF
    raise SystemExit("no local backend found. Install llama.cpp (macOS: brew install llama.cpp) to run GGUF files, "
                     "or install PyTorch with a GPU: pip install \"opendecisions[torch]\". "
                     "To route to hosted providers only, run: opendecisions serve --no-local --fallback openrouter")


def _resolve(name: str) -> str:
    if Path(name).exists():
        return name
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    snaps = [s for s in glob.glob(os.path.join(hub, "models--" + name.replace("/", "--"), "snapshots", "*"))
             if glob.glob(os.path.join(s, "*.safetensors"))]
    if snaps and os.environ.get("HF_HUB_OFFLINE") == "1":
        return max(snaps, key=os.path.getmtime)
    return name


def _calibration(repo_or_dir: str) -> Any:
    from .onejev.calibrate import Calibration

    local = Path(repo_or_dir) / "calibration.json"
    if local.exists():
        return Calibration.load(local)
    if Path(repo_or_dir).exists():
        return None
    try:
        from huggingface_hub import hf_hub_download

        return Calibration.load(hf_hub_download(repo_or_dir, "calibration.json"))
    except Exception:
        return None


def load_torch(model: str, device: str = "cuda:0", dtype: str = "bfloat16", media_root: str | None = None,
               max_branch_tokens: int = 32768, max_request_tokens: int = 65536) -> tuple[Any, str]:
    if importlib.util.find_spec("torch") is None:
        raise SystemExit('safetensors weights need PyTorch: pip install "opendecisions[torch]", or run a GGUF file with --gguf')
    from .onejev.mm_engine import MMDecisionEngine, local_model_dir

    path = local_model_dir(_resolve(model))
    config = json.loads((Path(path) / "config.json").read_text())
    kw = dict(device=device, dtype=dtype, calibration=_calibration(path), max_branch_tokens=max_branch_tokens,
              max_request_tokens=max_request_tokens)
    if config.get("vision_config") is not None:
        engine = MMDecisionEngine(path, media_root=media_root, allow_local_paths=media_root is not None,
                                  gpu_preprocess=device.startswith("cuda"), cuda_graphs=device.startswith("cuda"), **kw)
    else:
        from .onejev.engine import DecisionEngine

        engine = DecisionEngine(path, **kw)
    return engine, model_name(model)


def load_gguf(spec: str, llama_server: str = "llama-server", mmproj: str | None = None, tokenizer: str | None = None,
              parallel: int = 2, max_branch_tokens: int = 32768, max_request_tokens: int = 65536) -> tuple[Any, str, Any]:
    from huggingface_hub import snapshot_download

    from .onejev.llamacpp import LlamaCppEngine, LlamaServer, tokenizer_repo

    tok = tokenizer or tokenizer_repo(spec)
    if tok is None:
        raise SystemExit("cannot tell which OneJev size this GGUF file is; pass --tokenizer OmniJev/OneJev-<size>")
    tok_dir = tok if Path(tok).exists() else snapshot_download(tok, allow_patterns=["*.json", "*.jinja", "*.txt"])
    proc = Path(tok_dir) / "processor_config.json"
    image_config = json.loads(proc.read_text()).get("image_processor", {}) if proc.exists() else {}
    server = LlamaServer(spec, binary=llama_server, mmproj=mmproj, ctx=max_branch_tokens, parallel=parallel)
    name = model_name(spec)
    engine = LlamaCppEngine(server.url, tok_dir, calibration=_calibration(tok), name=name,
                            max_branch_tokens=max_branch_tokens, max_request_tokens=max_request_tokens,
                            image_config=image_config)
    return engine, name, server
