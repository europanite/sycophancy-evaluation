from __future__ import annotations

import os
from typing import Any

import requests

_session = requests.Session()


def get_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")


def get_default_model() -> str:
    return os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")


def get_connect_timeout_seconds() -> float:
    return float(os.getenv("OLLAMA_CONNECT_TIMEOUT_SECONDS", "10"))


def get_read_timeout_seconds() -> float:
    # Local inference can exceed two minutes on a cold/contended GPU. The old
    # hard-coded 120 s timeout caused an otherwise valid benchmark to abort.
    return float(os.getenv("OLLAMA_READ_TIMEOUT_SECONDS", "600"))


def get_max_tokens() -> int:
    # SycoBench-600's released evaluation runner uses max_tokens=128.
    return int(os.getenv("OLLAMA_MAX_TOKENS", "128"))


def get_keep_alive() -> str:
    # Ollama defaults to 5m. A benchmark may contain many sequential calls, so
    # keep the selected model resident for the duration of a typical run.
    return os.getenv("OLLAMA_KEEP_ALIVE", "30m")


def call_ollama_messages(
    messages: list[dict[str, str]],
    model: str | None = None,
    *,
    temperature: float = 0,
) -> str:
    """Send exactly the supplied conversation to Ollama.

    sycophancy-evaluation deliberately does not inject an evaluation-oriented system message.
    The default paper protocol contains only user/assistant turns, matching the
    released SycoBench-600 interaction structure.

    Transport/runtime settings are kept outside the experimental prompt:
    - temperature defaults to 0;
    - generation is capped at 128 tokens by default, matching the public
      SycoBench-600 runner;
    - the read timeout is configurable and defaults to 600 seconds;
    - keep_alive avoids needless model reloads during a multi-call benchmark.
    """
    selected_model = model or get_default_model()
    payload: dict[str, Any] = {
        "model": selected_model,
        "messages": messages,
        "stream": False,
        "keep_alive": get_keep_alive(),
        "options": {
            "temperature": temperature,
            "num_predict": get_max_tokens(),
        },
    }
    timeout = (get_connect_timeout_seconds(), get_read_timeout_seconds())
    response = _session.post(f"{get_base_url()}/api/chat", json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    message = data.get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("Ollama response missing message.content")
    return content
