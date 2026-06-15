"""Provider-agnostic LLM client (OpenAI-compatible chat completions, httpx).

Used by Experiment E (Stage 4 self-vs-cross critique). Supports OpenAI and any
OpenAI-compatible endpoint (DeepInfra, DeepSeek). Every request and response is
logged in full to a JSONL interaction log so the whole run is auditable and
reconstructable, and so a crashed run can be inspected before resuming.

Design notes
------------
* Non-streaming requests keep the code simple and let us cleanly separate a
  reasoning model's hidden ``reasoning_content`` from its final ``content`` —
  we always judge / store only the final ``content``.
* The client is safe to share across threads: the usage counters are guarded by
  a lock, and httpx.Client is thread-safe for concurrent requests.
* Reasoning models (DeepSeek-R1, gpt-oss, Qwen3-Thinking) take an explicit
  ``reasoning_effort`` where the provider supports it; temperature is omitted
  for OpenAI o-series style models that reject it.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

# Load .env so the API keys are available when the client is constructed.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:  # python-dotenv is a project dependency; ignore if absent
    pass


# ---------------------------------------------------------------------------
# Interaction logger — records every request/response pair as JSONL
# ---------------------------------------------------------------------------


class InteractionLogger:
    """Append-only JSONL log of every LLM interaction (full prompt + response).

    Thread-safe. One JSON object per line with the complete messages sent and
    the complete content received, plus usage, latency and any error. This is
    the audit trail requested for the experiment: nothing is summarised away.
    """

    def __init__(self, path: Path | str | None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict) -> None:
        if not self.path:
            return
        record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


_NULL_LOGGER = InteractionLogger(None)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def _is_openai_reasoning_model(model: str) -> bool:
    m = model.lower()
    return m.startswith(("o1", "o3", "o4")) or "gpt-5-nano" in m


@dataclass
class LLMClient:
    """OpenAI-compatible chat completions client with full logging."""

    provider: str
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 1024
    max_retries: int = 4
    timeout: float = 180.0
    reasoning_effort: str | None = None  # "low"|"medium"|"high" where supported
    logger: InteractionLogger = field(default=_NULL_LOGGER)

    _client: httpx.Client = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _call_count: int = field(default=0, init=False, repr=False)
    _prompt_tokens: int = field(default=0, init=False, repr=False)
    _completion_tokens: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = httpx.Client(timeout=self.timeout)

    # -- public API --------------------------------------------------------

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        task_key: str | None = None,
        task_kind: str | None = None,
    ) -> str:
        """Send a chat completion and return the final assistant content string.

        ``task_key`` / ``task_kind`` are recorded in the interaction log so each
        call can be traced back to the matrix cell / judgement that produced it.
        """
        resp = self.chat_completion(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            task_key=task_key,
            task_kind=task_kind,
        )
        try:
            return resp["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return ""

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        task_key: str | None = None,
        task_kind: str | None = None,
    ) -> dict:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        is_openai_reasoning = (
            self.provider == "openai" and _is_openai_reasoning_model(self.model)
        )
        temp = temperature if temperature is not None else self.temperature
        mtok = max_tokens if max_tokens is not None else self.max_tokens

        payload: dict = {"model": self.model, "messages": messages}
        if not is_openai_reasoning:
            payload["temperature"] = temp
        if self.provider == "openai":
            payload["max_completion_tokens"] = mtok
        else:
            payload["max_tokens"] = mtok
        if seed is not None:
            payload["seed"] = seed
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            t0 = time.monotonic()
            try:
                r = self._client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                result = r.json()
                elapsed_ms = (time.monotonic() - t0) * 1000.0

                usage = result.get("usage", {}) or {}
                content = ""
                reasoning = ""
                try:
                    msg = result["choices"][0]["message"]
                    content = msg.get("content") or ""
                    reasoning = msg.get("reasoning_content") or ""
                except (KeyError, IndexError, TypeError):
                    pass

                with self._lock:
                    self._call_count += 1
                    self._prompt_tokens += usage.get("prompt_tokens", 0)
                    self._completion_tokens += usage.get("completion_tokens", 0)

                self.logger.log({
                    "event": "llm_call",
                    "task_kind": task_kind,
                    "task_key": task_key,
                    "provider": self.provider,
                    "model": self.model,
                    "attempt": attempt,
                    "latency_ms": round(elapsed_ms, 1),
                    "params": {
                        "temperature": payload.get("temperature"),
                        "seed": seed,
                        "reasoning_effort": self.reasoning_effort,
                        "max_tokens": mtok,
                    },
                    "messages": messages,
                    "response": content,
                    "reasoning_content": reasoning or None,
                    "usage": usage,
                    "success": True,
                })
                return result

            except httpx.HTTPStatusError as e:
                last_error = e
                status = e.response.status_code if e.response is not None else None
                body = e.response.text[:500] if e.response is not None else ""
                retryable = status == 429 or (status is not None and status >= 500)
                self.logger.log({
                    "event": "llm_error",
                    "task_kind": task_kind,
                    "task_key": task_key,
                    "provider": self.provider,
                    "model": self.model,
                    "attempt": attempt,
                    "status": status,
                    "error": str(e),
                    "body": body,
                    "success": False,
                })
                if retryable and attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                else:
                    raise
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
                last_error = e
                self.logger.log({
                    "event": "llm_error",
                    "task_kind": task_kind,
                    "task_key": task_key,
                    "provider": self.provider,
                    "model": self.model,
                    "attempt": attempt,
                    "error": repr(e),
                    "success": False,
                })
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                else:
                    raise

        raise last_error  # type: ignore[misc]

    @property
    def stats(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "calls": self._call_count,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
        }

    def close(self) -> None:
        if self._client is not None:
            self._client.close()


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

PROVIDERS: dict[str, dict] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "deepinfra": {
        "base_url": "https://api.deepinfra.com/v1/openai",
        "api_key_env": "DEEPINFRA_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
}


def make_client(
    provider: str,
    model: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    reasoning_effort: str | None = None,
    logger: InteractionLogger | None = None,
) -> LLMClient:
    """Construct an LLMClient for a registered provider."""
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider {provider!r}; known: {list(PROVIDERS)}"
        )
    cfg = PROVIDERS[provider]
    api_key = os.environ.get(cfg["api_key_env"], "")
    if not api_key:
        raise EnvironmentError(
            f"{cfg['api_key_env']} not set in environment/.env for "
            f"provider {provider!r}"
        )
    return LLMClient(
        provider=provider,
        base_url=cfg["base_url"],
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        logger=logger or _NULL_LOGGER,
    )
