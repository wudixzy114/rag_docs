"""Model-agnostic client for the internal OpenAI-compatible LLM gateway.

This is the *only* component in Magnus Lens that talks to the LLM. Everything
else (poller, store, renderer, API) is LLM-free. The gateway is the single
outbound network call the system makes (不出内网).
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import httpx

from ragkb.config import LLMSettings, get_llm_settings

log = logging.getLogger(__name__)

_OPUS_VER_RE = re.compile(r"opus[-\s.]*(\d+)[.\-](\d+)")


def _model_rejects_temperature(model: str) -> bool:
    """Opus 4.7+ dropped the `temperature` parameter: the gateway rejects any
    request that carries it with HTTP 400 '`temperature` is deprecated for this
    model.' Those models must omit the field entirely (there is no temperature
    knob), while every older model still accepts it. Matches the JD gateway
    naming `Claude-Opus-4.7-joybuilder` and later."""
    m = _OPUS_VER_RE.search((model or "").lower())
    if not m:
        return False
    return (int(m.group(1)), int(m.group(2))) >= (4, 7)


class LLMError(RuntimeError):
    """Raised when the gateway cannot produce a completion after retries."""


class LLMQuotaError(LLMError):
    """A model's quota is exhausted (gateway 429 / code 2001 '模型配额已用尽').

    Distinct from a transient failure: retrying the SAME model is pointless, so
    the client immediately advances to the next model in the fallback chain
    rather than burning its retry budget on a model that's out of quota."""


class LLMContentBlockedError(LLMError):
    """The gateway's content-safety filter rejected the request (HTTP 400,
    status FAILED_PRECONDITION, body 'sensitive contain:[...]').

    This is a DETERMINISTIC refusal — the same content will always be blocked, so
    retrying (or advancing the fallback chain, which shares the same filter) is
    pure waste. The client raises this immediately; the caller should pre-scrub
    sensitive tokens (MAC/内网IP/密钥) before sending, or skip the item."""


def _is_quota_body(text: str) -> bool:
    """Detect the gateway's per-model quota-exhausted signature in a 429 body."""
    t = text or ""
    return "配额" in t or '"code":2001' in t or '"code": 2001' in t


def _is_content_blocked(status_code: int, text: str) -> bool:
    """Detect the content-safety rejection (400 + sensitive/FAILED_PRECONDITION)."""
    if status_code != 400:
        return False
    t = text or ""
    return "sensitive contain" in t or "FAILED_PRECONDITION" in t


def _drop_rejected_temperature(payload: dict, status_code: int, body: str) -> bool:
    """Remove ``temperature`` when the gateway explicitly rejects that field.

    Model-name allowlists age badly: a gateway can change a model's accepted
    parameters without changing the public id. The response is authoritative,
    so retry the same request once without the field. Returning False after the
    field has already been removed prevents retry loops.
    """
    text = (body or "").lower()
    if (status_code == 400 and "temperature" in payload
            and "temperature" in text
            and any(token in text for token in ("deprecated", "unsupported", "not support"))):
        payload.pop("temperature", None)
        return True
    return False


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def add(self, other: "LLMUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.calls += other.calls


@dataclass
class LLMResult:
    text: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    model: str = ""
    # Populated by complete_with_tools when the model asks to call tools.
    # Each entry: {"id": str, "name": str, "arguments": dict}.
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = ""


@dataclass
class VisionImage:
    """One image for a multimodal call. Holds the RAW bytes + media type and
    lazily base64-encodes into the block shape the active dialect wants. Built
    from a path via `from_path`, which sniffs the media type from the suffix."""
    data: bytes
    media_type: str = "image/png"

    _SUFFIX_MEDIA = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp",
    }

    @classmethod
    def from_path(cls, path) -> "VisionImage":
        from pathlib import Path
        p = Path(path)
        media = cls._SUFFIX_MEDIA.get(p.suffix.lower(), "image/png")
        return cls(data=p.read_bytes(), media_type=media)

    def _b64(self) -> str:
        import base64
        return base64.standard_b64encode(self.data).decode("ascii")

    def to_anthropic_block(self) -> dict:
        return {"type": "image", "source": {
            "type": "base64", "media_type": self.media_type, "data": self._b64()}}

    def to_openai_block(self) -> dict:
        return {"type": "image_url", "image_url": {
            "url": f"data:{self.media_type};base64,{self._b64()}"}}

    def to_gemini_block(self) -> dict:
        return {"inline_data": {"mime_type": self.media_type, "data": self._b64()}}


class LLMClient:
    """Thin, synchronous OpenAI-compatible chat client with retry/backoff.

    Model-agnostic: the model id is config-driven and can be hot-swapped. A
    weak model degrades *gracefully* because the pipeline's citation check
    (pure code) rejects unsupported claims regardless of model quality.
    """

    def __init__(self, settings: LLMSettings | None = None,
                 client: httpx.Client | None = None) -> None:
        self.settings = settings or get_llm_settings()
        # Explicit per-phase timeouts so no single request can hang the whole run.
        # A hung connect/read is the failure we saw (process frozen with a healthy
        # gateway). connect is short; read is the model's think+generate budget.
        timeout = httpx.Timeout(
            connect=15.0, read=self.settings.timeout_seconds,
            write=30.0, pool=15.0)
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None
        # Thread-local storage for the model in force during a call. The verify
        # stage runs multiple LLM calls concurrently in a ThreadPoolExecutor; a
        # plain instance attribute would race (thread A sets model X, thread B
        # overwrites with model Y before A reads it back).
        self._local = threading.local()
        self.last_model: str = ""
        # Quota exhaustion is stable for the useful lifetime of one worker Job.
        # LLMClient itself has that lifetime, so remember an explicit quota 429
        # and skip that model on every later agent turn. A new Job constructs a
        # new client and may probe again after quota has recovered.
        self._quota_lock = threading.Lock()
        self._quota_failures: dict[str, str] = {}

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _active_model(self) -> str:
        return getattr(self._local, "effective_model", None) or self.settings.model

    def _active_provider(self):
        """The ProviderSpec (endpoint + key + dialect) for the model in force on
        this thread. Resolution is model-driven so a per-call override or a
        fallback-chain model routes to the right endpoint, not just the default."""
        return self.settings.provider_for(self._active_model())

    def _active_is_anthropic(self) -> bool:
        return self._active_provider().dialect == "anthropic"

    def _active_is_gemini(self) -> bool:
        return self._active_provider().dialect == "gemini"

    def _available_models(self, chain: list[str]) -> list[str]:
        with self._quota_lock:
            available = [model for model in chain
                         if model not in self._quota_failures]
            failures = dict(self._quota_failures)
        if available:
            return available
        exhausted = ", ".join(model for model in chain if model in failures)
        raise LLMQuotaError(
            f"all models already marked quota-exhausted for this job: {exhausted}")

    def _mark_quota_exhausted(self, model: str, exc: LLMQuotaError) -> None:
        with self._quota_lock:
            self._quota_failures.setdefault(model, str(exc))

    def _with_model_fallback(self, chain: list[str], fn):
        """Run `fn()` under each model in `chain`, advancing on quota exhaustion.

        A model that returns a quota/429 error is skipped immediately (retrying
        it can't help); the next model in the chain is tried. Any other failure
        propagates as-is (fn already did its own transient retries). Raises
        LLMQuotaError only if EVERY model in the chain is exhausted."""
        last_quota: LLMQuotaError | None = None
        for model in self._available_models(chain):
            self._local.effective_model = model
            try:
                res = fn()
                self.last_model = res.model or model
                return res
            except LLMQuotaError as exc:
                last_quota = exc
                self._mark_quota_exhausted(model, exc)
                log.warning("model %s quota exhausted; advancing fallback chain", model)
                continue
            finally:
                self._local.effective_model = None
        raise last_quota or LLMError("no model available in fallback chain")

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        task: str | None = None,
        model: str | None = None,
        chain: list[str] | None = None,
    ) -> LLMResult:
        """Chat completion. `task` selects a per-task model (see LLMSettings.
        model_for); `model` forces a specific model; `chain` forces an explicit
        fallback chain (e.g. the simple-task chain). On a per-model quota error
        the client walks the fallback chain automatically."""
        chain = [model] if model else (chain or self.settings.chain_for(task))

        def _once() -> LLMResult:
            if self._active_is_anthropic():
                return self._anthropic_messages(
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    tools=None, temperature=temperature, max_tokens=max_tokens)
            if self._active_is_gemini():
                return self._gemini_responses(
                    system=system, user=user, max_tokens=max_tokens)
            payload = {
                "model": self._active_model(),
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            return self._post_chat(payload)

        return self._with_model_fallback(chain, _once)

    def complete_vision(
        self,
        *,
        system: str,
        user: str,
        images: list["VisionImage"],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        task: str | None = None,
        model: str | None = None,
    ) -> LLMResult:
        """Multimodal completion: `user` text + one or more `images` in a single
        user turn. Content blocks are built for whichever dialect the active
        model uses (Anthropic image blocks vs OpenAI image_url data-URIs), so the
        SAME call degrades correctly as the fallback chain advances.

        Quality-first: this is how the pipeline reads source screenshots — the
        model's own vision is far more faithful than the pre-baked OCR. The
        fallback chain is the VISION-ONLY chain (settings.vision_chain): it
        contains vision-capable models exclusively (Claude + Gemini, both
        multimodal), so a quota fallback can never land on a text-only model that
        would silently drop the image. If every vision model is quota-exhausted,
        this raises LLMQuotaError — the caller records the image as unread rather
        than trusting a fabricated transcription."""
        chain = [model] if model else self.settings.vision_chain()

        def _once() -> LLMResult:
            if self._active_is_anthropic():
                content = [img.to_anthropic_block() for img in images]
                content.append({"type": "text", "text": user})
                return self._anthropic_messages(
                    system=system,
                    messages=[{"role": "user", "content": content}],
                    tools=None, temperature=temperature, max_tokens=max_tokens)
            if self._active_is_gemini():
                return self._gemini_responses(
                    system=system, user=user, max_tokens=max_tokens, images=images)
            # Defensive: vision chain is vision-capable by construction; this only
            # fires if a caller forced a text-only model=. Refuse rather than send
            # an image a text model will ignore and hallucinate around.
            raise LLMError(
                f"model {self._active_model()} is not vision-capable; "
                "refusing to send image content")

        return self._with_model_fallback(chain, _once)

    def complete_with_tools(
        self,
        *,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        tool_choice: str = "auto",
        task: str | None = None,
        model: str | None = None,
    ) -> LLMResult:
        """Chat completion that may return tool calls (agentic exploration).

        `messages` is the FULL running transcript (system + user + prior
        assistant-with-tool_calls + tool results). The caller (repodoc.agent)
        drives the loop: append the assistant turn, execute the tool_calls,
        append the tool results, call again. Same retry/backoff + per-model
        quota fallback as complete().
        """
        chain = [model] if model else self.settings.chain_for(task)

        def _once() -> LLMResult:
            if self._active_is_anthropic():
                system, conv = _split_openai_messages(messages)
                return self._anthropic_messages(
                    system=system, messages=conv,
                    tools=_openai_tools_to_anthropic(tools),
                    tool_choice=tool_choice, temperature=temperature,
                    max_tokens=max_tokens)
            payload = {
                "model": self._active_model(),
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = tool_choice
            return self._post_chat(payload)

        return self._with_model_fallback(chain, _once)

    def stream(self, *, messages: list[dict], temperature: float = 0.4,
               max_tokens: int = 2048, task: str | None = None,
               model: str | None = None) -> Iterator[dict]:
        """Streaming chat completion (SSE). Yields incremental deltas:
            {"type": "reasoning", "text": <delta>}   # chain-of-thought (GLM)
            {"type": "content",   "text": <delta>}   # the answer

        Reasoning models stream `reasoning_content` first, then `content`; a
        non-reasoning model just streams `content`. Tolerates the trailing
        usage-only chunk (empty `choices`) and `[DONE]`. No mid-stream retry — a
        stream that fails can't be resumed; but the FIRST model's quota-exhausted
        error is caught before any token is emitted, and the next model in the
        chain is tried, so a spent primary model doesn't kill the answer."""
        chain = [model] if model else self.settings.chain_for(task)
        last_quota: LLMQuotaError | None = None
        for m in self._available_models(chain):
            self._local.effective_model = m
            try:
                self.last_model = m
                yield from self._stream_once(messages, temperature, max_tokens)
                return
            except LLMQuotaError as exc:
                last_quota = exc
                self._mark_quota_exhausted(m, exc)
                log.warning("model %s quota exhausted (stream); trying next", m)
                continue
            finally:
                self._local.effective_model = None
        raise last_quota or LLMError("no model available in fallback chain")

    def _stream_once(self, messages: list[dict], temperature: float,
                     max_tokens: int) -> Iterator[dict]:
        if self._active_is_anthropic():
            system, conv = _split_openai_messages(messages)
            yield from self._anthropic_stream(
                system=system, messages=conv, temperature=temperature,
                max_tokens=max_tokens)
            return
        payload = {
            "model": self._active_model(),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        provider = self._active_provider()
        url = provider.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"
        try:
            while True:
                with self._client.stream("POST", url, json=payload,
                                         headers=headers) as resp:
                    if resp.status_code >= 400:
                        body = resp.read().decode("utf-8", "replace")[:200]
                        if _drop_rejected_temperature(payload, resp.status_code, body):
                            log.info("model %s rejected temperature; retrying stream without it",
                                     self._active_model())
                            continue
                        if resp.status_code == 429 and _is_quota_body(body):
                            raise LLMQuotaError(f"quota exhausted: {body}")
                        raise LLMError(f"gateway {resp.status_code}: {body}")
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except (json.JSONDecodeError, ValueError):
                            continue
                        choices = obj.get("choices") or []
                        if not choices:      # trailing usage-only chunk
                            continue
                        delta = choices[0].get("delta") or {}
                        r = delta.get("reasoning_content")
                        if r:
                            yield {"type": "reasoning", "text": r}
                        c = delta.get("content")
                        if c:
                            yield {"type": "content", "text": c}
                    return
        except httpx.HTTPError as exc:
            raise LLMError(f"stream failed: {exc}") from exc

    # ---- Anthropic Messages API adapter (Claude-* models) ---------------
    # The gateway serves Claude models at a sibling /anthropic/v1/messages path
    # with the Anthropic wire format (system top-level, content blocks, tool_use
    # blocks, content_block_delta SSE). These helpers translate to/from the same
    # LLMResult / delta shape the rest of the code already consumes, so callers
    # (repodoc.agent, qa) are model-agnostic.

    def _anthropic_headers(self) -> dict:
        h = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
        key = self._active_provider().api_key
        if key:
            h["Authorization"] = f"Bearer {key}"
        return h

    def _anthropic_payload(self, *, system, messages, tools, tool_choice,
                           temperature, max_tokens) -> dict:
        payload = {
            "model": self._active_model(),
            "max_tokens": max_tokens,
            "messages": messages,
        }
        # Opus 4.7+ removed `temperature`; sending it (even 0.0) is a hard 400.
        # These models are always deterministic, so omitting it is equivalent.
        if not _model_rejects_temperature(self._active_model()):
            payload["temperature"] = temperature
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = tools
            if tool_choice == "required":
                payload["tool_choice"] = {"type": "any"}
            elif tool_choice == "none":
                payload["tool_choice"] = {"type": "none"}
            # "auto" is the default; omit
        return payload

    def _anthropic_messages(self, *, system, messages, tools=None,
                            tool_choice="auto", temperature=0.0,
                            max_tokens=2048) -> LLMResult:
        payload = self._anthropic_payload(
            system=system, messages=messages, tools=tools,
            tool_choice=tool_choice, temperature=temperature,
            max_tokens=max_tokens)
        url = self._active_provider().base_url + "/messages"
        headers = self._anthropic_headers()
        last_err: Exception | None = None
        for attempt in range(1, self.settings.max_retries + 1):
            try:
                while True:
                    resp = self._client.post(url, json=payload, headers=headers)
                    if _drop_rejected_temperature(payload, resp.status_code, resp.text):
                        log.info("model %s rejected temperature; retrying without it",
                                 self._active_model())
                        continue
                    break
                if resp.status_code == 429 and _is_quota_body(resp.text):
                    # Model out of quota: don't waste retries — surface to the
                    # fallback loop to switch models immediately.
                    raise LLMQuotaError(f"quota exhausted: {resp.text[:200]}")
                if _is_content_blocked(resp.status_code, resp.text):
                    # Content-safety refusal is deterministic — retrying the same
                    # content always 400s. Surface immediately (no retry, no
                    # fallback: the filter is gateway-wide).
                    raise LLMContentBlockedError(f"content blocked: {resp.text[:200]}")
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise LLMError(f"gateway {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                return _anthropic_result(resp.json(), self._active_model())
            except (LLMQuotaError, LLMContentBlockedError):
                raise  # bypass transient-retry; deterministic — no point retrying
            except (httpx.HTTPError, LLMError, KeyError, ValueError) as exc:
                last_err = exc
                wait = min(2 ** attempt, 10)
                log.warning("anthropic call failed (attempt %d/%d): %s; retry in %ds",
                            attempt, self.settings.max_retries, exc, wait)
                if attempt < self.settings.max_retries:
                    time.sleep(wait)
        raise LLMError(f"gateway failed after {self.settings.max_retries} attempts: {last_err}")

    def _anthropic_stream(self, *, system, messages, temperature, max_tokens
                          ) -> Iterator[dict]:
        payload = self._anthropic_payload(
            system=system, messages=messages, tools=None, tool_choice="auto",
            temperature=temperature, max_tokens=max_tokens)
        payload["stream"] = True
        url = self._active_provider().base_url + "/messages"
        headers = self._anthropic_headers()
        try:
            while True:
                with self._client.stream("POST", url, json=payload,
                                         headers=headers) as resp:
                    if resp.status_code >= 400:
                        body = resp.read().decode("utf-8", "replace")[:200]
                        if _drop_rejected_temperature(payload, resp.status_code, body):
                            log.info("model %s rejected temperature; retrying stream without it",
                                     self._active_model())
                            continue
                        if resp.status_code == 429 and _is_quota_body(body):
                            raise LLMQuotaError(f"quota exhausted: {body}")
                        raise LLMError(f"gateway {resp.status_code}: {body}")
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if not data:
                            continue
                        try:
                            obj = json.loads(data)
                        except (json.JSONDecodeError, ValueError):
                            continue
                        # Anthropic streams content_block_delta with {text} for text
                        # blocks; thinking blocks (if enabled) arrive as thinking_delta.
                        if obj.get("type") == "content_block_delta":
                            delta = obj.get("delta") or {}
                            if delta.get("type") == "thinking_delta" and delta.get("thinking"):
                                yield {"type": "reasoning", "text": delta["thinking"]}
                            elif delta.get("text"):
                                yield {"type": "content", "text": delta["text"]}
                    return
        except httpx.HTTPError as exc:
            raise LLMError(f"stream failed: {exc}") from exc

    def _gemini_responses(self, *, system: str, user: str,
                          max_tokens: int = 2048,
                          images: "list[VisionImage] | None" = None) -> LLMResult:
        """Gemini-native `/v1/responses` API (contents/parts). Used for cheap,
        fast tasks (paraphrase) routed to Gemini-Flash, and as a multimodal vision
        fallback. System text is prepended to the user turn (Gemini has no separate
        system role here). Images become inline_data parts. Same retry +
        quota/content-blocked handling as the other dialects."""
        provider = self._active_provider()
        url = provider.base_url.rstrip("/") + "/responses"
        headers = {"Content-Type": "application/json"}
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"
        text = (system + "\n\n" + user) if system else user
        parts: list[dict] = []
        for img in (images or []):
            parts.append(img.to_gemini_block())
        parts.append({"text": text})
        payload = {
            "model": self._active_model(),
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        last_err: Exception | None = None
        for attempt in range(1, self.settings.max_retries + 1):
            try:
                resp = self._client.post(url, json=payload, headers=headers)
                if resp.status_code == 429 and _is_quota_body(resp.text):
                    raise LLMQuotaError(f"quota exhausted: {resp.text[:200]}")
                if _is_content_blocked(resp.status_code, resp.text):
                    raise LLMContentBlockedError(f"content blocked: {resp.text[:200]}")
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise LLMError(f"gateway {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                return _gemini_result(resp.json(), self._active_model())
            except (LLMQuotaError, LLMContentBlockedError):
                raise
            except (httpx.HTTPError, LLMError, KeyError, ValueError) as exc:
                last_err = exc
                wait = min(2 ** attempt, 10)
                log.warning("gemini call failed (attempt %d/%d): %s; retry in %ds",
                            attempt, self.settings.max_retries, exc, wait)
                if attempt < self.settings.max_retries:
                    time.sleep(wait)
        raise LLMError(f"gateway failed after {self.settings.max_retries} attempts: {last_err}")

    def _post_chat(self, payload: dict) -> LLMResult:
        provider = self._active_provider()
        url = provider.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"

        last_err: Exception | None = None
        for attempt in range(1, self.settings.max_retries + 1):
            try:
                while True:
                    resp = self._client.post(url, json=payload, headers=headers)
                    if _drop_rejected_temperature(payload, resp.status_code, resp.text):
                        log.info("model %s rejected temperature; retrying without it",
                                 self._active_model())
                        continue
                    break
                if resp.status_code == 429 and _is_quota_body(resp.text):
                    # Model out of quota: don't retry the same model — surface to
                    # the fallback loop so it switches models immediately.
                    raise LLMQuotaError(f"quota exhausted: {resp.text[:200]}")
                if _is_content_blocked(resp.status_code, resp.text):
                    raise LLMContentBlockedError(f"content blocked: {resp.text[:200]}")
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise LLMError(f"gateway {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                message = choice.get("message", {})
                usage_raw = data.get("usage") or {}
                usage = LLMUsage(
                    prompt_tokens=usage_raw.get("prompt_tokens", 0),
                    completion_tokens=usage_raw.get("completion_tokens", 0),
                    total_tokens=usage_raw.get("total_tokens", 0),
                    calls=1,
                )
                content = message.get("content") or ""
                tool_calls = _parse_tool_calls(message.get("tool_calls"))
                # Dialect fallback: some models (e.g. DeepSeek-V4-Flash) emit tool
                # calls as a JSON block in `content` instead of the native
                # tool_calls field. Recover them, and strip the block from text.
                if not tool_calls:
                    recovered, content = _tool_calls_from_content(content)
                    tool_calls = recovered
                # Reasoning models (e.g. GLM-5.2) put chain-of-thought in
                # `reasoning_content` and the answer in `content`. If the answer
                # is empty but there are no tool calls (e.g. truncated on length),
                # fall back to the reasoning text so the caller gets *something*
                # rather than an empty string.
                if not content and not tool_calls:
                    content = (message.get("reasoning_content") or "").strip()
                return LLMResult(
                    text=content.strip(),
                    usage=usage,
                    model=data.get("model", self.settings.model),
                    tool_calls=tool_calls,
                    finish_reason=choice.get("finish_reason", "") or "",
                )
            except (LLMQuotaError, LLMContentBlockedError):
                raise  # bypass transient-retry; deterministic — no point retrying
            except (httpx.HTTPError, LLMError, KeyError, ValueError) as exc:
                last_err = exc
                wait = min(2 ** attempt, 10)
                log.warning("LLM call failed (attempt %d/%d): %s; retrying in %ds",
                            attempt, self.settings.max_retries, exc, wait)
                if attempt < self.settings.max_retries:
                    time.sleep(wait)
        raise LLMError(f"gateway failed after {self.settings.max_retries} attempts: {last_err}")


def _parse_tool_calls(raw: list | None) -> list[dict]:
    """Normalize OpenAI tool_calls into [{id, name, arguments(dict)}]. Malformed
    JSON arguments degrade to {} rather than crashing the agent loop."""
    if not raw:
        return []
    out: list[dict] = []
    for tc in raw:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        args_raw = fn.get("arguments", "")
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw) if args_raw.strip() else {}
            except (json.JSONDecodeError, ValueError):
                args = {}
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            args = {}
        out.append({
            "id": tc.get("id", ""),
            "name": fn.get("name", ""),
            "arguments": args,
        })
    return out


def _tool_calls_from_content(content: str) -> tuple[list[dict], str]:
    """Recover tool calls emitted as a JSON block in `content` (a dialect used
    by some gateway models instead of the native tool_calls field).

    Looks for a ```json fenced object (or a bare object) containing a
    "tool_calls" list of {name, arguments, id?}. Returns (tool_calls, remaining
    text with the block removed). If nothing parseable is found, returns
    ([], original_content).
    """
    if not content or "tool_calls" not in content:
        return [], content
    import re as _re

    # Candidate JSON blobs: fenced ```json ... ``` first, else the whole string.
    candidates: list[tuple[str, int, int]] = []  # (text, start, end)
    for m in _re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", content, _re.DOTALL):
        candidates.append((m.group(1), m.start(), m.end()))
    if not candidates:
        # bare object spanning the first { to the last }
        lo, hi = content.find("{"), content.rfind("}")
        if 0 <= lo < hi:
            candidates.append((content[lo:hi + 1], lo, hi + 1))

    for blob, start, end in candidates:
        try:
            obj = json.loads(blob)
        except (json.JSONDecodeError, ValueError):
            continue
        raw_calls = obj.get("tool_calls") if isinstance(obj, dict) else None
        if not isinstance(raw_calls, list) or not raw_calls:
            continue
        out: list[dict] = []
        for i, tc in enumerate(raw_calls):
            if not isinstance(tc, dict):
                continue
            name = tc.get("name") or tc.get("function", {}).get("name", "")
            args = tc.get("arguments", tc.get("function", {}).get("arguments", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except (json.JSONDecodeError, ValueError):
                    args = {}
            if not name:
                continue
            out.append({"id": tc.get("id", f"call_{i}"), "name": name,
                        "arguments": args if isinstance(args, dict) else {}})
        if out:
            remaining = (content[:start] + content[end:]).strip()
            return out, remaining
    return [], content


# ---- OpenAI <-> Anthropic translation (for Claude-* models) --------------

def _split_openai_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Convert an OpenAI-style transcript to (system_text, anthropic_messages).

    - system messages are concatenated into the top-level system string.
    - assistant messages with tool_calls -> content blocks incl. tool_use.
    - tool-role messages -> a user message carrying a tool_result block.
    - plain user/assistant text -> {role, content:str}.
    """
    system_parts: list[str] = []
    conv: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            if m.get("content"):
                system_parts.append(m["content"])
            continue
        if role == "tool":
            conv.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id", ""),
                "content": m.get("content") or "",
            }]})
            continue
        if role == "assistant" and m.get("tool_calls"):
            blocks: list[dict] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except (json.JSONDecodeError, ValueError):
                        args = {}
                blocks.append({"type": "tool_use", "id": tc.get("id", ""),
                               "name": fn.get("name", ""), "input": args})
            conv.append({"role": "assistant", "content": blocks})
            continue
        # plain text turn
        conv.append({"role": role, "content": m.get("content") or ""})
    return "\n\n".join(system_parts), conv


def _openai_tools_to_anthropic(tools: list[dict] | None) -> list[dict] | None:
    """OpenAI function tool schema -> Anthropic tool schema."""
    if not tools:
        return None
    out = []
    for t in tools:
        fn = t.get("function", t)
        out.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return out


def _anthropic_result(data: dict, model: str) -> LLMResult:
    """Anthropic Messages response -> LLMResult (text + normalized tool_calls)."""
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in data.get("content", []) or []:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append({"id": block.get("id", ""),
                               "name": block.get("name", ""),
                               "arguments": block.get("input", {}) or {}})
    usage_raw = data.get("usage") or {}
    pt = usage_raw.get("input_tokens", 0)
    ct = usage_raw.get("output_tokens", 0)
    usage = LLMUsage(prompt_tokens=pt, completion_tokens=ct,
                     total_tokens=pt + ct, calls=1)
    return LLMResult(text="".join(text_parts).strip(), usage=usage,
                     model=data.get("model", model), tool_calls=tool_calls,
                     finish_reason=data.get("stop_reason", "") or "")


def _gemini_result(data: dict, model: str) -> LLMResult:
    """Gemini `/v1/responses` response -> LLMResult. Text lives in
    candidates[0].content.parts[*].text; a `thoughtSignature` sibling field is
    ignored. finishReason maps to our finish_reason ('length' when truncated)."""
    text_parts: list[str] = []
    finish = ""
    cands = data.get("candidates") or []
    if cands:
        cand = cands[0]
        finish = str(cand.get("finishReason", "") or "")
        for part in (cand.get("content") or {}).get("parts", []) or []:
            if isinstance(part, dict) and part.get("text"):
                text_parts.append(part["text"])
    # Normalize Gemini's MAX_TOKENS finish reason to our 'length' sentinel so the
    # struct gate's truncation check works uniformly across dialects.
    if finish.upper() == "MAX_TOKENS":
        finish = "length"
    usage_raw = data.get("usageMetadata") or {}
    pt = usage_raw.get("promptTokenCount", 0)
    ct = usage_raw.get("candidatesTokenCount", 0)
    usage = LLMUsage(prompt_tokens=pt, completion_tokens=ct,
                     total_tokens=pt + ct, calls=1)
    return LLMResult(text="".join(text_parts).strip(), usage=usage,
                     model=data.get("modelVersion", model), finish_reason=finish)
