"""Configuration for the RAG knowledge-base pipeline.

Two settings groups:
- `LLMSettings` — the internal 京东 OpenAI-compatible gateway. Vendored verbatim
  from magnus-lens: model-agnostic, dual-dialect (OpenAI + Anthropic), per-model
  quota fallback, per-task routing. The pipeline's parallelism relies on the
  client being thread-safe (thread-local active model), so this class is the
  contract the vendored `llm/client.py` depends on — keep it in lockstep.
- `Settings` — this pipeline's own paths and knobs (input/output dirs, worker
  concurrency, gate policy).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env once at import so both RAGKB_* and the gateway's JD_LLM_* vars are
# visible to the two settings classes below.
load_dotenv()

# Project root: config.py is at <root>/src/ragkb/config.py, so parents[2] is
# <root>. Input/output roots anchor HERE, not the process cwd — the CLI, the API
# server and the test suite all run from different working directories, and a
# cwd-relative default would silently resolve to the wrong place.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProviderSpec:
    """One LLM provider: an endpoint + credentials + wire dialect + the models
    it serves. A single deployment can span several — e.g. the JD gateway
    exposes Claude models on an Anthropic-dialect path and everything else on an
    OpenAI-dialect path, so it's modelled as TWO providers over one host.

    `dialect` selects the request/response wire format (and therefore which URL
    suffix + payload shape the client uses): 'anthropic' -> `<base_url>/messages`
    with content blocks; 'openai' -> `<base_url>/chat/completions`. `base_url` is
    the dialect root already (the anthropic provider's base_url ends in
    `/anthropic/v1`), so the client only appends the method path."""
    name: str
    base_url: str
    api_key: str
    dialect: str = "openai"                 # "openai" | "anthropic"
    models: list[str] = field(default_factory=list)

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)


class Settings(BaseSettings):
    """Pipeline paths and runtime knobs. Env prefix RAGKB_."""

    model_config = SettingsConfigDict(env_prefix="RAGKB_", extra="ignore")

    input_root: Path = Path("input")
    output_root: Path = Path("output")
    cache_root: Path = Path("output/.cache")
    # Total in-flight LLM calls across the whole run. The gateway meters a
    # per-request RATE limit (429 "请求次数超过模型限流阈值") that is DISTINCT from
    # per-model quota — the client only falls back models on quota, not rate — so
    # a high worker count just burns retry budget. verify.py caps at 4; we keep
    # the same discipline. doc-pool × inner-batch concurrency compounds, so this
    # is the single global ceiling both layers share.
    max_workers: int = Field(default=4, ge=1, le=16)
    # Semantic gate (Layer 2) policy when a strong-model reviewer can't return a
    # verdict for an item after the repair attempt:
    #   fail_closed — drop the item (SAFE DEFAULT: never publish unreviewed data).
    #   keep        — publish it flagged for human review (higher recall, riskier).
    semantic_gate_policy: str = "fail_closed"
    # Cross-review model: the reviewer (Layer 2) SHOULD differ from the extractor
    # for an independent perspective. Empty => reuse the `verify` task route.
    reviewer_model: str = ""
    # Main production path stays source-faithful: generated query paraphrases are
    # disabled unless an operator explicitly opts in for an experiment.
    enable_paraphrases: bool = False
    # A failed semantic review gets one source-grounded regeneration. More retries
    # quickly multiply both generation and review cost with diminishing returns.
    review_regeneration_attempts: int = Field(default=1, ge=0, le=3)

    @property
    def input_dir(self) -> Path:
        return self._anchor(self.input_root)

    @property
    def output_dir(self) -> Path:
        return self._anchor(self.output_root)

    @property
    def cache_dir(self) -> Path:
        return self._anchor(self.cache_root)

    @staticmethod
    def _anchor(value: Path) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (_PROJECT_ROOT / p).resolve()


class LLMSettings(BaseSettings):
    """Internal 京东 OpenAI-compatible gateway. Model-agnostic / hot-swappable.

    Uses the JD_LLM_* names already present in .env rather than the RAGKB_
    prefix, so ops can reuse the department-standard variables.

    Multi-model: the gateway meters quota PER MODEL (a 429 `{"code":2001,
    "message":"模型配额已用尽"}` means that one model is exhausted, not the
    account). So `model` is the primary, and `fallback_models` is an ordered
    chain the client walks when the primary returns a quota/429 error — the fleet
    keeps running on the next model when one is spent. Per-task overrides
    (extract vs review vs classify) route through `model_for(task)`.

    Multi-provider: `providers` resolves each model to a `ProviderSpec`
    (endpoint + key + dialect). With no `JD_LLM_PROVIDERS` env set it auto-
    synthesizes two providers from this single gateway (an Anthropic-dialect one
    at `<root>/anthropic/v1` and an OpenAI-dialect one at `<root>/v1`, both using
    `api_key`), so existing single-gateway deployments need zero new config.
    """

    model_config = SettingsConfigDict(extra="ignore")

    base_url: str = Field(default="http://llm-gw.jd.local/v1", alias="JD_LLM_BASE_URL")
    api_key: str = Field(default="", alias="JD_LLM_API_KEY")
    model: str = Field(default="Gemini-3.1-Pro-Preview-joybuilder", alias="XIAOSHU_MODEL")
    # Ordered fallback chain, comma-separated in env. Tried in order after the
    # primary model returns a quota/429 error. Default keeps the fleet alive when
    # any single model is exhausted (the primary is prepended automatically, so
    # listing it here is harmless).
    fallback_models_raw: str = Field(
        default="Gemini-3.1-Pro-Preview-joybuilder,Gemini-3-Flash-Preview-joybuilder,Claude-Sonnet-4.6-joybuilder",
        alias="JD_LLM_FALLBACK_MODELS")
    # Optional multi-provider registry, JSON array in env. Each entry:
    # {"name","base_url","api_key","dialect":"openai"|"anthropic","models":[...]}.
    # When empty, providers are auto-synthesized from base_url/api_key (see
    # `providers`). A model not claimed by any provider falls back to the dialect
    # heuristic, so this only needs entries for models on a DIFFERENT endpoint/key.
    providers_raw: str = Field(default="", alias="JD_LLM_PROVIDERS")
    # Optional per-task model routing, JSON object in env, e.g.
    # {"classify":"DeepSeek-V4-Pro-joybuilder","review":"Claude-Sonnet-4.6-joybuilder"}.
    # A task not listed here uses `model`. Tasks: classify|extract|sop|
    # paraphrase|review|aggregate.
    task_models_raw: str = Field(default="", alias="JD_LLM_TASK_MODELS")
    # Vision-capable models ONLY, comma-separated. The vision fallback chain is
    # drawn from HERE, never from the general chain — a text-only model (e.g.
    # DeepSeek) would silently drop images and hallucinate a transcription. Empty
    # => derive from the general chain by keeping only Claude (anthropic) models.
    vision_models_raw: str = Field(default="", alias="JD_LLM_VISION_MODELS")
    # Simple-task fallback chain (paraphrase etc.): fast/cheap models first, with
    # a capable backstop. e.g. Gemini-3-Flash → Gemini-3.1-Pro → Sonnet.
    simple_models_raw: str = Field(default="", alias="JD_LLM_SIMPLE_MODELS")
    timeout_seconds: float = Field(default=90.0, alias="JD_LLM_TIMEOUT")
    max_retries: int = Field(default=3, alias="JD_LLM_MAX_RETRIES")
    max_concurrency: int = Field(default=4, ge=1, le=16,
                                 alias="JD_LLM_MAX_CONCURRENCY")

    @staticmethod
    def model_is_anthropic(model: str) -> bool:
        """Claude models on this gateway speak the Anthropic Messages API at a
        sibling /anthropic/v1 path, NOT the OpenAI /v1/chat/completions path.
        Static so the client can classify a per-call OVERRIDE model, not just
        the configured default."""
        return model.lower().startswith("claude")

    @staticmethod
    def model_is_gemini(model: str) -> bool:
        """Gemini models speak the Gemini-native `/v1/responses` API (contents/
        parts), NOT OpenAI or Anthropic. Routed to the 'gemini' dialect."""
        return model.lower().startswith("gemini")

    @property
    def is_anthropic(self) -> bool:
        return self.model_is_anthropic(self.model)

    @property
    def anthropic_base(self) -> str:
        """Sibling `/anthropic/v1` base derived from the OpenAI base_url."""
        root = self.base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        return root + "/anthropic/v1"

    def _synthesized_providers(self) -> list[ProviderSpec]:
        """The two providers implied by a single OpenAI-compatible gateway that
        also serves Claude on a sibling `/anthropic/v1` path. Shared host + key;
        they differ only in dialect + URL root. `models=[]` means "claims no
        model explicitly" — resolution falls through to the dialect heuristic."""
        return [
            ProviderSpec(name="anthropic", base_url=self.anthropic_base,
                         api_key=self.api_key, dialect="anthropic"),
            ProviderSpec(name="gemini", base_url=self.base_url.rstrip("/"),
                         api_key=self.api_key, dialect="gemini"),
            ProviderSpec(name="openai", base_url=self.base_url.rstrip("/"),
                         api_key=self.api_key, dialect="openai"),
        ]

    @property
    def providers(self) -> list[ProviderSpec]:
        """Resolved provider registry. Parses `JD_LLM_PROVIDERS` (JSON array) when
        set; otherwise auto-synthesizes the two single-gateway providers. Malformed
        JSON degrades to synthesis rather than crashing boot (mirrors
        `_task_models`)."""
        raw = (self.providers_raw or "").strip()
        if not raw:
            return self._synthesized_providers()
        try:
            import json
            obj = json.loads(raw)
        except (ValueError, TypeError):
            return self._synthesized_providers()
        if not isinstance(obj, list):
            return self._synthesized_providers()
        out: list[ProviderSpec] = []
        for entry in obj:
            if not isinstance(entry, dict) or not entry.get("base_url"):
                continue
            dialect = str(entry.get("dialect") or "openai").lower()
            if dialect not in ("openai", "anthropic", "gemini"):
                dialect = "openai"
            models = entry.get("models") or []
            out.append(ProviderSpec(
                name=str(entry.get("name") or dialect),
                base_url=str(entry["base_url"]).rstrip("/"),
                api_key=str(entry.get("api_key") or self.api_key),
                dialect=dialect,
                models=[str(m) for m in models] if isinstance(models, list) else [],
            ))
        # A registry that parsed to nothing usable is no registry — synthesize so
        # the client always has a resolvable provider.
        return out or self._synthesized_providers()

    def provider_for(self, model: str) -> ProviderSpec:
        """Resolve a model to its provider. TOTAL — never raises (a chain may
        include a model no provider lists explicitly): (1) explicit
        `provider.models` membership; (2) dialect heuristic → first provider of
        the matching dialect; (3) first provider as a last resort."""
        providers = self.providers
        for p in providers:
            if model in p.models:
                return p
        if self.model_is_anthropic(model):
            want = "anthropic"
        elif self.model_is_gemini(model):
            want = "gemini"
        else:
            want = "openai"
        for p in providers:
            if p.dialect == want:
                return p
        return providers[0]

    @property
    def fallback_chain(self) -> list[str]:
        """Ordered, de-duped model chain to try: primary first, then the
        configured fallbacks (skipping the primary if it repeats)."""
        chain = [self.model]
        for m in self.fallback_models_raw.split(","):
            m = m.strip()
            if m and m not in chain:
                chain.append(m)
        return chain

    def _task_models(self) -> dict[str, str]:
        raw = (self.task_models_raw or "").strip()
        if not raw:
            return {}
        try:
            import json
            obj = json.loads(raw)
            return {str(k): str(v) for k, v in obj.items()} if isinstance(obj, dict) else {}
        except (ValueError, TypeError):
            return {}

    def model_for(self, task: str | None) -> str:
        """Primary model for a task (falls back to the global `model`). The
        client still walks `fallback_chain` from this model on a quota error."""
        if not task:
            return self.model
        configured = self._task_models()
        if task in configured:
            return configured[task]
        if task in {"classify", "paraphrase"}:
            return "Gemini-3-Flash-Preview-joybuilder"
        if task in {"extract", "sop", "review", "aggregate"}:
            return "Gemini-3.1-Pro-Preview-joybuilder"
        return self.model

    def chain_for(self, task: str | None) -> list[str]:
        """Fallback chain to try for a task: the task's primary first, then any
        configured fallbacks not already tried."""
        primary = self.model_for(task)
        chain = [primary]
        for m in self.fallback_chain:
            if m not in chain:
                chain.append(m)
        return chain

    def _is_vision_capable(self, model: str) -> bool:
        """Claude and Gemini are multimodal; DeepSeek etc. are text-only."""
        return self.model_is_anthropic(model) or self.model_is_gemini(model)

    def vision_chain(self) -> list[str]:
        """Fallback chain for VISION calls — vision-capable models ONLY (Claude +
        Gemini, both multimodal). Drawn from JD_LLM_VISION_MODELS if set, else the
        general chain filtered to vision-capable models. Never includes a text-only
        model, so a multimodal call can't silently degrade to one that drops the
        image and fabricates a transcription."""
        raw = (self.vision_models_raw or "").strip()
        if raw:
            models = [m.strip() for m in raw.split(",") if m.strip()]
        else:
            models = [m for m in self.fallback_chain if self._is_vision_capable(m)]
        seen, out = set(), []
        for m in models:
            if self._is_vision_capable(m) and m not in seen:
                seen.add(m)
                out.append(m)
        return out or [self.model]

    def simple_chain(self) -> list[str]:
        """Fallback chain for SIMPLE tasks (paraphrase): fast/cheap models first
        with a capable backstop. From JD_LLM_SIMPLE_MODELS, else falls back to the
        general chain. Lets cheap-model work run on a different model (and thus a
        separate quota pool) than the Opus-heavy complex tasks."""
        raw = (self.simple_models_raw or "").strip()
        if raw:
            models = [m.strip() for m in raw.split(",") if m.strip()]
        else:
            models = ["Gemini-3-Flash-Preview-joybuilder",
                      "Gemini-3.1-Pro-Preview-joybuilder", *self.fallback_chain]
        seen, out = set(), []
        for m in models:
            if m and m not in seen:
                seen.add(m)
                out.append(m)
        return out or [self.model]


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_llm_settings() -> LLMSettings:
    return LLMSettings()
