"""Stable failure taxonomy used by manifest, API, retry policy and dashboard."""
from __future__ import annotations

from ragkb.llm.client import LLMContentBlockedError, LLMOutputError, LLMQuotaError


def classify_failure(exc: BaseException) -> tuple[str, bool]:
    text = str(exc).lower()
    if isinstance(exc, LLMContentBlockedError) or "content blocked" in text:
        return "content_blocked", False
    if isinstance(exc, LLMOutputError) or "invalid json" in text:
        return "invalid_model_output", True
    if isinstance(exc, LLMQuotaError) or "quota" in text or "配额" in text:
        return "quota_exhausted", True
    if "429" in text or "rate limit" in text or "限流" in text:
        return "rate_limited", True
    if ("503" in text or "502" in text or "upstream" in text
            or "timeout" in text or "timed out" in text):
        return "gateway_unavailable", True
    if "coverage gap" in text:
        return "completeness_gap", False
    if "truncated" in text:
        return "truncated_output", True
    if "vision" in text or "image" in text or "图片" in text:
        return "vision_failed", True
    return "pipeline_error", True


def sanitize_failure_message(message: str) -> str:
    """Remove sensitive gateway echoes while preserving an actionable category."""
    text = str(message or "")
    if "content blocked" not in text.lower() and "sensitive contain" not in text:
        return text
    categories = []
    for label, cues in (
        ("email", ("邮箱", "email")),
        ("password", ("密码", "password", "passwd")),
        ("network_address", ("MAC地址", "IP地址")),
        ("credential", ("密钥", "token", "secret")),
    ):
        if any(cue.lower() in text.lower() for cue in cues):
            categories.append(label)
    suffix = ",".join(categories) if categories else "sensitive_content"
    return f"gateway content safety blocked the request; categories={suffix}"
