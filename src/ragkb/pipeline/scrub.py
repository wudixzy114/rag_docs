"""Sensitive-content scrubbing — applied BEFORE sending to the gateway.

Two reasons this runs pre-send, not just at export:
1. The gateway's content-safety filter hard-rejects (400 FAILED_PRECONDITION) any
   request containing MAC addresses / internal IPs / credential tokens. Scrubbing
   first is what lets those sections be processed at all.
2. A diagnostic KB should never surface internal MAC/IP/keys to end users anyway,
   so redacting at the source keeps every downstream artifact clean.

We redact credential-shaped and machine-identifier tokens, but deliberately KEEP
internal hostnames/URLs (e.g. storage.jd.local upgrade scripts) — those are often
the legitimate, actionable content of an answer.

Placeholders are human-readable ([MAC]/[IP]/[REDACTED]) so a reader still sees
that a value was there, and diagnostic meaning ("检查网卡 MAC") survives.
"""
from __future__ import annotations

import re

# MAC address: six hex pairs separated by : or - (e.g. a2:16:80:bb:aa:2b).
_MAC_RE = re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b")
# IPv6-ish link-local fragments the gateway also flags (e.g. fe80::ecee:eeff:feee).
_IPV6_RE = re.compile(r"\bfe80::[0-9A-Fa-f:]{4,}\b")
# Private IPv4 ranges (10./172.16-31./192.168.). Public IPs are left alone — they
# are rarely sensitive and sometimes part of a legitimate example.
_PRIV_IP_RE = re.compile(
    r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
    r"|192\.168\.\d{1,3}\.\d{1,3})\b")
# Credential-shaped tokens.
_CREDS = [
    re.compile(r"(?i)(authorization:\s*bearer\s+)[A-Za-z0-9._\-]+"),
    re.compile(r"(?i)\b(api[_-]?key\s*[=:]\s*)[A-Za-z0-9._\-]{12,}"),
    re.compile(r"(?i)\b(token\s*[=:]\s*)[A-Za-z0-9._\-]{16,}"),
    re.compile(r"\b[0-9a-f]{32}\b"),                 # bare 32-hex (gateway key shape)
]


def scrub(text: str) -> str:
    """Redact MAC/private-IP/credential tokens; keep hostnames and public content."""
    s = text or ""
    s = _MAC_RE.sub("[MAC]", s)
    s = _IPV6_RE.sub("[IPv6]", s)
    s = _PRIV_IP_RE.sub("[IP]", s)
    for pat in _CREDS:
        if pat.groups:
            s = pat.sub(lambda m: m.group(1) + "[REDACTED]", s)
        else:
            s = pat.sub("[REDACTED]", s)
    return s
