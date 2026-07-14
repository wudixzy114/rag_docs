"""Vision layer — read each source image with the model's own eyes.

Quality-first mandate (user): the pre-baked OCR is bad; we have the originals, so
a strong vision model transcribes them instead. Output per image:
  - transcript: faithful text (tables as GFM, logs verbatim)
  - meaning:    1-3 sentence diagnostic reading

Cached by (image bytes sha + model + VISION_VERSION): re-running the pipeline
never re-pays for an unchanged image, and bumping the prompt version re-runs all.
The result is written back onto the Image node (`vision_text`), which every later
stage reads via `Image.best_text`.
"""
from __future__ import annotations

import hashlib
import logging

from ragkb.llm.client import LLMClient, LLMError, VisionImage
from ragkb.parse.model import Document, Image
from ragkb.pipeline.prompts import VISION_VERSION, VISION_SYSTEM, build_vision_user
from ragkb.pipeline.scrub import scrub
from ragkb.store.cache import Cache, key_for

log = logging.getLogger(__name__)

_VISION_MAX_TOKENS = 4096


def _parse_vision(text: str) -> tuple[str, str]:
    """Split the model output into (transcript, meaning). Tolerant: if the markers
    are missing, treat the whole thing as transcript."""
    t = text or ""
    transcript, meaning = t, ""
    if "[TRANSCRIPT]" in t:
        rest = t.split("[TRANSCRIPT]", 1)[1]
        if "[MEANING]" in rest:
            transcript, meaning = rest.split("[MEANING]", 1)
        else:
            transcript = rest
    elif "[MEANING]" in t:
        transcript, meaning = t.split("[MEANING]", 1)
    return transcript.strip(), meaning.strip()


def vision_read_image(img: Image, llm: LLMClient, cache: Cache,
                      model: str | None = None) -> dict:
    """Transcribe one image. Returns {transcript, meaning, model, truncated}.
    Cached by image content + model + prompt version. Fills img.vision_text."""
    if not img.exists:
        return {"transcript": "", "meaning": "", "model": "", "truncated": False,
                "error": "image file missing"}
    img_sha = hashlib.sha256(img.data_bytes()).hexdigest()
    model_id = model or "task:vision"
    ck = key_for(img_sha, model_id, VISION_VERSION)
    cached = cache.get("vision", ck)
    if cached is not None:
        img.vision_text = cached.get("transcript", "")
        return cached

    vi = VisionImage(data=img.data_bytes(), media_type=img.media_type())
    try:
        r = llm.complete_vision(
            system=VISION_SYSTEM,
            user=build_vision_user(scrub(img.inline_ocr)),
            images=[vi], max_tokens=_VISION_MAX_TOKENS, task="vision", model=model)
    except LLMError as exc:
        log.warning("vision read failed for %s: %s", img.rel_path, exc)
        return {"transcript": "", "meaning": "", "model": "", "truncated": False,
                "error": str(exc)}

    transcript, meaning = _parse_vision(r.text)
    result = {
        "transcript": transcript,
        "meaning": meaning,
        "model": r.model,
        # finish_reason == length means the transcription was cut off — a
        # structural defect the user explicitly forbids. Surface it so the
        # orchestrator can re-run with a bigger budget rather than caching a
        # half-image silently.
        "truncated": r.finish_reason == "length",
    }
    if not result["truncated"]:
        cache.put("vision", ck, result)
    img.vision_text = transcript
    return result
