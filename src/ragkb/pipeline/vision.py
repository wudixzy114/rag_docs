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

from ragkb.llm.client import LLMClient, LLMError, LLMQuotaError, VisionImage
from ragkb.parse.model import Image
from ragkb.pipeline.prompts import VISION_VERSION, VISION_SYSTEM, build_vision_user
from ragkb.pipeline.imageprep import fit_image
from ragkb.pipeline.scrub import mask
from ragkb.store.cache import Cache, key_for

log = logging.getLogger(__name__)

_VISION_MAX_TOKENS = 4096
_VISION_RETRY_MAX_TOKENS = 12288


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
                      model: str | None = None,
                      max_tokens: int = _VISION_MAX_TOKENS,
                      retry_max_tokens: int | None = _VISION_RETRY_MAX_TOKENS) -> dict:
    """Transcribe one image. Returns {transcript, meaning, model, truncated}.
    Cached by image content + model + prompt version. Fills img.vision_text.

    A length-truncated transcription is retried once with a larger completion
    budget. If that retry is also truncated, fail the image read instead of
    exposing a partial transcript to extraction."""
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

    # Fit the image under the gateway's size cap before sending: shrink oversize
    # rasters, take the first frame of an animated GIF. Keyed cache above is on the
    # ORIGINAL bytes, so this reprocessing happens once per uncached image.
    fitted_data, fitted_media = fit_image(img.data_bytes(), img.media_type())
    vi = VisionImage(data=fitted_data, media_type=fitted_media)
    budgets = [max_tokens]
    if retry_max_tokens is not None and retry_max_tokens > max_tokens:
        budgets.append(retry_max_tokens)

    for attempt, budget in enumerate(budgets):
        try:
            r = llm.complete_vision(
                system=VISION_SYSTEM,
                user=build_vision_user(mask(img.inline_ocr)),
                images=[vi], max_tokens=budget, task="vision", model=model)
        except LLMQuotaError:
            # Every vision-capable model is quota-exhausted. Do NOT fabricate and
            # do NOT fall back to a text model. This transient result is not cached.
            log.warning("vision quota exhausted for %s; leaving unread", img.rel_path)
            img.vision_failed = True
            return {"transcript": "", "meaning": "", "model": "",
                    "truncated": False, "error": "vision_quota_exhausted"}
        except LLMError as exc:
            log.warning("vision read failed for %s: %s", img.rel_path, exc)
            img.vision_failed = True
            return {"transcript": "", "meaning": "", "model": "",
                    "truncated": False, "error": str(exc)}

        transcript, meaning = _parse_vision(r.text)
        truncated = r.finish_reason == "length"
        if truncated and attempt + 1 < len(budgets):
            log.warning("vision transcription truncated for %s at %d tokens; retrying at %d",
                        img.rel_path, budget, budgets[attempt + 1])
            continue
        if truncated:
            # Do not let Image.best_text select a half transcript. Raising aborts
            # this document run, and the orchestrator preserves its last good snapshot.
            img.vision_text = ""
            img.vision_failed = True
            raise LLMError(
                f"vision transcription still truncated after retry [{img.rel_path}]")

        result = {"transcript": transcript, "meaning": meaning,
                  "model": r.model, "truncated": False}
        cache.put("vision", ck, result)
        img.vision_text = transcript
        img.vision_failed = False
        return result

    raise AssertionError("vision token budget list must not be empty")
