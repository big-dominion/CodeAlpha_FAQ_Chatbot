"""
cache.py

Bounded in-memory response cache for /api/v1/chat.

Why this exists: during a traffic spike, many different users tend to
ask the same handful of common questions (e.g. "who can become
president of Nigeria"). Without this cache, every one of those
repeated questions independently calls Pinecone + Groq + translation,
which is exactly what exhausts free-tier rate limits under load.

This cache is intentionally simple and intentionally bounded:
- LRU eviction via OrderedDict, capped at _MAX_ENTRIES, so memory usage
  can never grow unbounded no matter how many unique questions arrive.
- A short TTL (_CACHE_TTL) so cached legal answers don't go stale
  indefinitely if the underlying statute data or prompt changes.
- Keyed on a normalized (lowercased, trimmed) hash of the question, so
  "Who can become President of Nigeria?" and "who can become president
  of nigeria" hit the same cache entry.

Worst-case memory: _MAX_ENTRIES * (avg response size, ~2-5KB) - with
the defaults below, roughly 2.5MB guaranteed ceiling, regardless of
how much traffic or how many unique questions the app receives.
"""

import time
import hashlib
from collections import OrderedDict

_MAX_ENTRIES = 500   # hard ceiling - oldest entries evicted past this
_CACHE_TTL = 1200     # seconds (20 minutes)

_response_cache: "OrderedDict[str, tuple[dict, float]]" = OrderedDict()


def _cache_key(question: str) -> str:
    normalized = question.strip().lower()
    return hashlib.sha256(normalized.encode()).hexdigest()


def get_cached_response(question: str) -> dict | None:
    """Returns the cached response dict if present and not expired, else None."""
    key = _cache_key(question)
    entry = _response_cache.get(key)
    if entry is None:
        return None

    response, timestamp = entry
    if (time.time() - timestamp) >= _CACHE_TTL:
        del _response_cache[key]
        return None

    _response_cache.move_to_end(key)  # mark as recently used
    return response


def set_cached_response(question: str, response: dict) -> None:
    """Stores a response, evicting the oldest entry if over the cap."""
    key = _cache_key(question)
    _response_cache[key] = (response, time.time())
    _response_cache.move_to_end(key)

    while len(_response_cache) > _MAX_ENTRIES:
        _response_cache.popitem(last=False)