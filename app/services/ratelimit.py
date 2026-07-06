"""In-memory sliding-window rate limiter (ТЗ п.6): 5 requests/hour per IP for
requests without paid credits. Single-instance Cloud Run (min=0, low traffic)
makes an in-memory limiter sufficient for v1.0."""

import threading
import time


class RateLimiter:
    def __init__(self, limit: int, window_seconds: int = 3600):
        self._limit = limit
        self._window = window_seconds
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = [t for t in self._hits.get(key, []) if now - t < self._window]
            if len(hits) >= self._limit:
                self._hits[key] = hits
                return False
            hits.append(now)
            self._hits[key] = hits
            return True
