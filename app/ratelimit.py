"""Token-bucket rate limiter (in-process). Capacity = limit per minute; refills continuously."""
import math
import threading
import time


class RateLimiter:
    def __init__(self, clock=time.monotonic):
        self._clock, self._buckets, self._lock = clock, {}, threading.Lock()

    def check(self, key: str, per_min: int) -> tuple[bool, int]:
        """Spend one token. Returns (allowed, seconds_to_wait). per_min <= 0 disables limiting."""
        if per_min <= 0:
            return True, 0
        now, rate = self._clock(), per_min / 60.0
        with self._lock:
            tokens, last = self._buckets.get(key, (float(per_min), now))
            tokens = min(float(per_min), tokens + (now - last) * rate)
            allowed = tokens >= 1
            self._buckets[key] = (tokens - 1 if allowed else tokens, now)
            if len(self._buckets) > 10_000:  # drop idle keys so memory stays bounded
                self._buckets = {k: v for k, v in self._buckets.items() if v[1] > now - 3600}
        return allowed, 0 if allowed else math.ceil((1 - tokens) / rate)
