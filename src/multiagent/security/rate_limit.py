"""应用层限流（W14）：按 API Key 指纹或客户端 IP 的令牌桶。

单进程口径（与任务表/指标一致）；多 worker 下每个进程各自计数，
部署阶段若要多副本，应换成 Redis 令牌桶。
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    def __init__(self, per_minute: int, burst: int) -> None:
        self.rate = max(1, int(per_minute)) / 60.0
        self.burst = max(1, int(burst))
        self._buckets: dict[str, tuple[float, float]] = {}  # identity -> (tokens, last)
        self._lock = threading.Lock()
        self._calls = 0

    def allow(self, identity: str) -> tuple[bool, float]:
        """返回 (是否放行, 建议等待秒数)。"""
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(identity, (float(self.burst), now))
            tokens = min(float(self.burst), tokens + (now - last) * self.rate)
            self._buckets[identity] = (tokens, now)
            self._calls += 1
            if self._calls % 1000 == 0:
                self._prune(now)
            if tokens < 1.0:
                return False, (1.0 - tokens) / self.rate
            self._buckets[identity] = (tokens - 1.0, now)
            return True, 0.0

    def _prune(self, now: float) -> None:
        stale = [key for key, (_, last) in self._buckets.items() if now - last > 3600]
        for key in stale:
            self._buckets.pop(key, None)
