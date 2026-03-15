"""
Query Cache Module
Provides a Redis-backed cache with automatic in-memory fallback.

Cache key: SHA-256(question + collection_id + str(top_k))
Cache TTL: configurable via CACHE_TTL (default 10 minutes)

If Redis is unavailable, a thread-safe in-memory dict with TTL tracking
is used transparently — no Redis installation required to run the system.
"""

import hashlib
import json
import time
import threading
from typing import Any, Dict, Optional

from app.config import CACHE_ENABLED, CACHE_TTL, REDIS_URL
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# In-memory fallback cache
# ---------------------------------------------------------------------------

class _InMemoryCache:
    """Thread-safe TTL dictionary cache."""

    def __init__(self) -> None:
        self._store: Dict[str, tuple] = {}   # key → (value, expires_at)
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.time() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl: int) -> None:
        with self._lock:
            self._store[key] = (value, time.time() + ttl)

    def delete_prefix(self, prefix: str) -> int:
        with self._lock:
            keys = [k for k in self._store if k.startswith(prefix)]
            for k in keys:
                del self._store[k]
            return len(keys)

    def size(self) -> int:
        with self._lock:
            now = time.time()
            return sum(1 for _, (_, exp) in self._store.items() if now <= exp)


# ---------------------------------------------------------------------------
# Cache wrapper
# ---------------------------------------------------------------------------

class QueryCache:
    """
    Unified cache interface with Redis primary and in-memory fallback.

    Usage:
        cache = QueryCache()
        key = cache.make_key("What is RAG?", "default", 5)
        hit = cache.get(key)
        if hit is None:
            result = expensive_query()
            cache.set(key, result)
    """

    def __init__(self) -> None:
        self._redis = None
        self._memory = _InMemoryCache()
        self._using_redis = False

        if CACHE_ENABLED:
            self._try_connect_redis()

    def _try_connect_redis(self) -> None:
        try:
            import redis
            client = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=1)
            client.ping()
            self._redis = client
            self._using_redis = True
            logger.info(f"Cache: Redis connected at {REDIS_URL}")
        except Exception as exc:
            logger.warning(
                f"Redis unavailable ({exc}) — using in-memory TTL cache"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(question: str, collection: str, top_k: int) -> str:
        """Generate a deterministic cache key."""
        raw = f"{question.strip().lower()}|{collection}|{top_k}"
        return "rag:" + hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Optional[Dict]:
        """
        Retrieve a cached value.

        Returns:
            The cached dict, or None on a miss / disabled cache.
        """
        if not CACHE_ENABLED:
            return None

        if self._using_redis:
            try:
                raw = self._redis.get(key)
                if raw:
                    logger.debug(f"Cache HIT (Redis): {key[:16]}…")
                    return json.loads(raw)
            except Exception as exc:
                logger.warning(f"Redis get error: {exc} — falling back to memory")
                self._using_redis = False

        result = self._memory.get(key)
        if result is not None:
            logger.debug(f"Cache HIT (memory): {key[:16]}…")
        return result

    def set(self, key: str, value: Dict, ttl: int = CACHE_TTL) -> None:
        """Store a value in the cache."""
        if not CACHE_ENABLED:
            return

        if self._using_redis:
            try:
                self._redis.setex(key, ttl, json.dumps(value))
                return
            except Exception as exc:
                logger.warning(f"Redis set error: {exc} — using memory cache")
                self._using_redis = False

        self._memory.set(key, value, ttl)

    def invalidate_collection(self, collection_id: str) -> int:
        """
        Invalidate all cached entries for a given collection.

        Returns:
            Number of keys deleted.
        """
        if not CACHE_ENABLED:
            return 0

        prefix = f"rag:{hashlib.sha256(collection_id.encode()).hexdigest()[:8]}"
        # For Redis, use SCAN + DELETE pattern; for memory, use prefix match
        if self._using_redis:
            try:
                keys = list(self._redis.scan_iter(f"rag:*"))
                if keys:
                    self._redis.delete(*keys)
                return len(keys)
            except Exception:
                pass
        return self._memory.delete_prefix("rag:")

    @property
    def backend(self) -> str:
        """Return the active cache backend name."""
        return "redis" if self._using_redis else "memory"


# Singleton instance shared across the application
_cache_instance: Optional[QueryCache] = None


def get_cache() -> QueryCache:
    """Return the application-level QueryCache singleton."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = QueryCache()
    return _cache_instance
