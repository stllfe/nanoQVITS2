from __future__ import annotations

from os import getenv
from typing import ClassVar


class ContextVar:
    _cache: ClassVar[dict[str, ContextVar]] = {}
    value: int

    def __new__(cls, key: str, default: int) -> ContextVar:
        if key in ContextVar._cache:
            return ContextVar._cache[key]
        instance = ContextVar._cache[key] = object.__new__(cls)
        instance.value = int(getenv(key) or default)
        return instance

    def __bool__(self):
        return bool(self.value)

    def __ge__(self, x):
        return self.value >= x

    def __gt__(self, x):
        return self.value > x

    def __lt__(self, x):
        return self.value < x


# 0 — no debug info
# 1 — little debug info
# 2 — highly detailed debug log
DEBUG = ContextVar('DEBUG', 0)


def debug(*args, level: int = 1, rank: int | None = None) -> None:
    if DEBUG >= level:
        print(*args)
