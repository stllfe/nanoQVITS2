from __future__ import annotations

import contextlib

from os import getenv
from typing import ClassVar


class Context(contextlib.ContextDecorator):
    stack: ClassVar[list[dict[str, int]]] = [{}]

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def __enter__(self) -> None:
        Context.stack[-1] = {
            k: o.value for k, o in ContextVar._cache.items()
        }  # store current state.
        for k, v in self.kwargs.items():
            ContextVar._cache[k].value = v  # update to new temporary state.
        Context.stack.append(
            self.kwargs
        )  # store the temporary state so we know what to undo later.

    def __exit__(self, *args) -> None:
        for k in Context.stack.pop():
            ContextVar._cache[k].value = Context.stack[-1].get(k, ContextVar._cache[k].value)


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
