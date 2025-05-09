from __future__ import annotations

from typing import Callable


class CounterDict[K, V](dict[K, V]):
    __slots__ = ("default_func",)

    def __init__(self, default_func: Callable[[], V]):
        super().__init__()

        self.default_func = default_func

    def modify(self, key: K, func: Callable[[V], V]) -> V:
        val = func(self.get(key, self.default_func()))
        self[key] = val
        return val

    def inc(self, key: K) -> V:
        return self.modify(key, lambda x: x + 1)


class Cell[T]:
    value: T

    def __init__(self, value: T):
        self.value = value

    def get(self) -> T:
        return self.value

    def set(self, value: T):
        self.value = value

    def swap(self, value: T) -> T:
        old = self.value
        self.value = value
        return old
