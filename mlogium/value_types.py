from typing import Any
from dataclasses import dataclass


@dataclass(slots=True)
class TypeRef:
    type: str
    data: Any


class _TypeSupplier:
    def __getattr__(self, item: str) -> TypeRef:
        return TypeRef("basic", item)


Types = _TypeSupplier()


def BasicType(name: str) -> TypeRef:
    return TypeRef("basic", name)


def UnionType(types: list[TypeRef]) -> TypeRef:
    return TypeRef("union", types)
