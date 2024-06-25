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


def BasicTypeRef(name: str) -> TypeRef:
    return TypeRef("basic", name)


def UnionTypeRef(types: list[TypeRef]) -> TypeRef:
    return TypeRef("union", types)
