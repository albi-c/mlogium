from __future__ import annotations

from dataclasses import dataclass
from abc import abstractmethod, ABC

from .value import *


class ComptimeInterpreterContext(ABC):
    @abstractmethod
    def error(self, msg: str):
        raise NotImplementedError


class CValue(ABC):
    @classmethod
    def null(cls) -> CValue:
        return NullCValue()

    @classmethod
    def of_int(cls, value: int) -> CValue:
        return IntCValue(value)

    @classmethod
    def of_float(cls, value: float) -> CValue:
        return FloatCValue(value)

    @classmethod
    def of_number(cls, value: int | float) -> CValue:
        if isinstance(value, int) or value.is_integer():
            return IntCValue(value)
        return FloatCValue(value)

    @classmethod
    def of_string(cls, value: str) -> CValue:
        return StringCValue(value)

    @classmethod
    def of_tuple(cls, values: list[CValue]) -> CValue:
        pass

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def to_runtime(self, ctx: CompilationContext) -> Value:
        raise NotImplementedError

    def unpackable(self) -> bool:
        return False

    def unpack(self) -> list[CValue]:
        raise NotImplementedError

    def unpack_req(self, ctx: ComptimeInterpreterContext) -> list[CValue]:
        if not self.unpackable():
            ctx.error(f"Value '{self}' is not unpackable")
        return self.unpack()


@dataclass(slots=True, eq=True)
class CType(ABC):
    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def to_runtime(self) -> Type:
        raise NotImplementedError

    def contains(self, other: CType) -> bool:
        return self == other


@dataclass(slots=True, eq=True)
class NullCType(CType):
    def __str__(self):
        return "null"

    def to_runtime(self) -> Type:
        return NullType()


@dataclass(slots=True, eq=True)
class NumCType(CType):
    def __str__(self):
        return "num"

    def to_runtime(self) -> Type:
        return NumberType()


@dataclass(slots=True, eq=True)
class StrCType(CType):
    def __str__(self):
        return "str"

    def to_runtime(self) -> Type:
        return StringType()


@dataclass(slots=True, eq=True)
class TupleCType(CType):
    types: list[CType]

    def __str__(self):
        return f"({', '.join(map(str, self.types))})"

    def to_runtime(self) -> Type:
        return TupleType([t.to_runtime() for t in self.types])


@dataclass(slots=True, eq=True)
class UnionCType(CType):
    types: list[CType]

    def __str__(self):
        return f"{' | '.join(map(str, self.types))}"

    def to_runtime(self) -> Type:
        return UnionType([t.to_runtime() for t in self.types])

    def contains(self, other: CType) -> bool:
        if isinstance(other, CType.Union):
            return any(self.contains(t) for t in other.types)
        return any(t.contains(other) for t in self.types)


class NullCValue(CValue):
    def __str__(self):
        return "null"

    def to_runtime(self, ctx: CompilationContext) -> Value:
        return Value.null()


@dataclass(slots=True)
class IntCValue(CValue):
    value: int

    def __str__(self):
        return str(self.value)

    def to_runtime(self, ctx: CompilationContext) -> Value:
        return Value.of_number(self.value)


@dataclass(slots=True)
class FloatCValue(CValue):
    value: float

    def __str__(self):
        return str(self.value)

    def to_runtime(self, ctx: CompilationContext) -> Value:
        return Value.of_number(self.value)


@dataclass(slots=True)
class StringCValue(CValue):
    value: str

    def __str__(self):
        return f"\"{self.value}\""

    def to_runtime(self, ctx: CompilationContext) -> Value:
        return Value.of_string(self.value)


@dataclass(slots=True)
class TupleCValue(CValue):
    values: list[CValue]

    def __str__(self):
        return f"({', '.join(map(str, self.values))})"

    def to_runtime(self, ctx: CompilationContext) -> Value:
        return Value.of_tuple(ctx, [v.to_runtime(ctx) for v in self.values])

    def unpackable(self) -> bool:
        return True

    def unpack(self) -> list[CValue]:
        return self.values


@dataclass(slots=True)
class FunctionCValue(CValue):
    name: str
    params: list[CType | None]
    result: CType | None
    function: Callable[[list[CValue]], CValue]

    def __str__(self):
        return f"comptime fn {self.name}({', '.join(str(p) if p is not None else '?' for p in params)}) \
-> {result if result is not None else '?'}"

    def to_runtime(self, ctx: CompilationContext) -> Value:
        # TODO: accept parameters from runtime context

        assert len(self.params) == 0

        return Value(SpecialFunctionType(
            self.name,
            [t.to_runtime() if t is not None else None for t in self.params],
            self.result if self.result is not None else None,
            lambda ctx_, _: self.function([]).to_runtime(ctx_)
        ), "", True)
