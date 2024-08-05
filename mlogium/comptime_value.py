from __future__ import annotations

from dataclasses import dataclass
from abc import abstractmethod, ABC

from .value import *


class ComptimeInterpreterContext(ABC):
    scope: 'ComptimeScopeStack'

    def __init__(self, scope: 'ComptimeScopeStack'):
        self.scope = scope

    @abstractmethod
    def error(self, msg: str):
        raise NotImplementedError

    @abstractmethod
    def interpret(self, node: Node) -> CValue:
        raise NotImplementedError

    @abstractmethod
    def tmp_num(self) -> int:
        raise NotImplementedError

    def tmp(self) -> str:
        return f"__tmp_{self.tmp_num()}"


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
        return TupleCValue(values)

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def type(self) -> CType:
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

    def callable(self) -> bool:
        return False

    def callable_with(self, types: list[CType]) -> list[CType | None] | None:
        return None if self.callable() else []

    def call(self, ctx: ComptimeInterpreterContext, values: list[CValue]) -> CValue:
        raise NotImplementedError


@dataclass(slots=True)
class OpaqueType(Type):
    def __str__(self):
        return "[comptime]"

    def __eq__(self, other):
        return False

    def contains(self, other: Type) -> bool:
        return False


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
class NumberCType(CType):
    def __str__(self):
        return "num"

    def to_runtime(self) -> Type:
        return NumberType()


@dataclass(slots=True, eq=True)
class StringCType(CType):
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


@dataclass(slots=True, eq=True)
class FunctionCType(CType):
    params: list[CType | None]
    result: CType | None
    code_hash: int

    def __str__(self):
        return f"comptime fn {self.name}({', '.join(str(p) if p is not None else '?' for p in params)}) \
-> {result if result is not None else '?'}"

    def to_runtime(self) -> Type:
        return OpaqueType()


class NullCValue(CValue):
    def __str__(self):
        return "null"

    @property
    def type(self) -> CType:
        return NullCType()

    def to_runtime(self, ctx: CompilationContext) -> Value:
        return Value.null()


@dataclass(slots=True)
class IntCValue(CValue):
    value: int

    def __str__(self):
        return str(self.value)

    @property
    def type(self) -> CType:
        return NumberCType()

    def to_runtime(self, ctx: CompilationContext) -> Value:
        return Value.of_number(self.value)


@dataclass(slots=True)
class FloatCValue(CValue):
    value: float

    def __str__(self):
        return str(self.value)

    @property
    def type(self) -> CType:
        return NumberCType()

    def to_runtime(self, ctx: CompilationContext) -> Value:
        return Value.of_number(self.value)


@dataclass(slots=True)
class StringCValue(CValue):
    value: str

    def __str__(self):
        return f"\"{self.value}\""

    @property
    def type(self) -> CType:
        return StringCType()

    def to_runtime(self, ctx: CompilationContext) -> Value:
        return Value.of_string(self.value)


@dataclass(slots=True)
class TupleCValue(CValue):
    values: list[CValue]

    def __str__(self):
        return f"({', '.join(map(str, self.values))})"

    @property
    def type(self) -> CType:
        return TupleCType([v.type for v in self.values])

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

    @property
    def type(self) -> CType:
        return FunctionCType(self.params, self.result, id(self.function))

    def callable(self) -> bool:
        return True

    def callable_with(self, types: list[CType]) -> list[CType | None] | None:
        return None if len(types) == len(self.params) and all(
            p is None or p.contains(t) for p, t in zip(self.params, types)) else self.params

    def call(self, ctx: ComptimeInterpreterContext, values: list[CValue]) -> CValue:
        # TODO: add params to scope
        with ctx.scope(ctx.tmp()):
            return self.function(values)
