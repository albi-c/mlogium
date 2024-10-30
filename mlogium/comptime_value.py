from __future__ import annotations

from decimal import DivisionByZero
from logging import error
from pty import slave_open
from typing import Iterator
from dataclasses import dataclass

from .structure import Cell
from .value import *
from .compilation_context import ErrorContext


class ComptimeInterpreterContext(ErrorContext, ABC):
    scope: 'ComptimeScopeStack'

    def __init__(self, scope: 'ComptimeScopeStack'):
        self.scope = scope

    @abstractmethod
    def interpret(self, node: Node) -> CValue:
        raise NotImplementedError

    @abstractmethod
    def tmp_num(self) -> int:
        raise NotImplementedError

    def tmp(self) -> str:
        return f"__tmp_{self.tmp_num()}"


class Return(Exception):
    pos: Position
    value: CValue | None

    def __init__(self, pos: Position, value: CValue | None):
        self.pos = pos
        self.value = value


@dataclass(slots=True)
class BaseCValue(ABC):
    @abstractmethod
    def deref(self) -> CValue:
        raise NotImplementedError

    @abstractmethod
    def assign(self, ctx: ComptimeInterpreterContext, value: CValue) -> bool:
        raise NotImplementedError


@dataclass(slots=True)
class CLValue(BaseCValue, ABC):
    pass


@dataclass(slots=True)
class VariableCLValue(CLValue):
    value: CValue
    constant: bool

    def deref(self) -> CValue:
        return self.value

    def assign(self, ctx: ComptimeInterpreterContext, value: CValue) -> bool:
        if self.constant:
            return False

        self.value = value.into_req(ctx, self.value.type)
        return True


@dataclass(slots=True)
class TupleElementCLValue(CLValue):
    tup: TupleCValue
    index: int

    def deref(self) -> CValue:
        return self.tup.values[self.index]

    def assign(self, ctx: ComptimeInterpreterContext, value: CValue) -> bool:
        self.tup.values[self.index] = value
        return True


@dataclass(slots=True)
class CValue(BaseCValue, ABC):
    def deref(self) -> CValue:
        return self

    def assign(self, ctx: ComptimeInterpreterContext, value: CValue) -> bool:
        return False

    @classmethod
    def null(cls) -> CValue:
        return NullCValue()

    @classmethod
    def of_float(cls, value: float) -> CValue:
        return NumberCValue(value)

    @classmethod
    def of_number(cls, value: int | float) -> CValue:
        return NumberCValue(float(value))

    @classmethod
    def of_boolean(cls, value: bool) -> CValue:
        return BooleanCValue(value)

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
    def to_runtime(self, ctx: CompilationContext, i_ctx: ComptimeInterpreterContext) -> Value:
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

    def get_wrapped_type(self, ctx: ComptimeInterpreterContext) -> CType | None:
        return None

    def get_wrapped_type_req(self, ctx: ComptimeInterpreterContext) -> CType:
        if (type_ := self.get_wrapped_type(ctx)) is None:
            ctx.error(f"Value of type '{self.type}' does not contain a type")

        return type_

    def is_true(self, ctx: ComptimeInterpreterContext) -> bool | None:
        return None

    def is_true_req(self, ctx: ComptimeInterpreterContext) -> bool:
        if (cond := self.is_true(ctx)) is None:
            ctx.error(f"Value of type '{self.type}' cannot be used as a condition")

        return cond

    def iterable(self) -> bool:
        return False

    def iterate(self, ctx: ComptimeInterpreterContext) -> Iterator[CValue]:
        raise NotImplementedError

    def into(self, ctx: ComptimeInterpreterContext, type_: CType) -> CValue | None:
        if type_.contains(self.type):
            return self

        return self.into_impl(ctx, type_)

    def into_req(self, ctx: ComptimeInterpreterContext, type_: CType) -> CValue:
        if (val := self.into(ctx, type_)) is None:
            ctx.error(f"Value of type '{self.type}' is not convertible to type {type_}")

        return val

    def into_impl(self, ctx: ComptimeInterpreterContext, type_: CType) -> CValue | None:
        if (truthy := self.is_true(ctx)) is not None:
            return CValue.of_boolean(truthy)

    def index(self, ctx: ComptimeInterpreterContext, indices: list[CValue]) -> BaseCValue | None:
        return None

    def index_req(self, ctx: ComptimeInterpreterContext, indices: list[CValue]) -> BaseCValue:
        if (val := self.index(ctx, indices)) is None:
            ctx.error(f"Value of type '{self.type}' is not indexable with indices of types [{', '.join(str(i.type) for i in indices)}]")

        return val

    def unary_op(self, ctx: ComptimeInterpreterContext, op: str) -> CValue | None:
        if op == "!" and (truthy := self.is_true(ctx)) is not None:
            return CValue.of_boolean(not truthy)

    def binary_op(self, ctx: ComptimeInterpreterContext, op: str, other: CValue) -> CValue | None:
        return None

    def binary_op_r(self, ctx: ComptimeInterpreterContext, other: CValue, op: str) -> CValue | None:
        return None

    def do_binary_op(self, ctx: ComptimeInterpreterContext, op: str, other: CValue) -> CValue:
        if (val := self.binary_op(ctx, op, other)) is None:
            if (val_r := other.binary_op_r(ctx, self, op)) is None:
                ctx.error(f"Operator {op} is not supported for values of types '{self.type}' and {other.type}")
            return val_r
        return val

    def getattr(self, ctx: ComptimeInterpreterContext, name: str, static: bool) -> BaseCValue | None:
        return None

    def getattr_req(self, ctx: ComptimeInterpreterContext, name: str, static: bool) -> BaseCValue:
        if (val := self.getattr(ctx, name, static)) is None:
            ctx.error(f"Value of type '{self.type}' has no{' static' if static else ''} attribute '{name}'")

        return val


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
class TypeCType(CType):
    def __str__(self):
        return "Type"

    def to_runtime(self) -> Type:
        return OpaqueType()


@dataclass(slots=True, eq=True)
class TupleSourceCType(CType):
    def __str__(self):
        return "Tuple"

    def to_runtime(self) -> Type:
        return OpaqueType()


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
class BooleanCType(CType):
    def __str__(self):
        return "bool"

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
class OpaqueCType(CType):
    def __str__(self):
        return "Opaque"

    def to_runtime(self) -> Type:
        return OpaqueType()


@dataclass(slots=True, eq=True)
class UnionCType(CType):
    types: list[CType]

    def __str__(self):
        return f"{' | '.join(map(str, self.types))}"

    def to_runtime(self) -> Type:
        return UnionType([t.to_runtime() for t in self.types])

    def contains(self, other: CType) -> bool:
        if isinstance(other, UnionCType):
            return any(self.contains(t) for t in other.types)
        return any(t.contains(other) for t in self.types)


@dataclass(slots=True, eq=True)
class FunctionCType(CType):
    params: list[CType | None]
    result: CType | None

    def __str__(self):
        return f"comptime fn({', '.join(str(p) if p is not None else '?' for p in self.params)}) \
-> {self.result if self.result is not None else '?'}"

    def to_runtime(self) -> Type:
        return OpaqueType()


@dataclass(slots=True, eq=True)
class RangeCType(CType):
    def __str__(self):
        return "Range"

    def to_runtime(self) -> Type:
        return RangeType()


@dataclass(slots=True, eq=True)
class RangeWithStepCType(CType):
    def __str__(self):
        return "RangeWithStep"

    def to_runtime(self) -> Type:
        return RangeWithStepType()


@dataclass(slots=True)
class TypeCValue(CValue):
    type: CType

    def __str__(self):
        return f"Type[{self.type}]"

    @property
    def type(self) -> CType:
        return TypeCType()

    def to_runtime(self, ctx: CompilationContext, i_ctx: ComptimeInterpreterContext) -> Value:
        return Value.of_type(self.type.to_runtime())

    def get_wrapped_type(self, ctx: ComptimeInterpreterContext) -> CType | None:
        return self.type


@dataclass(slots=True)
class TupleSourceCValue(CValue):
    def __str__(self):
        return "Tuple"

    @property
    def type(self) -> CType:
        return TupleSourceCType()

    def to_runtime(self, ctx: CompilationContext, i_ctx: ComptimeInterpreterContext) -> Value:
        return Value.null()

    def callable(self) -> bool:
        return True

    def callable_with(self, types: list[CType]) -> list[CType | None] | None:
        return None if len(types) == 2 and isinstance(types[0], TypeCType) and isinstance(types[1], NumberCType) \
            else [TypeCType(), NumberCType()]

    def call(self, ctx: ComptimeInterpreterContext, values: list[CValue]) -> CValue:
        type_ = values[0].get_wrapped_type_req(ctx)
        count = values[1].into_req(ctx, NumberCType())
        assert isinstance(count, NumberCValue)
        count = count.value
        if not count.is_integer():
            ctx.error("Tuple count must be an integer")
        count = int(count)
        if count < 0:
            ctx.error("Tuple count cannot be negative")
        return TypeCValue(TupleCType([type_] * count))

    def index(self, ctx: ComptimeInterpreterContext, indices: list[CValue]) -> BaseCValue | None:
        types = [i.get_wrapped_type_req(ctx) for i in indices]
        return TypeCValue(TupleCType(types))


@dataclass(slots=True)
class NullCValue(CValue):
    def __str__(self):
        return "null"

    @property
    def type(self) -> CType:
        return NullCType()

    def to_runtime(self, ctx: CompilationContext, i_ctx: ComptimeInterpreterContext) -> Value:
        return Value.null()

    def is_true(self, ctx: ComptimeInterpreterContext) -> bool | None:
        return False


@dataclass(slots=True)
class NumberCValue(CValue):
    BINARY_FLOAT_OPS = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: a / b,
        "//": lambda a, b: a // b,
        "/.": lambda a, b: a // b,
        "%": lambda a, b: a % b,
        "**": lambda a, b: a ** b
    }

    BINARY_INT_OPS = {
        "<<": lambda a, b: a << b,
        ">>": lambda a, b: a >> b,
        "|": lambda a, b: a | b,
        "&": lambda a, b: a & b,
        "^": lambda a, b: a ^ b
    }

    COMPARISON_OPS = {
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b
    }

    value: float

    def __str__(self):
        return str(self.value)

    @property
    def type(self) -> CType:
        return NumberCType()

    def as_int_or_float(self) -> int | float:
        if self.value.is_integer():
            return int(self.value)
        else:
            return self.value

    def to_runtime(self, ctx: CompilationContext, i_ctx: ComptimeInterpreterContext) -> Value:
        return Value.of_number(self.as_int_or_float())

    def is_true(self, ctx: ComptimeInterpreterContext) -> bool | None:
        return self.value != 0

    def into_impl(self, ctx: ComptimeInterpreterContext, type_: CType) -> CValue | None:
        if type_.contains(StringCType()):
            return CValue.of_string(str(self.as_int_or_float()))

        elif type_.contains(BooleanCType()):
            return CValue.of_boolean(self.value != 0)

    def unary_op(self, ctx: ComptimeInterpreterContext, op: str) -> CValue | None:
        if op == "-":
            return CValue.of_float(-self.value)

        elif op == "~":
            return CValue.of_number(~int(self.value))

        elif op == "!":
            return CValue.of_boolean(self.value == 0)

    @classmethod
    def _binary_op(cls, ctx: ComptimeInterpreterContext, left: float, op: str, right: float) -> CValue | None:
        if (func := cls.BINARY_FLOAT_OPS.get(op)) is not None:
            try:
                return CValue.of_number(func(left, right))
            except ZeroDivisionError:
                ctx.error("Division by zero")

        elif (func := cls.BINARY_INT_OPS.get(op)) is not None:
            return CValue.of_float(float(func(int(left), int(right))))

        elif (func := cls.COMPARISON_OPS.get(op)) is not None:
            return CValue.of_boolean(func(left, right))

    def binary_op(self, ctx: ComptimeInterpreterContext, op: str, other: CValue) -> CValue | None:
        if NumberCType().contains(other.type):
            assert isinstance(other, NumberCValue)
            other_val = other.value
            return self._binary_op(ctx, self.value, op, other_val)

    def binary_op_r(self, ctx: ComptimeInterpreterContext, other: CValue, op: str) -> CValue | None:
        if NumberCType().contains(other.type):
            assert isinstance(other, NumberCValue)
            other_val = other.value
            return self._binary_op(ctx, other_val, op, self.value)


@dataclass(slots=True)
class BooleanCValue(CValue):
    value: bool

    def __str__(self):
        return "true" if self.value else "false"

    @property
    def type(self) -> CType:
        return BooleanCType()

    def to_runtime(self, ctx: CompilationContext, i_ctx: ComptimeInterpreterContext) -> Value:
        return Value.of_number(1 if self.value else 0)

    def is_true(self, ctx: ComptimeInterpreterContext) -> bool | None:
        return self.value

    def into_impl(self, ctx: ComptimeInterpreterContext, type_: CType) -> CValue | None:
        if type_.contains(NumberCType()):
            return CValue.of_number(1 if self.value else 0)

    def binary_op(self, ctx: ComptimeInterpreterContext, op: str, other: CValue) -> CValue | None:
        if op in ("&&", "||", "^"):
            other_val = other.is_true_req(ctx)

            if op == "&&":
                return CValue.of_boolean(self.value and other_val)

            elif op == "||":
                return CValue.of_boolean(self.value or other_val)

            elif op == "^":
                return CValue.of_boolean(self.value != other_val)

    def binary_op_r(self, ctx: ComptimeInterpreterContext, other: CValue, op: str) -> CValue | None:
        return self.binary_op(ctx, op, other)


@dataclass(slots=True)
class StringCValue(CValue):
    value: str

    def __str__(self):
        return f"\"{self.value}\""

    @property
    def type(self) -> CType:
        return StringCType()

    def to_runtime(self, ctx: CompilationContext, i_ctx: ComptimeInterpreterContext) -> Value:
        return Value.of_string(f"\"{self.value}\"")

    def is_true(self, ctx: ComptimeInterpreterContext) -> bool | None:
        return len(self.value) != 0

    def binary_op(self, ctx: ComptimeInterpreterContext, op: str, other: CValue) -> CValue | None:
        if op in ("==", "!=", "+") and StringCType().contains(other.type):
            assert isinstance(other, StringCValue)
            other_val = other.value

            if op == "==":
                return CValue.of_boolean(self.value == other_val)
            elif op == "!=":
                return CValue.of_boolean(self.value != other_val)
            elif op == "+":
                return CValue.of_string(self.value + other_val)

        elif op == "*" and NumberCType().contains(other.type):
            assert isinstance(other, NumberCValue)
            count = int(other.value)
            if count < 0:
                ctx.error("String repeat count cannot be negative")
            return CValue.of_string(self.value * count)

    def binary_op_r(self, ctx: ComptimeInterpreterContext, other: CValue, op: str) -> CValue | None:
        if op == "+" and StringCType().contains(other.type):
            assert isinstance(other, StringCValue)
            return CValue.of_string(other.value + self.value)

        return self.binary_op(ctx, op, other)


@dataclass(slots=True)
class TupleCValue(CValue):
    values: list[CValue]

    def __str__(self):
        return f"({', '.join(map(str, self.values))})"

    @property
    def type(self) -> CType:
        return TupleCType([v.type for v in self.values])

    def to_runtime(self, ctx: CompilationContext, i_ctx: ComptimeInterpreterContext) -> Value:
        return Value.of_tuple(ctx, [v.to_runtime(ctx, i_ctx) for v in self.values])

    def unpackable(self) -> bool:
        return True

    def unpack(self) -> list[CValue]:
        return self.values

    def iterable(self) -> bool:
        return True

    def iterate(self, ctx: ComptimeInterpreterContext) -> Iterator[CValue]:
        return iter(self.values)

    def into_impl(self, ctx: ComptimeInterpreterContext, type_: CType) -> CValue | None:
        if isinstance(type_, TupleCType):
            types = type_.types
            if len(types) == len(self.values):
                values = []
                for value, t, in zip(self.values, types):
                    values.append(value.into_req(ctx, t))
                return CValue.of_tuple(values)

    def unary_op(self, ctx: ComptimeInterpreterContext, op: str) -> CValue | None:
        results = []
        for val in self.values:
            if (result := val.unary_op(ctx, op)) is None:
                ctx.error(f"Operator {op} is not supported for value of type {val.type}")
            results.append(result)
        return CValue.of_tuple(results)

    @staticmethod
    def _do_binary_op(ctx: ComptimeInterpreterContext, left: list[CValue], op: str, right: list[CValue]) -> CValue:
        results = []
        for a, b in zip(left, right):
            results.append(a.do_binary_op(ctx, op, b))
        return CValue.of_tuple(results)

    def binary_op(self, ctx: ComptimeInterpreterContext, op: str, other: CValue) -> CValue | None:
        if other.unpackable():
            values = other.unpack_req(ctx)

            if op == "++":
                return CValue.of_tuple(self.values + values)

            if len(self.values) == len(values):
                return self._do_binary_op(ctx, self.values, op, values)
            else:
                ctx.error(f"Element count mismatch: {len(self.values)} and {len(values)}")

        else:
            return self._do_binary_op(ctx, self.values, op, [other] * len(self.values))

    def binary_op_r(self, ctx: ComptimeInterpreterContext, other: CValue, op: str) -> CValue | None:
        if other.unpackable():
            values = other.unpack_req(ctx)

            if op == "++":
                return CValue.of_tuple(values + self.values)

            if len(self.values) == len(values):
                return self._do_binary_op(ctx, values, op, self.values)
            else:
                ctx.error(f"Element count mismatch: {len(values)} and {len(self.values)}")

        else:
            return self._do_binary_op(ctx, [other] * len(self.values), op, self.values)

    def getattr(self, ctx: ComptimeInterpreterContext, name: str, static: bool) -> BaseCValue | None:
        if not static:
            if name == "len":
                return CValue.of_number(len(self.values))

            elif len(self.values) >= 1 and name == "split":
                return CValue.of_tuple([self.values[0], CValue.of_tuple(self.values[1:])])

            try:
                i = int(name)
            except ValueError:
                return
            else:
                if 0 <= i < len(self.values):
                    return TupleElementCLValue(self, i)

    def index(self, ctx: ComptimeInterpreterContext, indices: list[CValue]) -> BaseCValue | None:
        if len(indices) != 1:
            return
        index = indices[0]
        if not NumberCType().contains(index.type):
            return
        assert isinstance(index, NumberCValue)
        i = int(index.value)
        if 0 <= i < len(self.values):
            return TupleElementCLValue(self, i)
        else:
            ctx.error(f"Tuple index out of range: {i}")


@dataclass(slots=True)
class OpaqueCValue(CValue):
    value: Value

    def __str__(self):
        return f"Opaque[{self.value.type}]"

    @property
    def type(self) -> CType:
        return OpaqueCType()

    def to_runtime(self, ctx: CompilationContext, i_ctx: ComptimeInterpreterContext) -> Value:
        return self.value


def from_runtime_type(ctx: ErrorContext, type_: Type) -> CType:
    if isinstance(type_, NullType):
        return NullCType()

    elif isinstance(type_, NumberType):
        return NumberCType()

    elif isinstance(type_, StringType):
        return StringCType()

    elif isinstance(type_, TupleType):
        return TupleCType([from_runtime_type(ctx, t) for t in type_.types])

    ctx.error(f"Cannot convert type '{type_}' to compile time")


def from_runtime(ctx: CompilationContext, value: Value) -> CValue:
    if NumberType().contains(value.type):
        try:
            return CValue.of_float(float(value.value))
        except ValueError:
            pass

    elif StringType().contains(value.type):
        if value.value.startswith("\"") and value.value.endswith("\""):
            return CValue.of_string(value.value[1:-1])

    elif isinstance(value.type, TupleType):
        return CValue.of_tuple([from_runtime(ctx, val) for val in value.unpack_req(ctx)])

    elif isinstance(value.type, TypeType):
        return TypeCValue(from_runtime_type(ctx, value.type.wrapped_type(ctx)))

    elif (val := value.into(ctx, NumberType())) is not None:
        try:
            return CValue.of_float(float(val.value))
        except ValueError:
            pass

    elif (val := value.into(ctx, StringType())) is not None:
        if val.value.startswith("\"") and val.value.endswith("\""):
            return CValue.of_string(val.value[1:-1])

    else:
        OpaqueCValue(value)


@dataclass(slots=True)
class FunctionCValue(CValue):
    name: str
    params: list[CType | None]
    param_names: list[str]
    result: CType | None
    function: Callable[[list[CValue]], CValue]

    def __str__(self):
        return f"comptime fn {self.name}({', '.join(str(p) if p is not None else '?' for p in self.params)}) \
-> {self.result if self.result is not None else '?'}"

    @staticmethod
    def to_runtime_impl(self: CValue, i_ctx: ComptimeInterpreterContext,
                        name: str, params: list[CType | None], result: CType | None) -> Value:

        assert self.callable()

        def func_wrapper(ctx_: CompilationContext, params_: list[Value]) -> Value:
            converted = [from_runtime(ctx_, p) for p in params_]
            converted = [p.into_req(i_ctx, t) if t is not None else p for p, t in zip(converted, params)]
            if (exp := self.callable_with([p.type for p in converted])) is None:
                return self.call(i_ctx, converted).to_runtime(ctx_, i_ctx)
            else:
                ctx.error(f"Cannot call function with parameters of types \
({', '.join(f'\'{p.type}\'' for p in converted)}), expected ({', '.join(f'\'{t}\'' for t in exp)})")

        return Value(SpecialFunctionType(
            name,
            [t.to_runtime() if t is not None else None for t in params],
            result if result is not None else None,
            func_wrapper
        ), "", True)

    def to_runtime(self, ctx: CompilationContext, i_ctx: ComptimeInterpreterContext) -> Value:
        return self.to_runtime_impl(self, i_ctx, self.name, self.params, self.result)

    @property
    def type(self) -> CType:
        return FunctionCType(self.params, self.result)

    def callable(self) -> bool:
        return True

    def callable_with(self, types: list[CType]) -> list[CType | None] | None:
        return None if len(types) == len(self.params) and all(
            p is None or p.contains(t) for p, t in zip(self.params, types)) else self.params

    def call(self, ctx: ComptimeInterpreterContext, values: list[CValue]) -> CValue:
        with ctx.scope(ctx.tmp()):
            for name, type_, value in zip(self.param_names, self.params, values):
                assert ctx.scope.declare(name, value if type_ is None else value.into_req(ctx, type_), False)
            try:
                result = self.function(values)
            except Return as e:
                result = e.value if e.value is not None else CValue.null()
            result_type = self.result
            if result_type is not None:
                result = result.into_req(ctx, result_type)
            return result

    def into_impl(self, ctx: ComptimeInterpreterContext, type_: CType) -> CValue | None:
        if isinstance(type_, FunctionCType):
            if len(self.params) == len(type_.params):
                params = []
                for a, b in zip(self.params, type_.params):
                    if a is None or a.contains(b):
                        params.append(b)
                    else:
                        return None
                if type_.result is None or type_.result.contains(self.result):
                    return FunctionCValue(self.name, params, self.param_names, type_.result, self.function)
                elif self.result is None:
                    return FunctionCValue(self.name, params, self.param_names, type_.result,
                                          lambda p: self.function(p).into_req(ctx, type_.result))


@dataclass
class LambdaCValue(CValue):
    params: list[CType | None]
    param_names: list[str]
    captures: dict[str, VariableCLValue]
    result: CType | None
    function: Callable[[list[CValue]], CValue]

    def __str__(self):
        if len(self.captures) == 0:
            return f"comptime |{', '.join(str(p) if p is not None else '?' for p in self.params)}| \
-> {self.result if self.result is not None else '?'}"
        else:
            return f"comptime |{', '.join(str(p) if p is not None else '?' for p in self.params)}|\
[{', '.join(self.captures.keys())}] -> {self.result if self.result is not None else '?'}"

    def to_runtime(self, ctx: CompilationContext, i_ctx: ComptimeInterpreterContext) -> Value:
        return FunctionCValue.to_runtime_impl(self, i_ctx, "__comptime_lambda", self.params, self.result)

    @property
    def type(self) -> CType:
        return FunctionCType(self.params, self.result)

    def callable(self) -> bool:
        return True

    def callable_with(self, types: list[CType]) -> list[CType | None] | None:
        return None if len(types) == len(self.params) and all(
            p is None or p.contains(t) for p, t in zip(self.params, types)) else self.params

    def call(self, ctx: ComptimeInterpreterContext, values: list[CValue]) -> CValue:
        with ctx.scope(ctx.tmp(), self.captures):
            with ctx.scope(ctx.tmp()):
                for name, type_, value in zip(self.param_names, self.params, values):
                    assert ctx.scope.declare(name, value if type_ is None else value.into_req(ctx, type_), False)
                try:
                    result = self.function(values)
                except Return as e:
                    result = e.value if e.value is not None else CValue.null()
                result_type = self.result
                if result_type is not None:
                    result = result.into_req(ctx, result_type)
                return result

    def into_impl(self, ctx: ComptimeInterpreterContext, type_: CType) -> CValue | None:
        if isinstance(type_, FunctionCType):
            if len(self.params) == len(type_.params):
                params = []
                for a, b in zip(self.params, type_.params):
                    if a is None or a.contains(b):
                        params.append(b)
                    else:
                        return None
                if type_.result is None or type_.result.contains(self.result):
                    return LambdaCValue(params, self.param_names, self.captures, type_.result, self.function)
                elif self.result is None:
                    return LambdaCValue(params, self.param_names, self.captures, type_.result,
                                        lambda p: self.function(p).into_req(ctx, type_.result))


@dataclass(slots=True)
class RangeCValue(CValue):
    start: int | float
    end: int | float

    def __str__(self):
        return f"{self.start}..{self.end}"

    def to_runtime(self, ctx: CompilationContext, i_ctx: ComptimeInterpreterContext) -> Value:
        return Value.of_range(ctx, Value.of_number(self.start), Value.of_number(self.end))

    @property
    def type(self) -> CType:
        return RangeCType()

    def getattr(self, ctx: ComptimeInterpreterContext, name: str, static: bool) -> BaseCValue | None:
        if not static:
            if name == "start":
                return CValue.of_number(self.start)
            elif name == "end":
                return CValue.of_number(self.end)

    def iterable(self) -> bool:
        return True

    def iterate(self, ctx: ComptimeInterpreterContext) -> Iterator[CValue]:
        i = self.start
        while i < self.end:
            yield CValue.of_number(i)
            i += 1

    def unpackable(self) -> bool:
        return True

    def unpack(self) -> list[CValue]:
        values = []
        i = self.start
        while i < self.end:
            values.append(CValue.of_number(i))
            i += 1
        return values


@dataclass(slots=True)
class RangeWithStepCValue(CValue):
    start: int | float
    end: int | float
    step: int | float

    def __str__(self):
        return f"{self.start}..{self.end}..{self.step}"

    def to_runtime(self, ctx: CompilationContext, i_ctx: ComptimeInterpreterContext) -> Value:
        return Value.of_range_with_step(ctx, Value.of_number(self.start), Value.of_number(self.end),
                                        Value.of_number(self.step))

    @property
    def type(self) -> CType:
        return RangeCType()

    def getattr(self, ctx: ComptimeInterpreterContext, name: str, static: bool) -> BaseCValue | None:
        if not static:
            if name == "start":
                return CValue.of_number(self.start)
            elif name == "end":
                return CValue.of_number(self.end)
            elif name == "step":
                return CValue.of_number(self.step)

    def iterable(self) -> bool:
        return True

    def iterate(self, ctx: ComptimeInterpreterContext) -> Iterator[CValue]:
        i = self.start
        while i < self.end:
            yield CValue.of_number(i)
            i += self.step

    def unpackable(self) -> bool:
        return True

    def unpack(self) -> list[CValue]:
        values = []
        i = self.start
        while i < self.end:
            values.append(CValue.of_number(i))
            i += self.step
        return values
