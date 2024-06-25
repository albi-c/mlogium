from __future__ import annotations

from dataclasses import dataclass
from abc import abstractmethod, ABC
from typing import Callable

from .compilation_context import CompilationContext
from .instruction import Instruction
from .abi import ABI


@dataclass(slots=True)
class Value:
    type: Type
    value: str
    const: bool = True
    const_on_write: bool = False

    @classmethod
    def null(cls) -> Value:
        return Value(NullType(), "null")

    @classmethod
    def of_type(cls, type_: Type) -> Value:
        return Value(TypeType(type_), "")

    @classmethod
    def of_number(cls, value: int | float | str) -> Value:
        return Value(NumberType(), str(value))

    @classmethod
    def of_string(cls, value: str) -> Value:
        return Value(StringType(), value)

    @classmethod
    def of_tuple(cls, ctx: CompilationContext, values: list[Value]) -> Value:
        tup = Value(TupleType([val.type for val in values]), ctx.tmp(), False)
        for dst, src in zip(tup.unpack(ctx), values):
            dst.assign(ctx, src)
        return tup

    @classmethod
    def of_range(cls, ctx: CompilationContext, start: Value, end: Value) -> Value:
        val = Value(RangeType(), ctx.tmp(), False)
        val.getattr_req(ctx, False, "start").assign(ctx, start)
        val.getattr_req(ctx, False, "end").assign(ctx, end)
        return val

    def assign(self, ctx: CompilationContext, other: Value):
        if self.const:
            ctx.error(f"Assignment to constant of type '{self.type}'")

        self.type.assign(ctx, self, other.into_req(ctx, self.assignable_type()))

        if self.const_on_write:
            self.const = True

    def assignable_type(self) -> Type:
        return self.type.assignable_type()

    def to_strings(self, ctx: CompilationContext) -> list[str]:
        return self.type.to_strings(ctx, self)

    def to_condition(self, ctx: CompilationContext) -> str | None:
        return self.type.to_condition(ctx, self)

    def to_condition_req(self, ctx: CompilationContext) -> str:
        if (val := self.to_condition(ctx)) is None:
            ctx.error(f"Value of type '{self.type}' cannot be used as a condition")
        return val

    def unpackable(self) -> bool:
        return self.unpack_count() >= 0

    def unpack_count(self) -> int:
        return self.type.unpack_count()

    def unpack(self, ctx: CompilationContext) -> list[Value]:
        return self.type.unpack(ctx, self)

    def unpack_req(self, ctx: CompilationContext) -> list[Value]:
        if not self.unpackable():
            ctx.error(f"Value of type '{self.type}' is not unpackable")
        return self.unpack(ctx)

    def callable(self) -> bool:
        return self.type.callable()

    def call_with_suggestion(self) -> list[Type | None] | None:
        return self.type.call_with_suggestion()

    def callable_with(self, param_types: list[Type]) -> bool:
        return self.type.callable_with(param_types)

    def call(self, ctx: CompilationContext, params: list[Value]) -> Value:
        return self.type.call(ctx, self, params)

    def getattr(self, ctx: CompilationContext, static: bool, name: str) -> Value | None:
        return self.type.getattr(ctx, self, static, name)

    def getattr_req(self, ctx: CompilationContext, static: bool, name: str) -> Value | None:
        if (val := self.getattr(ctx, static, name)) is None:
            ctx.error(f"Value of type '{self.type}' has no {'static ' if static else ''} attribute '{name}'")
        return val

    def indexable(self) -> bool:
        return self.type.indexable()

    # return -1 if valid, else required count
    def validate_index_count(self, count: int) -> int:
        return self.type.validate_index_count(count)

    def index(self, ctx: CompilationContext, indices: list[Value]) -> Value:
        return self.type.index(ctx, self, indices)

    def into(self, ctx: CompilationContext, type_: Type) -> Value | None:
        return self.type.into(ctx, self, type_)

    def into_req(self, ctx: CompilationContext, type_: Type) -> Value:
        if (val := self.into(ctx, type_)) is None:
            ctx.error(f"Incompatible types: '{type_}', '{self.type}'")
        return val

    def binary_op(self, ctx: CompilationContext, op: str, other: Value) -> Value | None:
        return self.type.binary_op(ctx, self, op, other)

    def binary_op_r(self, ctx: CompilationContext, other: Value, op: str) -> Value | None:
        return self.type.binary_op_r(ctx, other, op, self)

    def unary_op(self, ctx: CompilationContext, op: str) -> Value | None:
        return self.type.unary_op(ctx, self, op)


def _stringify(val) -> str:
    return f"\"{val}\""


class Type(ABC):
    EQUALITY_OPS = {
        "==": "equal",
        "!=": "notEqual",
        "===": "strictEqual"
    }

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    def contains(self, other: Type) -> bool:
        return self == other

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        ctx.emit(Instruction.set(value.value, other.value))

    def assignable_type(self) -> Type:
        return self

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [value.value]

    def to_condition(self, ctx: CompilationContext, value: Value) -> str | None:
        return None

    def unpack_count(self) -> int:
        return -1

    def unpack(self, ctx: CompilationContext, value: Value) -> list[Value]:
        raise NotImplementedError

    def callable(self) -> bool:
        return False

    def call_with_suggestion(self) -> list[Type] | None:
        return None

    def callable_with(self, param_types: list[Type]) -> bool:
        return False

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        raise NotImplementedError

    def getattr(self, ctx: CompilationContext, value: Value, static: bool, name: str) -> Value | None:
        return None

    def indexable(self) -> bool:
        return False

    def validate_index_count(self, count: int) -> int:
        return -1

    def index(self, ctx: CompilationContext, value: Value, indices: list[Value]) -> Value:
        raise NotImplementedError

    def into(self, ctx: CompilationContext, value: Value, type_: Type) -> Value | None:
        if type_.contains(self):
            return value

    def binary_op(self, ctx: CompilationContext, value: Value, op: str, other: Value) -> Value | None:
        if op in ("=", ":="):
            value.assign(ctx, other)
            return value

        elif (operator := self.EQUALITY_OPS.get(op)) is not None:
            # TODO: properly implement for tuples and structs
            result = Value(NumberType(), ctx.tmp(), False)
            ctx.emit(Instruction.op(operator, result.value, value.value, other.value))
            return result

        elif op == "!==":
            return value.binary_op(ctx, "===", other).unary_op(ctx, "!")

        elif op.endswith("=") and op not in ("<=", ">=", ":="):
            value.assign(ctx, value.binary_op(ctx, op[:-1], other))
            return value

        return other.binary_op_r(ctx, value, op)

    def binary_op_r(self, ctx: CompilationContext, other: Value, op: str, value: Value) -> Value | None:
        return None

    def unary_op(self, ctx: CompilationContext, value: Value, op: str) -> Value | None:
        if op == "!":
            condition = value.to_condition_req(ctx)
            result = Value(NumberType(), ctx.tmp(), False)
            ctx.emit(Instruction.op("notEqual", result.value, condition, "0"))
            return result


class AnyType(Type):
    def __str__(self):
        return "?"

    def __eq__(self, other):
        return isinstance(other, AnyType)

    def contains(self, other: Type) -> bool:
        return True

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        raise NotImplementedError

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        raise NotImplementedError


@dataclass(slots=True)
class TypeType(Type):
    type: Type

    def __str__(self):
        return f"Type[{self.type}]"

    def __eq__(self, other):
        return isinstance(other, TypeType) and self.type == other.type

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        pass

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [_stringify(str(self))]


class NullType(Type):
    def __str__(self):
        return "null"

    def __eq__(self, other):
        return isinstance(other, NullType)

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        pass

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [_stringify("null")]


class NumberType(Type):
    BINARY_OPS = {
        "+": "add",
        "-": "sub",
        "*": "mul",
        "/": "div",
        "//": "idiv",
        "%": "mod",
        "**": "pow",
        "&&": "land",
        "||": "or",
        "<": "lessThan",
        "<=": "lessThanEq",
        ">": "greaterThan",
        ">=": "greaterThanEq",
        "<<": "shl",
        ">>": "shr",
        "|": "or",
        "&": "and",
        "^": "xor"
    }

    def __str__(self):
        return "num"

    def __eq__(self, other):
        return isinstance(other, NumberType)

    def to_condition(self, ctx: CompilationContext, value: Value) -> str | None:
        return value.value

    def unary_op(self, ctx: CompilationContext, value: Value, op: str) -> Value | None:
        if op == "-":
            result = Value(NumberType(), ctx.tmp(), False)
            ctx.emit(Instruction.op("sub", result.value, "0", value.value))
            return result

        elif op == "~":
            result = Value(NumberType(), ctx.tmp(), False)
            ctx.emit(Instruction.op("flip", result.value, value.value, "_"))
            return result

        return super().unary_op(ctx, value, op)

    def binary_op(self, ctx: CompilationContext, value: Value, op: str, other: Value) -> Value | None:
        if (other_num := other.into(ctx, NumberType())) is not None and (operator := self.BINARY_OPS.get(op)):
            result = Value(NumberType(), ctx.tmp(), False)
            ctx.emit(Instruction.op(operator, result.value, value.value, other_num.value))
            return result

        return super().binary_op(ctx, value, op, other)


class StringType(Type):
    def __str__(self):
        return "str"

    def __eq__(self, other):
        return isinstance(other, StringType)

    def to_condition(self, ctx: CompilationContext, value: Value) -> str | None:
        return value.value


@dataclass(slots=True)
class TupleType(Type):
    types: list[Type]

    def __str__(self):
        return f"({', '.join(map(str, self.types))})"

    def __eq__(self, other):
        return isinstance(other, TupleType) and self.types == other.types

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        for dst, src in zip(value.unpack(ctx), other.unpack(ctx)):
            dst.assign(ctx, src)

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        strings = ["\"(\""]
        for value in value.unpack(ctx):
            if len(strings) > 1:
                strings.append("\", \"")
            strings += value.to_strings(ctx)
        strings.append("\")\"")
        return strings

    def unpack_count(self) -> int:
        return len(self.types)

    def unpack(self, ctx: CompilationContext, value: Value) -> list[Value]:
        return [Value(t, ABI.attribute(value.value, i), value.const) for i, t in enumerate(self.types)]


class RangeType(Type):
    def __str__(self):
        return f"Range"

    def __eq__(self, other):
        return isinstance(other, RangeType)

    @staticmethod
    def _attr(value: Value, name: str, const: bool) -> Value:
        return Value(NumberType(), ABI.attribute(value.value, name), const)

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        self._attr(value, "start", value.const).assign(ctx, other.getattr_req(ctx, False, "start"))
        self._attr(value, "end", value.const).assign(ctx, other.getattr_req(ctx, False, "end"))

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [ABI.attribute(value.value, "start"), "\"..\"", ABI.attribute(value.value, "end")]

    def getattr(self, ctx: CompilationContext, value: Value, static: bool, name: str) -> Value | None:
        if not static:
            if name in ("start", "end"):
                return self._attr(value, name, value.const)

        return super().getattr(ctx, value, static, name)


def _format_function_signature(params: list[Type | None], result: Type | None) -> str:
    return f"({', '.join(str(p) if p is not None else '?' for p in params)}) -> {result if result is not None else '?'}"


def _check_params(expected: list[Type | None], provided: list[Type]) -> bool:
    return all(e is None or e.contains(t) for e, t in zip(expected, provided))


@dataclass(slots=True)
class SpecialFunctionType(Type):
    name: str
    params: list[Type | None]
    result: Type | None
    function: Callable[[CompilationContext, list[Value]], Value]

    def __str__(self):
        return f"fn {self.name}{_format_function_signature(self.params, self.result)}"

    def __eq__(self, other):
        return isinstance(other, SpecialFunctionType) and self.name == other.name and self.params == other.params and self.result == other.result

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        pass

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [_stringify(str(self))]

    def callable(self) -> bool:
        return True

    def call_with_suggestion(self) -> list[Type | None] | None:
        return self.params

    def callable_with(self, param_types: list[Type]) -> bool:
        return _check_params(self.params, param_types)

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        return self.function(ctx, params)
