from __future__ import annotations

import contextlib
from dataclasses import dataclass
from abc import abstractmethod, ABC
from typing import Callable

from .compilation_context import CompilationContext
from .instruction import Instruction
from .abi import ABI
from .node import Node


@dataclass(slots=True)
class Value:
    type: Type
    value: str
    const: bool = True
    const_on_write: bool = False
    no_discard: bool = False

    def with_no_discard(self, value: bool = True) -> Value:
        return Value(self.type, self.value, self.const, self.const_on_write, value)

    @classmethod
    def null(cls) -> Value:
        return Value(NullType(), "null")

    @classmethod
    def make_default(cls, ctx: CompilationContext, type_: Type) -> Value:
        val = Value(type_, ctx.tmp(), False)
        val.assign_default(ctx)
        return val

    @classmethod
    def of_type(cls, type_: Type) -> Value:
        return Value(TypeType(type_), "")

    @classmethod
    def of_number(cls, value: int | float | str) -> Value:
        return Value(NumberType(), str(value))

    @classmethod
    def of_boolean(cls, value: bool) -> Value:
        return Value.of_number(int(value))

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

    @classmethod
    def of_range_with_step(cls, ctx: CompilationContext, start: Value, end: Value, step: Value) -> Value:
        val = Value(RangeWithStepType(), ctx.tmp(), False)
        val.getattr_req(ctx, False, "start").assign(ctx, start)
        val.getattr_req(ctx, False, "end").assign(ctx, end)
        val.getattr_req(ctx, False, "step").assign(ctx, step)
        return val

    @contextlib.contextmanager
    def do_assignment(self, ctx: CompilationContext):
        if self.const:
            ctx.error(f"Assignment to constant of type '{self.type}'")

        try:
            yield

        finally:
            if self.const_on_write:
                self.const = True

    def assign(self, ctx: CompilationContext, other: Value):
        if NullType().contains(other.type):
            return self.assign_default(ctx)

        with self.do_assignment(ctx):
            self.type.assign(ctx, self, other.into_req(ctx, self.assignable_type()))

    def assign_default(self, ctx: CompilationContext):
        with self.do_assignment(ctx):
            self.type.assign_default(ctx, self)

    def copy(self, ctx: CompilationContext, name: str = None) -> Value:
        if name is None:
            name = ctx.tmp()
        val = Value(self.type, name, False)
        val.assign(ctx, self)
        return val

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

    def unpackable(self, ctx: CompilationContext) -> bool:
        return self.type.unpackable(ctx, self)

    def unpack_count(self, ctx: CompilationContext) -> int | None:
        return self.type.unpack_count(ctx, self)

    def unpack(self, ctx: CompilationContext) -> list[Value]:
        return self.type.unpack(ctx, self)

    def unpack_req(self, ctx: CompilationContext) -> list[Value]:
        if not self.unpackable(ctx):
            ctx.error(f"Value of type '{self.type}' is not unpackable")
        return self.unpack(ctx)

    def callable(self, ctx: CompilationContext) -> bool:
        return self.type.callable(ctx, self)

    def call_with_suggestion(self, ctx: CompilationContext) -> list[Type | None] | None:
        return self.type.call_with_suggestion(ctx, self)

    def callable_with(self, ctx: CompilationContext, param_types: list[Type]) -> bool:
        if not self.callable(ctx):
            return False

        # To stop IDE from complaining that ctx has type 'Callable' instead of 'CompilationContext'
        assert isinstance(ctx, CompilationContext)

        return self.type.callable_with(ctx, self, param_types)

    def call(self, ctx: CompilationContext, params: list[Value]) -> Value:
        return self.type.call(ctx, self, params)

    def getattr(self, ctx: CompilationContext, static: bool, name: str) -> Value | None:
        return self.type.getattr(ctx, self, static, name)

    def getattr_req(self, ctx: CompilationContext, static: bool, name: str) -> Value | None:
        if (val := self.getattr(ctx, static, name)) is None:
            ctx.error(f"Value of type '{self.type}' has no {'static ' if static else ''}attribute '{name}'")
        return val

    def indexable(self, ctx: CompilationContext) -> bool:
        return self.type.indexable(ctx, self)

    # return None if valid, else required types
    def validate_index_types(self, ctx: CompilationContext, indices: list[Type]) -> None | list[Type]:
        return self.type.validate_index_types(ctx, self, indices)

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

    def binary_op_req(self, ctx: CompilationContext, op: str, other: Value) -> Value | None:
        if (result := self.binary_op(ctx, op, other)) is None:
            ctx.error(f"Operator '{op}' is not supported between values of types '{self.type}' and '{other.type}'")
        return result

    def binary_op_r(self, ctx: CompilationContext, other: Value, op: str) -> Value | None:
        return self.type.binary_op_r(ctx, other, op, self)

    def unary_op(self, ctx: CompilationContext, op: str) -> Value | None:
        return self.type.unary_op(ctx, self, op)

    def unary_op_req(self, ctx: CompilationContext, op: str) -> Value | None:
        if (result := self.unary_op(ctx, op)) is None:
            ctx.error(f"Operator '{op}' is not supported for value of type '{self.type}'")
        return result

    def bottom_scope(self) -> dict[str, Value] | None:
        return self.type.bottom_scope()

    def iterable(self, ctx: CompilationContext) -> bool:
        return self.iterate(ctx) is not None

    def iterate(self, ctx: CompilationContext) -> ValueIterator | None:
        return self.type.iterate(ctx, self)

    def iterate_req(self, ctx: CompilationContext) -> ValueIterator:
        if (val := self.iterate(ctx)) is None:
            ctx.error(f"Value of type '{self.type}' is not iterable")
        return val

    def memcell_serializable(self, ctx: CompilationContext) -> bool:
        return self.type.memcell_serializable(ctx, self)

    def memcell_size(self, ctx: CompilationContext) -> int:
        return self.type.memcell_size(ctx, self)

    def memcell_write(self, ctx: CompilationContext) -> list[str]:
        return self.type.memcell_write(ctx, self)

    def memcell_read(self, ctx: CompilationContext, values: list[str]):
        with self.do_assignment(ctx):
            self.type.memcell_read(ctx, self, values)

    def table_copyable(self, ctx: CompilationContext) -> bool:
        return self.type.table_copyable(ctx, self)

    def table_size(self, ctx: CompilationContext) -> int:
        return self.type.table_size(ctx, self)

    def table_variables(self, ctx: CompilationContext) -> list[str]:
        return self.type.table_variables(ctx, self)


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

    def wrapped_type(self, ctx: CompilationContext) -> Type:
        ctx.error(f"Value of type '{self}' is not a type")
        return NullType()

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        ctx.emit(Instruction.set(value.value, other.value))

    def assign_default(self, ctx: CompilationContext, value: Value):
        self.assign(ctx, value, Value(self, "null"))

    def assignable_type(self) -> Type:
        return self

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [value.value]

    def to_condition(self, ctx: CompilationContext, value: Value) -> str | None:
        return None

    def unpackable(self, ctx: CompilationContext, value: Value) -> bool:
        return self.unpack_count(ctx, value) is not None

    def unpack_count(self, ctx: CompilationContext, value: Value) -> int | None:
        return None

    def unpack(self, ctx: CompilationContext, value: Value) -> list[Value]:
        raise NotImplementedError

    def callable(self, ctx: CompilationContext, value: Value) -> bool:
        return False

    def call_with_suggestion(self, ctx: CompilationContext, value: Value) -> list[Type | None] | None:
        return None

    def callable_with(self, ctx: CompilationContext, value: Value, param_types: list[Type]) -> bool:
        return False

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        raise NotImplementedError

    def getattr(self, ctx: CompilationContext, value: Value, static: bool, name: str) -> Value | None:
        return None

    def indexable(self, ctx: CompilationContext, value: Value) -> bool:
        return False

    def validate_index_types(self, ctx: CompilationContext, value: Value, indices: list[Type]) -> None | list[Type]:
        return []

    def index(self, ctx: CompilationContext, value: Value, indices: list[Value]) -> Value:
        raise NotImplementedError

    def into(self, ctx: CompilationContext, value: Value, type_: Type) -> Value | None:
        if type_.contains(self):
            return value

        elif isinstance(type_, UnionType):
            for t in type_.types:
                if (val := value.into(ctx, t)) is not None:
                    return val

        return type_.construct_from(ctx, value)

    def construct_from(self, ctx: CompilationContext, value: Value) -> Value | None:
        return None

    def binary_op(self, ctx: CompilationContext, value: Value, op: str, other: Value) -> Value | None:
        if op in ("=", ":="):
            value.assign(ctx, other)
            return value.with_no_discard(False)

        elif (operator := self.EQUALITY_OPS.get(op)) is not None:
            # TODO: properly implement for tuples and structs
            result = Value(NumberType(), ctx.tmp(), False)
            ctx.emit(Instruction.op(operator, result.value, value.value, other.value))
            return result

        elif op == "!==":
            return value.binary_op(ctx, "===", other).unary_op(ctx, "!")

        elif op.endswith("=") and op not in ("<=", ">=", ":="):
            value.assign(ctx, value.binary_op(ctx, op[:-1], other))
            return value.with_no_discard(False)

        return other.binary_op_r(ctx, value, op)

    def binary_op_r(self, ctx: CompilationContext, other: Value, op: str, value: Value) -> Value | None:
        return None

    def unary_op(self, ctx: CompilationContext, value: Value, op: str) -> Value | None:
        if op == "!":
            condition = value.to_condition_req(ctx)
            result = Value(NumberType(), ctx.tmp(), False)
            ctx.emit(Instruction.op("notEqual", result.value, condition, "0"))
            return result

    def bottom_scope(self) -> dict[str, Value] | None:
        return None

    def iterate(self, ctx: CompilationContext, value: Value) -> ValueIterator | None:
        pass

    def memcell_serializable(self, ctx: CompilationContext, value: Value) -> bool:
        return False

    def memcell_size(self, ctx: CompilationContext, value: Value) -> int:
        return 1

    def memcell_write(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [value.value]

    def memcell_read(self, ctx: CompilationContext, value: Value, values: list[str]):
        ctx.emit(Instruction.set(value.value, values[0]))

    def table_copyable(self, ctx: CompilationContext, value: Value) -> bool:
        return value.memcell_serializable(ctx)

    def table_size(self, ctx: CompilationContext, value: Value) -> int:
        return value.memcell_size(ctx)

    def table_variables(self, ctx: CompilationContext, value: Value) -> list[str]:
        return value.memcell_write(ctx)


class ValueIterator(ABC):
    @abstractmethod
    def has_value(self, ctx: CompilationContext) -> Value:
        raise NotImplementedError

    @abstractmethod
    def next_value(self, ctx: CompilationContext) -> Value:
        raise NotImplementedError

    @abstractmethod
    def end_loop(self, ctx: CompilationContext):
        raise NotImplementedError


class AnyType(Type):
    def __str__(self):
        return "?"

    def __eq__(self, other):
        return isinstance(other, AnyType)

    def contains(self, other: Type) -> bool:
        return True

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        ctx.error("Cannot assign to value of unknown type")

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        ctx.error("Cannot print value of unknown type")
        return []


class AnyTrivialType(Type):
    def __str__(self):
        return "?"

    def __eq__(self, other):
        return isinstance(other, AnyTrivialType)

    def contains(self, other: Type) -> bool:
        return UnionType([NumberType(), StringType(), BlockType(),
                          UnitType(), BlockType(), ControllerType()]).contains(other) \
            or isinstance(other, EnumInstanceType) \
            or (isinstance(other, BuiltinEnumInstanceType) and other.base.copyable)

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        ctx.emit(
            Instruction.write(value.value, other.value)
        )

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [value.value]


class EllipsisType(Type):
    def __str__(self):
        return "..."

    def __eq__(self, other):
        return isinstance(other, EllipsisType)

    def contains(self, other: Type) -> bool:
        return True

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        raise NotImplementedError

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        raise NotImplementedError


class UnderscoreType(Type):
    def __str__(self):
        return "?"

    def __eq__(self, other):
        return isinstance(other, UnderscoreType)

    def contains(self, other: Type) -> bool:
        return True

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        pass

    def assign_default(self, ctx: CompilationContext, value: Value):
        pass

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [_stringify(str(self))]


@dataclass(slots=True)
class TypeType(Type):
    type: Type

    def __str__(self):
        return f"Type[{self.type}]"

    def __eq__(self, other):
        return isinstance(other, TypeType) and self.type == other.type

    def wrapped_type(self, ctx: CompilationContext) -> Type:
        return self.type

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        pass

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [_stringify(str(self))]

    def getattr(self, ctx: CompilationContext, value: Value, static: bool, name: str) -> Value | None:
        if not static:
            val = Value(self.type, "null")

            if name == "name":
                return Value.of_string(_stringify(str(self.type)))

            elif name == "equals":
                return Value(SpecialFunctionType(
                    f"{str(self)}.equals",
                    [AnyType()],
                    NumberType(),
                    lambda ctx_, params: Value.of_boolean(self.type == params[0].type.wrapped_type(ctx_))
                ), "")

            elif name == "contains":
                return Value(SpecialFunctionType(
                    f"{str(self)}.contains",
                    [AnyType()],
                    NumberType(),
                    lambda ctx_, params: Value.of_boolean(self.type.contains(params[0].type.wrapped_type(ctx_)))
                ), "")

            elif name == "callable":
                return Value.of_boolean(val.callable(ctx))

            elif name == "callable_with":
                return Value(SpecialFunctionType(
                    f"{str(self)}.callable_with",
                    [AnyType()],
                    NumberType(),
                    lambda ctx_, params: Value.of_boolean(
                        val.callable_with(ctx_, [p.type.wrapped_type(ctx_) for p in params[0].unpack_req(ctx_)]))
                ), "")

            elif name == "unpackable":
                return Value.of_boolean(val.unpackable(ctx))

            elif name == "len" and (count := val.unpack_count(ctx)) is not None:
                return Value.of_number(count)

            elif name == "serializable":
                return Value.of_boolean(val.memcell_serializable(ctx))

            elif name == "size":
                return Value.of_number(val.memcell_size(ctx))

            elif name == "default":
                return Value(SpecialFunctionType(
                    f"{str(self)}.default",
                    [],
                    self.type,
                    lambda ctx_, _: Value.make_default(ctx_, self.type)
                ), "")

            elif name == "iterable":
                return Value.of_boolean(val.iterable(ctx))

            elif name == "is_null":
                return Value.of_boolean(NullType().contains(val.type))

            elif name == "is_condition":
                return Value.of_boolean(val.to_condition(ctx) is not None)

            elif name == "is_tuple":
                return Value.of_boolean(isinstance(self.type, TupleType))

            elif name == "from":
                return Value(SpecialFunctionType(
                    f"{str(self)}.from",
                    [AnyType()],
                    self.type,
                    lambda ctx_, params: params[0].into_req(ctx_, self.type)
                ), "")

            elif name == "base":
                if isinstance(self.type, StructInstanceType | EnumInstanceType):
                    return Value(self.type.base, "")

                ctx.error(f"Value of type '{self.type}' is not a struct or enum instance")

        return super(TypeType, self).getattr(ctx, value, static, name)


class GenericTypeType(Type):
    def __str__(self):
        return "Type"

    def __eq__(self, other):
        return isinstance(other, GenericTypeType)

    def contains(self, other: Type) -> bool:
        return True

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        pass

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [_stringify(str(self))]


class TupleTypeSourceType(Type):
    def __str__(self):
        return "Tuple"

    def __eq__(self, other):
        return isinstance(other, TupleTypeSourceType)

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        pass

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [_stringify(str(self))]

    def indexable(self, ctx: CompilationContext, value: Value) -> bool:
        return True

    def validate_index_types(self, ctx: CompilationContext, value: Value, indices: list[Type]) -> None | list[Type]:
        return None

    def index(self, ctx: CompilationContext, value: Value, indices: list[Value]) -> Value:
        types = [val.type.wrapped_type(ctx) for val in indices]
        return Value.of_type(TupleType(types))

    def callable(self, ctx: CompilationContext, value: Value) -> bool:
        return True

    def callable_with(self, ctx: CompilationContext, value: Value, param_types: list[Type]) -> bool:
        return len(param_types) == 2 and GenericTypeType().contains(param_types[0]) \
            and NumberType().contains(param_types[1])

    def call_with_suggestion(self, ctx: CompilationContext, value: Value) -> list[Type | None] | None:
        return [GenericTypeType(), NumberType()]

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        count = 0
        try:
            count = int(params[1].value)
        except ValueError:
            ctx.error(f"Count is not a compile time known integer")
        if count < 0:
            ctx.error(f"Count cannot be negative")

        return Value.of_type(TupleType([params[0].type.wrapped_type(ctx)] * count))


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
        "/.": "idiv",
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

    def assign_default(self, ctx: CompilationContext, value: Value):
        ctx.emit(Instruction.set(value.value, "0"))

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

        return super(NumberType, self).unary_op(ctx, value, op)

    def binary_op(self, ctx: CompilationContext, value: Value, op: str, other: Value) -> Value | None:
        if (other_num := other.into(ctx, NumberType())) is not None and (operator := self.BINARY_OPS.get(op)):
            result = Value(NumberType(), ctx.tmp(), False)
            ctx.emit(Instruction.op(operator, result.value, value.value, other_num.value))
            return result

        return super(NumberType, self).binary_op(ctx, value, op, other)

    def memcell_serializable(self, ctx: CompilationContext, value: Value) -> bool:
        return True


class StringType(Type):
    def __str__(self):
        return "str"

    def __eq__(self, other):
        return isinstance(other, StringType)

    def assign_default(self, ctx: CompilationContext, value: Value):
        ctx.emit(Instruction.set(value.value, "\"\""))

    def to_condition(self, ctx: CompilationContext, value: Value) -> str | None:
        return value.value

    def table_copyable(self, ctx: CompilationContext, value: Value) -> bool:
        return True


class BlockBaseType(TypeType):
    def __init__(self):
        super().__init__(BlockType())

    def getattr(self, ctx: CompilationContext, value: Value, static: bool, name: str) -> Value | None:
        if static:
            return Value(BlockType(), name)

        return super(BlockBaseType, self).getattr(ctx, value, static, name)


class BlockType(Type):
    def __str__(self):
        return "Block"

    def __eq__(self, other):
        return isinstance(other, BlockType)

    def to_condition(self, ctx: CompilationContext, value: Value) -> str | None:
        return value.value

    def indexable(self, ctx: CompilationContext, value: Value) -> bool:
        return True

    def validate_index_types(self, ctx: CompilationContext, value: Value, indices: list[Type]) -> None | list[Type]:
        return None if len(indices) == 1 and NumberType().contains(indices[0]) else [NumberType()]

    def index(self, ctx: CompilationContext, value: Value, indices: list[Value]) -> Value:
        return Value(MemoryCellReferenceType(value, indices[0]), "", False)

    @staticmethod
    def _memcell_write_impl(ctx: CompilationContext, params: list[Value], value: Value) -> Value:
        val = params[1]
        if not val.memcell_serializable(ctx):
            ctx.error(f"Value of type '{val.type}' is not serializable")
        values = val.memcell_write(ctx)

        next_index = params[0]
        for v in values:
            ctx.emit(Instruction.write(v, value.value, next_index.value))
            next_index = next_index.binary_op(ctx, "+", Value.of_number(1))

        return Value.null()

    @staticmethod
    def _memcell_read_impl(ctx: CompilationContext, params: list[Value], value: Value) -> Value:
        val = Value.make_default(ctx, params[1].type.wrapped_type(ctx))
        if not val.memcell_serializable(ctx):
            ctx.error(f"Value of type '{val.type}' is not serializable")

        values = []
        next_index = params[0]
        for _ in range(val.memcell_size(ctx)):
            v = ctx.tmp()
            ctx.emit(Instruction.read(v, value.value, next_index.value))
            values.append(v)
            next_index = next_index.binary_op(ctx, "+", Value.of_number(1))

        val.memcell_read(ctx, values)

        return val

    def getattr(self, ctx: CompilationContext, value: Value, static: bool, name: str) -> Value | None:
        if not static:
            if name == "write":
                return Value(SpecialFunctionType(
                    "Block.write",
                    [NumberType(), AnyType()],
                    NullType(),
                    lambda ctx_, params, value_=value: self._memcell_write_impl(ctx_, params, value_)
                ), "")

            elif name == "read":
                return Value(SpecialFunctionType(
                    "Block.read",
                    [NumberType(), GenericTypeType()],
                    AnyType(),
                    lambda ctx_, params, value_=value: self._memcell_read_impl(ctx_, params, value_)
                ), "")

        return super(BlockType, self).getattr(ctx, value, static, name)

    def table_copyable(self, ctx: CompilationContext, value: Value) -> bool:
        return True


@dataclass(slots=True)
class MemoryCellReferenceType(Type):
    cell: Value
    index: Value

    def __str__(self):
        return "MemCellRef"

    def __eq__(self, other):
        return isinstance(other, MemoryCellReferenceType) and self.cell == other.cell and self.index == other.index

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return value.into_req(ctx, NumberType()).to_strings(ctx)

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        if NumberType().contains(other.type):
            ctx.emit(Instruction.write(other.value, self.cell.value, self.index.value))

    def assignable_type(self) -> Type:
        return UnionType([NumberType(), self])

    def into(self, ctx: CompilationContext, value: Value, type_: Type) -> Value | None:
        if type_.contains(NumberType()):
            val = Value(NumberType(), ctx.tmp())
            ctx.emit(Instruction.read(val.value, self.cell.value, self.index.value))
            return val

        return super(MemoryCellReferenceType, self).into(ctx, value, type_)


class UnitType(Type):
    def __str__(self):
        return "Unit"

    def __eq__(self, other):
        return isinstance(other, UnitType)

    def to_condition(self, ctx: CompilationContext, value: Value) -> str | None:
        return value.value

    def table_copyable(self, ctx: CompilationContext, value: Value) -> bool:
        return True


class ControllerType(Type):
    def __str__(self):
        return "Controller"

    def __eq__(self, other):
        return isinstance(other, ControllerType)

    def table_copyable(self, ctx: CompilationContext, value: Value) -> bool:
        return True


def _check_callable_with(ctx: CompilationContext, func: Value, params: list[Type]) -> Value:
    if not func.callable_with(ctx, params):
        ctx.error(f"Value of type '{func.type}' is not callable with parameters of types \
({', '.join(f'\'{t}\'' for t in params)})")
    return func


def _check_same_length(ctx: CompilationContext, a: list[Value], b: list[Value]) -> tuple[list[Value], list[Value]]:
    if len(a) != len(b):
        ctx.error(f"Tuple length mismatch: {len(a)} is not equal to {len(b)}")
    return a, b


def _unpackable_reduce(values: list[Value], ctx: CompilationContext, func: Value, start: Value) -> Value | None:
    for val in values:
        start = _check_callable_with(ctx, func, [start.type, val.type]).call(ctx, [start, val])
    return start


def _unpackable_getattr(ctx: CompilationContext, value: Value, name: str) -> Value | None:
    if name == "len" and (count := value.unpack_count(ctx)) is not None:
        return Value.of_number(count)

    elif name == "reversed":
        values = value.unpack_req(ctx)
        return Value(SpecialFunctionType(
            f"{value.type}.reversed",
            [],
            TupleType([v.type for v in values]),
            lambda ctx_, _: Value.of_tuple(ctx_, values[::-1])
        ), "")

    elif name == "enumerate":
        values = value.unpack_req(ctx)
        return Value(SpecialFunctionType(
            f"{value.type}.enumerate",
            [],
            TupleType([TupleType([NumberType(), v.type]) for v in values]),
            lambda ctx_, _: Value.of_tuple(ctx_, [
                Value.of_tuple(ctx_, [Value.of_number(i), v]) for i, v in enumerate(values)])
        ), "")

    elif name == "take":
        values = value.unpack_req(ctx)
        if len(values) == 0:
            ctx.error(f"Cannot take from an empty sequence")
        return Value(SpecialFunctionType(
            f"{value.type}.take",
            [],
            TupleType([values[0].type, TupleType([v.type for v in values[1:]])]),
            lambda ctx_, _: Value.of_tuple(ctx_, [values[0], Value.of_tuple(ctx_, values[1:])])
        ), "")

    elif name == "map":
        values = value.unpack_req(ctx)
        return Value(SpecialFunctionType(
            f"{value.type}.map",
            [AnyType()],
            TupleType([AnyType()] * len(values)),
            lambda ctx_, params: Value.of_tuple(ctx_, [
                _check_callable_with(ctx, params[0], [v.type]).call(ctx, [v]) for v in values])
        ), "")

    elif name == "unpack_map":
        values = value.unpack_req(ctx)
        return Value(SpecialFunctionType(
            f"{value.type}.unpack_map",
            [AnyType()],
            TupleType([AnyType()] * len(values)),
            lambda ctx_, params: Value.of_tuple(ctx_, [
                _check_callable_with(ctx, params[0], [v_.type for v_ in v.unpack_req(ctx)]).call(
                    ctx, v.unpack_req(ctx)) for v in values])
        ), "")

    elif name == "foreach":
        values = value.unpack_req(ctx)
        return Value(SpecialFunctionType(
            f"{value.type}.foreach",
            [AnyType()],
            NullType(),
            lambda ctx_, params: (
                [_check_callable_with(ctx, params[0], [v.type]).call(ctx, [v]) for v in values],
                Value.null())[1]
        ), "")

    elif name == "zip":
        values = value.unpack_req(ctx)
        return Value(SpecialFunctionType(
            f"{value.type}.zip",
            [AnyType()],
            TupleType([TupleType([v.type, AnyType()]) for v in values]),
            lambda ctx_, params: Value.of_tuple(ctx_, [Value.of_tuple(ctx_, [a, b]) for a, b in zip(
                *_check_same_length(ctx_, values, params[0].unpack_req(ctx_)))])
        ), "")

    elif name == "reduce":
        values = value.unpack_req(ctx)
        return Value(SpecialFunctionType(
            f"{value.type}.reduce",
            [AnyType(), AnyType()],
            AnyType(),
            lambda ctx_, params: _unpackable_reduce(values, ctx_, *params)
        ), "")

    elif name == "all":
        values = value.unpack_req(ctx)
        return Value(SpecialFunctionType(
            f"{value.type}.all",
            [],
            NumberType(),
            lambda ctx_, _: _unpackable_reduce(values, ctx_, Value(SpecialFunctionType(
                f"{value.type}.all.combine",
                [AnyType(), AnyType()],
                NumberType(),
                lambda ctx_2, params: params[0].binary_op_req(ctx_2, "&&", params[1])
            ), ""), Value.of_boolean(True))
        ), "")

    elif name == "any":
        values = value.unpack_req(ctx)
        return Value(SpecialFunctionType(
            f"{value.type}.any",
            [],
            NumberType(),
            lambda ctx_, _: _unpackable_reduce(values, ctx_, Value(SpecialFunctionType(
                f"{value.type}.any.combine",
                [AnyType(), AnyType()],
                NumberType(),
                lambda ctx_2, params: params[0].binary_op_req(ctx_2, "||", params[1])
            ), ""), Value.of_boolean(False))
        ), "")

    elif name == "sum":
        values = value.unpack_req(ctx)
        return Value(SpecialFunctionType(
            f"{value.type}.sum",
            [],
            NumberType(),
            lambda ctx_, _: _unpackable_reduce(values, ctx_, Value(SpecialFunctionType(
                f"{value.type}.sum.combine",
                [AnyType(), AnyType()],
                NumberType(),
                lambda ctx_2, params: params[0].binary_op_req(ctx_2, "+", params[1])
            ), ""), Value.of_number(0))
        ), "")

    elif name == "prod":
        values = value.unpack_req(ctx)
        return Value(SpecialFunctionType(
            f"{value.type}.prod",
            [],
            NumberType(),
            lambda ctx_, _: _unpackable_reduce(values, ctx_, Value(SpecialFunctionType(
                f"{value.type}.prod.combine",
                [AnyType(), AnyType()],
                NumberType(),
                lambda ctx_2, params: params[0].binary_op_req(ctx_2, "*", params[1])
            ), ""), Value.of_number(1))
        ), "")


@dataclass(slots=True)
class TupleType(Type):
    class ValueIterator(ValueIterator):
        has_func: Value
        next_func: Value

        def __init__(self, has_func: Value, next_func: Value):
            self.has_func = has_func
            self.next_func = next_func

        def has_value(self, ctx: CompilationContext) -> Value:
            return self.has_func.call(ctx, [])

        def next_value(self, ctx: CompilationContext) -> Value:
            return self.next_func.call(ctx, [])

        def end_loop(self, ctx: CompilationContext):
            pass

    types: list[Type]

    def __str__(self):
        return f"({', '.join(map(str, self.types))})"

    def __eq__(self, other):
        return isinstance(other, TupleType) and self.types == other.types

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        for dst, src in zip(value.unpack(ctx), other.unpack(ctx)):
            dst.assign(ctx, src)

    def assign_default(self, ctx: CompilationContext, value: Value):
        for dst in value.unpack(ctx):
            dst.assign_default(ctx)

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        strings = ["\"(\""]
        for value in value.unpack(ctx):
            if len(strings) > 1:
                strings.append("\", \"")
            strings += value.to_strings(ctx)
        strings.append("\")\"")
        return strings

    def unpack_count(self, ctx: CompilationContext, value: Value) -> int:
        return len(self.types)

    def unpack(self, ctx: CompilationContext, value: Value) -> list[Value]:
        return [Value(t, ABI.attribute(value.value, i), value.const) for i, t in enumerate(self.types)]

    def unary_op(self, ctx: CompilationContext, value: Value, op: str) -> Value | None:
        return Value.of_tuple(ctx, [val.unary_op_req(ctx, op) for val in value.unpack_req(ctx)])

    def binary_op(self, ctx: CompilationContext, value: Value, op: str, other: Value) -> Value | None:
        if op == "++":
            return Value.of_tuple(ctx, value.unpack_req(ctx) + other.unpack_req(ctx))

        if other.unpackable(ctx):
            values = value.unpack_req(ctx)
            other_values = other.unpack_req(ctx)
            if len(values) != len(other_values):
                ctx.error(f"Tuple length mismatch: {len(values)} is not equal to {len(other_values)}")
            return Value.of_tuple(ctx, [a.binary_op_req(ctx, op, b) for a, b in zip(values, other_values)])

        else:
            return Value.of_tuple(ctx, [val.binary_op_req(ctx, op, other) for val in value.unpack_req(ctx)])

    def binary_op_r(self, ctx: CompilationContext, other: Value, op: str, value: Value) -> Value | None:
        if other.unpackable(ctx):
            values = value.unpack_req(ctx)
            other_values = other.unpack_req(ctx)
            if len(values) != len(other_values):
                ctx.error(f"Tuple length mismatch: {len(values)} is not equal to {len(other_values)}")
            return Value.of_tuple(ctx, [b.binary_op_req(ctx, op, a) for a, b in zip(values, other_values)])

        else:
            return Value.of_tuple(ctx, [other.binary_op_req(ctx, op, val) for val in value.unpack_req(ctx)])

    def iterate(self, ctx: CompilationContext, value: Value) -> ValueIterator | None:
        if len(self.types) != 2:
            return None

        next_func, has_func = value.unpack_req(ctx)

        if not next_func.callable_with(ctx, []):
            return None
        if not has_func.callable_with(ctx, []):
            return None

        return TupleType.ValueIterator(has_func, next_func)

    def getattr(self, ctx: CompilationContext, value: Value, static: bool, name: str) -> Value | None:
        if not static:
            try:
                idx = int(name)
            except ValueError:
                pass
            else:
                if 0 <= idx < len(self.types):
                    return value.unpack_req(ctx)[idx]

            if (val := _unpackable_getattr(ctx, value, name)) is not None:
                return val

        return super(TupleType, self).getattr(ctx, value, static, name)

    def memcell_serializable(self, ctx: CompilationContext, value: Value) -> bool:
        return all(v.memcell_serializable(ctx) for v in value.unpack_req(ctx))

    def memcell_size(self, ctx: CompilationContext, value: Value) -> int:
        return sum(v.memcell_size(ctx) for v in value.unpack_req(ctx))

    def memcell_write(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [s for v in value.unpack_req(ctx) for s in v.memcell_write(ctx)]

    def memcell_read(self, ctx: CompilationContext, value: Value, values: list[str]):
        i = 0
        for v in value.unpack_req(ctx):
            size = v.memcell_size(ctx)
            v.memcell_read(ctx, values[i:i+size])
            i += size

    def table_copyable(self, ctx: CompilationContext, value: Value) -> bool:
        return all(v.table_copyable(ctx) for v in value.unpack_req(ctx))

    def table_size(self, ctx: CompilationContext, value: Value) -> int:
        return sum(v.table_size(ctx) for v in value.unpack_req(ctx))

    def table_variables(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [s for v in value.unpack_req(ctx) for s in v.table_variables(ctx)]

    def indexable(self, ctx: CompilationContext, value: Value) -> bool:
        return True

    def validate_index_types(self, ctx: CompilationContext, value: Value, indices: list[Type]) -> None | list[Type]:
        return None if len(indices) == 1 and NumberType().contains(indices[0]) else [NumberType()]

    def index(self, ctx: CompilationContext, value: Value, indices: list[Value]) -> Value:
        if len(self.types) == 0:
            ctx.error("Cannot index empty tuple")

        if any(t != self.types[0] for t in self.types[1:]):
            ctx.error(f"Cannot index tuples with different types")

        values = value.unpack_req(ctx)
        if not values[0].table_copyable(ctx):
            ctx.error(f"Indexing is not available for tuples of '{values[0].type}'")

        return Value(TupleIndexReferenceType(values[0].type, indices[0], values), "null", False)


class TupleIndexReferenceType(Type):
    type: Type
    index: Value
    values: list[Value]

    def __init__(self, type_: Type, index: Value, values: list[Value]):
        self.type = type_
        self.index = index
        self.values = values

    def __str__(self):
        return "TupleIndexRef"

    def __eq__(self, other):
        return (isinstance(other, TupleIndexReferenceType) and self.type == other.type and self.index == other.index
                and self.values == other.values)

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return value.into_req(ctx, self.type).to_strings(ctx)

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        if self != other.type:
            with value.do_assignment(ctx):
                ctx.emit(Instruction.TableWrite(
                    [val.table_variables(ctx) for val in self.values],
                    other.table_variables(ctx),
                    self.index.value
                ))

    def assignable_type(self) -> Type:
        return UnionType([self.type, self])

    def into(self, ctx: CompilationContext, value: Value, type_: Type) -> Value | None:
        if type_.contains(self.type):
            val = Value(self.type, ctx.tmp(), False)
            ctx.emit(Instruction.TableRead(
                val.table_variables(ctx),
                [val.table_variables(ctx) for val in self.values],
                self.index.value
            ))
            return val

        return super(TupleIndexReferenceType, self).into(ctx, value, type_)


@dataclass(slots=True)
class UnionType(Type):
    types: list[Type]

    def __str__(self):
        return " | ".join(map(str, self.types))

    def __eq__(self, other):
        return isinstance(other, UnionType) and self.types == other.types

    def contains(self, other: Type) -> bool:
        return any(t.contains(other) for t in self.types)

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        raise NotImplementedError

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        raise NotImplementedError


class RangeType(Type):
    class ValueIterator(ValueIterator):
        index: Value
        end: Value

        def __init__(self, ctx: CompilationContext, start: Value, end: Value):
            self.index = start.copy(ctx)
            self.end = end

        def has_value(self, ctx: CompilationContext) -> Value:
            return self.index.binary_op(ctx, "<", self.end)

        def next_value(self, ctx: CompilationContext) -> Value:
            return self.index

        def end_loop(self, ctx: CompilationContext):
            self.index.binary_op(ctx, "+=", Value.of_number(1))

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

    def assign_default(self, ctx: CompilationContext, value: Value):
        self._attr(value, "start", value.const).assign_default(ctx)
        self._attr(value, "end", value.const).assign_default(ctx)

    def into(self, ctx: CompilationContext, value: Value, type_: Type) -> Value | None:
        if type_.contains(RangeWithStepType()):
            return Value.of_range_with_step(ctx, self._attr(value, "start", True),
                                            self._attr(value, "end", True), Value.of_number(1))

        return super(RangeType, self).into(ctx, value, type_)

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [ABI.attribute(value.value, "start"), "\"..\"", ABI.attribute(value.value, "end")]

    def getattr(self, ctx: CompilationContext, value: Value, static: bool, name: str) -> Value | None:
        if not static:
            if name in ("start", "end"):
                return self._attr(value, name, value.const)

        return super(RangeType, self).getattr(ctx, value, static, name)

    def iterate(self, ctx: CompilationContext, value: Value) -> ValueIterator | None:
        return RangeType.ValueIterator(
            ctx,
            self._attr(value, "start", True),
            self._attr(value, "end", True)
        )

    def memcell_serializable(self, ctx: CompilationContext, value: Value) -> bool:
        return True

    def memcell_size(self, ctx: CompilationContext, value: Value) -> int:
        return 2

    def memcell_write(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [
            self._attr(value, "start", True).value,
            self._attr(value, "end", True).value
        ]

    def memcell_read(self, ctx: CompilationContext, value: Value, values: list[str]):
        self._attr(value, "start", False).assign(ctx, Value.of_number(values[0]))
        self._attr(value, "end", False).assign(ctx, Value.of_number(values[1]))


class RangeWithStepType(Type):
    class ValueIterator(ValueIterator):
        index: Value
        end: Value
        step: Value

        def __init__(self, ctx: CompilationContext, start: Value, end: Value, step: Value):
            self.index = start.copy(ctx)
            self.end = end
            self.step = step

        def has_value(self, ctx: CompilationContext) -> Value:
            return self.index.binary_op(ctx, "<", self.end)

        def next_value(self, ctx: CompilationContext) -> Value:
            return self.index

        def end_loop(self, ctx: CompilationContext):
            self.index.binary_op(ctx, "+=", self.step)

    def __str__(self):
        return f"RangeWithStep"

    def __eq__(self, other):
        return isinstance(other, RangeWithStepType)

    @staticmethod
    def _attr(value: Value, name: str, const: bool) -> Value:
        return Value(NumberType(), ABI.attribute(value.value, name), const)

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        self._attr(value, "start", value.const).assign(ctx, other.getattr_req(ctx, False, "start"))
        self._attr(value, "end", value.const).assign(ctx, other.getattr_req(ctx, False, "end"))
        self._attr(value, "step", value.const).assign(ctx, other.getattr_req(ctx, False, "step"))

    def assign_default(self, ctx: CompilationContext, value: Value):
        self._attr(value, "start", value.const).assign_default(ctx)
        self._attr(value, "end", value.const).assign_default(ctx)
        self._attr(value, "step", value.const).assign_default(ctx)

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [ABI.attribute(value.value, "start"), "\"..\"", ABI.attribute(value.value, "end"), "\"..\"",
                ABI.attribute(value.value, "step")]

    def getattr(self, ctx: CompilationContext, value: Value, static: bool, name: str) -> Value | None:
        if not static:
            if name in ("start", "end", "step"):
                return self._attr(value, name, value.const)

        return super(RangeWithStepType, self).getattr(ctx, value, static, name)

    def iterate(self, ctx: CompilationContext, value: Value) -> ValueIterator | None:
        return RangeWithStepType.ValueIterator(
            ctx,
            self._attr(value, "start", True),
            self._attr(value, "end", True),
            self._attr(value, "step", True)
        )

    def memcell_serializable(self, ctx: CompilationContext, value: Value) -> bool:
        return True

    def memcell_size(self, ctx: CompilationContext, value: Value) -> int:
        return 3

    def memcell_write(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [
            self._attr(value, "start", True).value,
            self._attr(value, "end", True).value,
            self._attr(value, "step", True).value
        ]

    def memcell_read(self, ctx: CompilationContext, value: Value, values: list[str]):
        self._attr(value, "start", False).assign(ctx, Value.of_number(values[0]))
        self._attr(value, "end", False).assign(ctx, Value.of_number(values[1]))
        self._attr(value, "step", False).assign(ctx, Value.of_number(values[2]))


def _format_function_signature(params: list[Type | None], result: Type | None) -> str:
    return f"({', '.join(str(p) if p is not None else '?' for p in params)}) -> {result if result is not None else '?'}"


def _check_params(expected: list[Type | None], provided: list[Type]) -> bool:
    return len(expected) == len(provided) and all(e is None or e.contains(t) for e, t in zip(expected, provided))


@dataclass(slots=True)
class SpecialFunctionType(Type):
    name: str
    params: list[Type | None]
    result: Type | None
    function: Callable[[CompilationContext, list[Value]], Value]

    def __str__(self):
        return f"fn {self.name}{_format_function_signature(self.params, self.result)}"

    def __eq__(self, other):
        return isinstance(other, SpecialFunctionType) and self.name == other.name and self.params == other.params and \
            self.result == other.result and self.function is other.function

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        pass

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [_stringify(str(self))]

    def callable(self, ctx: CompilationContext, value: Value) -> bool:
        return True

    def call_with_suggestion(self, ctx: CompilationContext, value: Value) -> list[Type | None] | None:
        return self.params

    def callable_with(self, ctx: CompilationContext, value: Value, param_types: list[Type]) -> bool:
        return _check_params(self.params, param_types)

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        return self.function(ctx, params)


@dataclass(slots=True)
class IntrinsicFunctionType(Type):
    name: str
    params: list[Type]
    outputs: list[int]
    function: Callable[[CompilationContext, list[str]], None]
    subcommand: str | None = None

    def __post_init__(self):
        self.input_params: list[Type] = [p for i, p in enumerate(self.params) if i not in self.outputs]

        self.all_params: list[tuple[Type, bool]] = [(p, i in self.outputs) for i, p in enumerate(self.params)]

        if len(self.outputs) == 0:
            self.result: Type = NullType()
        elif len(self.outputs) == 1:
            self.result: Type = self.params[self.outputs[0]]
        else:
            self.result: Type = TupleType([p for i, p in enumerate(self.params) if i in self.outputs])

    def __str__(self):
        return f"Intrinsic[{self.name}]"

    def __eq__(self, other):
        return isinstance(other, IntrinsicFunctionType) and self.name == other.name

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        pass

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [_stringify(str(self))]

    def callable(self, ctx: CompilationContext, value: Value) -> bool:
        return True

    def call_with_suggestion(self, ctx: CompilationContext, value: Value) -> list[Type | None] | None:
        return self.input_params

    def callable_with(self, ctx: CompilationContext, value: Value, param_types: list[Type]) -> bool:
        return len(param_types) == len(self.input_params) and all(
            a.contains(b) for a, b in zip(self.input_params, param_types))

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        params_all = []
        output_vars = []
        input_i = 0
        for type_, is_output in self.all_params:
            if is_output:
                val = Value(type_, ctx.tmp())
                params_all.append(val.value)
                output_vars.append(val)
            else:
                params_all.append(params[input_i].value)
                input_i += 1
        assert input_i == len(params)

        self.function(ctx, params_all)

        if len(output_vars) == 0:
            return Value.null()
        elif len(output_vars) == 1:
            return output_vars[0]
        else:
            return Value.of_tuple(ctx, output_vars)


@dataclass
class IntrinsicSubcommandFunctionType(Type):
    name: str
    subcommands: dict[str, Value]

    def __str__(self):
        return f"Intrinsic[{self.name}]"

    def __eq__(self, other):
        return isinstance(other, IntrinsicSubcommandFunctionType) and self.name == other.name

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        pass

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [_stringify(str(self))]

    def getattr(self, ctx: CompilationContext, value: Value, static: bool, name: str) -> Value | None:
        if not static:
            if (val := self.subcommands.get(name)) is not None:
                return val

        return super(IntrinsicSubcommandFunctionType, self).getattr(ctx, value, static, name)


@dataclass(slots=True)
class FunctionType(Type):
    @dataclass(slots=True)
    class Param:
        name: str
        reference: bool
        type: Type | None
        variadic: bool

        def __str__(self):
            return f"{'&' if self.reference else ''}{self.name}{'...' if self.variadic else ''}\
{': ' + str(self.type) if self.type else ''}"

    @staticmethod
    def check_callable_with(params: list[FunctionType.Param], param_types: list[Type]) -> bool:
        if len(params) == 0:
            return len(param_types) == 0

        elif params[-1].variadic:
            return len(params) <= len(param_types) and \
                all(p.type is None or p.type.contains(t) for p, t in zip(
                    params[:-1], param_types[:len(params)-1])) and \
                all(params[-1].type is None or params[-1].type.contains(t) for t in param_types[len(params)-1:])

        else:
            return len(params) == len(param_types) and \
                all(p.type is None or p.type.contains(t) for p, t in zip(params, param_types))

    @staticmethod
    def declare_param(ctx: CompilationContext, i: int, dst: FunctionType.Param, srcs: list[Value]):
        if dst.variadic:
            if dst.reference:
                ctx.error(f"Variadic parameters cannot be references")
            else:
                ctx.scope.declare_special(dst.name, Value.of_tuple(ctx, srcs[i:]))
        else:
            if dst.reference:
                ctx.scope.declare_special(dst.name, srcs[i])
            else:
                ctx.scope.declare(
                    dst.name, dst.type if dst.type is not None else srcs[i].type, False).assign(ctx, srcs[i])

    @classmethod
    def declare_params(cls, ctx: CompilationContext, dsts: list[FunctionType.Param], srcs: list[Value]):
        for i, dst in enumerate(dsts):
            cls.declare_param(ctx, i, dst, srcs)

    name: str
    params: list[FunctionType.Param]
    result: Type | None
    code: Node
    global_closure: list[dict[str, Value]]
    attributes: set[str]

    def __str__(self):
        return f"fn({', '.join(map(str, self.params))}){' -> ' + str(self.result) if self.result else ''}"

    def __eq__(self, other):
        return isinstance(other, FunctionType) and self.params == other.params and self.result == other.result and \
            self.code == other.code

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        pass

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [_stringify(str(self))]

    def callable(self, ctx: CompilationContext, value: Value) -> bool:
        return True

    def call_with_suggestion(self, ctx: CompilationContext, value: Value) -> list[Type | None] | None:
        return [p.type for p in self.params]

    def callable_with(self, ctx: CompilationContext, value: Value, param_types: list[Type]) -> bool:
        return FunctionType.check_callable_with(self.params, param_types)

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        with ctx.scope.function_call(ctx, f"$function_{self.name}:{ctx.tmp_num()}", self.result,
                                     ctx.scope.combine_global_closures(self.global_closure)):
            FunctionType.declare_params(ctx, self.params, params)

            result = ctx.generate_node(self.code)

            return_val = Value(self.result if self.result is not None else result.type,
                               ABI.return_value(ctx.scope.get_function()), False)
            if NullType().contains(result.type):
                if self.result is not None and not self.result.contains(NullType()):
                    ctx.error("Function code block doesn't return value", self.code.pos)
                else:
                    return_val.assign_default(ctx)
            else:
                return_val.assign(ctx, result)

            ctx.emit(Instruction.label(ABI.function_end(ctx.scope.get_function())))

            if "nodiscard" in self.attributes:
                return_val = return_val.with_no_discard()

            return return_val


@dataclass(slots=True)
class LambdaType(Type):
    @dataclass(slots=True)
    class Capture:
        name: str
        value: Value
        display_str: str

        def __str__(self):
            return self.display_str

    name: str
    params: list[FunctionType.Param]
    captures: list[LambdaType.Capture]
    result: Type | None
    code: Node
    global_closure: list[dict[str, Value]]

    def __str__(self):
        return f"|{', '.join(map(str, self.params))}|[{', '.join(map(str, self.captures))}]\
{' -> ' + str(self.result) if self.result else ''}"

    def __eq__(self, other):
        return isinstance(other, LambdaType) and self.params == other.params and self.captures == other.captures and \
            self.result == other.result and self.code == other.code

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        pass

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [_stringify(str(self))]

    def callable(self, ctx: CompilationContext, value: Value) -> bool:
        return True

    def call_with_suggestion(self, ctx: CompilationContext, value: Value) -> list[Type | None] | None:
        return [p.type for p in self.params]

    def callable_with(self, ctx: CompilationContext, value: Value, param_types: list[Type]) -> bool:
        return FunctionType.check_callable_with(self.params, param_types)

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        with ctx.scope.function_call(ctx, f"$lambda_{self.name}:{ctx.tmp_num()}", self.result,
                                     ctx.scope.combine_global_closures(self.global_closure)):
            for capture in self.captures:
                ctx.scope.declare_special(capture.name, capture.value)

            FunctionType.declare_params(ctx, self.params, params)

            result = ctx.generate_node(self.code)

            return_val = Value(self.result if self.result is not None else result.type,
                               ABI.return_value(ctx.scope.get_function()), False)
            if NullType().contains(result.type):
                if self.result is not None and not self.result.contains(NullType()):
                    ctx.error("Function code block doesn't return value", self.code.pos)
                else:
                    return_val.assign_default(ctx)
            else:
                return_val.assign(ctx, result)

            ctx.emit(Instruction.label(ABI.function_end(ctx.scope.get_function())))

            return return_val


@dataclass(slots=True)
class StructMethodData:
    name: str
    params: list[FunctionType.Param]
    result: Type | None
    code: Node
    global_closure: list[dict[str, Value]]
    attributes: set[str]


@dataclass(slots=True)
class StructMethodType(Type):
    self_value: Value
    name: str
    params: list[FunctionType.Param]
    result: Type | None
    code: Node
    global_closure: list[dict[str, Value]]
    attributes: set[str]

    @classmethod
    def create_value(cls, self_value: Value, data: StructMethodData) -> Value:
        return Value(cls(self_value, data.name, data.params, data.result, data.code,
                         data.global_closure, data.attributes), "")

    def __str__(self):
        return f"fn({', '.join(map(str, self.params))}){' -> ' + str(self.result) if self.result else ''}"

    def __eq__(self, other):
        return isinstance(other, StructMethodType) and self.params == other.params and self.result == other.result and \
            self.code == other.code and self.self_value == other.self_value

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        pass

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [_stringify(str(self))]

    def callable(self, ctx: CompilationContext, value: Value) -> bool:
        return True

    def call_with_suggestion(self, ctx: CompilationContext, value: Value) -> list[Type | None] | None:
        return [p.type for p in self.params]

    def callable_with(self, ctx: CompilationContext, value: Value, param_types: list[Type]) -> bool:
        return FunctionType.check_callable_with(self.params, param_types)

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        with ctx.scope.function_call(ctx, f"$method_{self.name}:{ctx.tmp_num()}", self.result,
                                     ctx.scope.combine_global_closures(self.global_closure)):
            ctx.scope.declare_special("self", self.self_value)

            FunctionType.declare_params(ctx, self.params, params)

            result = ctx.generate_node(self.code)

            return_val = Value(self.result if self.result is not None else result.type,
                               ABI.return_value(ctx.scope.get_function()), False)
            if NullType().contains(result.type):
                if self.result is not None and not self.result.contains(NullType()):
                    ctx.error("Function code block doesn't return value", self.code.pos)
                else:
                    return_val.assign_default(ctx)
            else:
                return_val.assign(ctx, result)

            ctx.emit(Instruction.label(ABI.function_end(ctx.scope.get_function())))

            if "nodiscard" in self.attributes:
                return_val = return_val.with_no_discard()

            return return_val


@dataclass(slots=True)
class StructBaseType(Type):
    UNARY_OP_NAMES = {
        "!": "@not",
        "~": "@flip",
        "-": "@neg"
    }

    BINARY_OP_NAMES = {
        "+": "@add",
        "-": "@sub",
        "*": "@mul",
        "/": "@div",
        "//": "@idiv",
        "/.": "@idiv",
        "%": "@mod",
        "**": "@pow",
        "&&": "@land",
        "||": "@lor",
        "<": "@lt",
        "<=": "@lte",
        ">": "@gt",
        ">=": "@gte",
        "<<": "@shl",
        ">>": "@shr",
        "|": "@or",
        "&": "@and",
        "^": "@xor"
    }

    name: str | None
    fields: list[tuple[str, Type]]
    static_fields: dict[str, Value]
    methods: dict[str, tuple[bool, StructMethodData]]
    static_methods: dict[str, StructMethodData]
    _instance_type: StructInstanceType = None

    def __post_init__(self):
        self.reload_instance_type()

    def reload_instance_type(self):
        self._instance_type = StructInstanceType(self)

    def __str__(self):
        return f"Struct[{self.name if self.name else '?'}]"

    def __eq__(self, other):
        return isinstance(other, StructBaseType) and self.name == other.name and self.fields == other.fields and \
            self.static_fields == other.static_fields and self.methods == other.methods and \
            self.static_methods == other.static_methods

    def wrapped_type(self, ctx: CompilationContext) -> Type:
        return self._instance_type

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        pass

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [_stringify(str(self))]

    def get_static(self, value: Value, name: str) -> Value | None:
        if (val := self.static_fields.get(name)) is not None:
            return val

        elif (val := self.static_methods.get(name)) is not None:
            return StructMethodType.create_value(value, val)

    def getattr(self, ctx: CompilationContext, value: Value, static: bool, name: str) -> Value | None:
        if static:
            if (val := self.get_static(value, name)) is not None:
                return val

        return super(StructBaseType, self).getattr(ctx, value, static, name)

    def into(self, ctx: CompilationContext, value: Value, type_: Type) -> Value | None:
        if (func := value.getattr(ctx, True, "@cast")) is not None:
            if not func.callable_with(ctx, [TypeType(type_)]):
                ctx.error(
                    f"Value of type '{func.type}' does not have the correct function signature to implement type casts")
            return func.call(ctx, [Value.of_type(type_)])

        return super(StructBaseType, self).into(ctx, value, type_)

    def construct_from(self, ctx: CompilationContext, value: Value) -> Value | None:
        if (func := Value(self, "").getattr(ctx, True, "@from")) is not None:
            if not func.callable_with(ctx, [value.type]):
                ctx.error(
                    f"Value of type '{func.type}' does not have the correct function signature to implement type casts")
            return func.call(ctx, [value])

        return super(StructBaseType, self).construct_from(ctx, value)

    def callable(self, ctx: CompilationContext, value: Value) -> bool:
        return True

    @staticmethod
    def _get_call_func(ctx: CompilationContext, value: Value) -> Value:
        return value.getattr(ctx, True, "@call")

    def call_with_suggestion(self, ctx: CompilationContext, value: Value) -> list[Type | None] | None:
        if (func := self._get_call_func(ctx, value)) is not None:
            return func.call_with_suggestion(ctx)
        return [t for _, t in self.fields]

    def callable_with(self, ctx: CompilationContext, value: Value, param_types: list[Type]) -> bool:
        if (func := self._get_call_func(ctx, value)) is not None:
            return func.callable_with(ctx, param_types)
        return len(self.fields) == len(param_types) and all(f[1].contains(t) for f, t in zip(self.fields, param_types))

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        if (func := self._get_call_func(ctx, value)) is not None:
            return func.call(ctx, params)

        value = Value(self._instance_type, ctx.tmp(), False)

        for (field, _), param in zip(self.fields, params):
            value.getattr_req(ctx, False, field).assign(ctx, param)

        return value

    @staticmethod
    def _get_index_func(ctx: CompilationContext, value: Value) -> Value:
        return value.getattr_req(ctx, True, "@index")

    def indexable(self, ctx: CompilationContext, value: Value) -> bool:
        return value.getattr(ctx, True, "@index") is not None

    def validate_index_types(self, ctx: CompilationContext, value: Value, indices: list[Type]) -> None | list[Type]:
        func = self._get_index_func(ctx, value)
        if func.callable_with(ctx, indices):
            return None
        else:
            if (suggested := func.call_with_suggestion(ctx)) is not None:
                return suggested
            else:
                return [EllipsisType()]

    def index(self, ctx: CompilationContext, value: Value, indices: list[Value]) -> Value:
        return self._get_index_func(ctx, value).call(ctx, indices)

    def unary_op(self, ctx: CompilationContext, value: Value, op: str) -> Value | None:
        if (name := self.UNARY_OP_NAMES.get(op)) is not None:
            if (func := value.getattr(ctx, True, name)) is not None:
                if not func.callable_with(ctx, []):
                    ctx.error(f"Value of type '{func.type}' does not have the correct function signature to implement \
unary operator '{op}'")
                return func.call(ctx, [])

        return super(StructBaseType, self).unary_op(ctx, value, op)

    def binary_op(self, ctx: CompilationContext, value: Value, op: str, other: Value) -> Value | None:
        if (name := self.BINARY_OP_NAMES.get(op)) is not None:
            if (func := value.getattr(ctx, True, name)) is not None:
                if not func.callable_with(ctx, [other.type]):
                    ctx.error(f"Value of type '{func.type}' does not have the correct function signature to implement \
binary operator '{op}' with other value of type '{other.type}'")
                return func.call(ctx, [other])

        return super(StructBaseType, self).binary_op(ctx, value, op, other)

    def binary_op_r(self, ctx: CompilationContext, other: Value, op: str, value: Value) -> Value | None:
        if (name := self.BINARY_OP_NAMES.get(op)) is not None:
            name = f"@r_{name[1:]}"
            if (func := value.getattr(ctx, True, name)) is not None:
                if not func.callable_with(ctx, [other.type]):
                    ctx.error(f"Value of type '{func.type}' does not have the correct function signature to implement \
reversed binary operator '{op}' with other value of type '{other.type}'")
                return func.call(ctx, [other])

        return super(StructBaseType, self).binary_op_r(ctx, other, op, value)


@dataclass(slots=True)
class StructInstanceType(Type):
    base: StructBaseType

    UNARY_OP_NAMES = {
        "!": "@not",
        "~": "@flip",
        "-": "@neg"
    }

    BINARY_OP_NAMES = {
        "+": "@add",
        "-": "@sub",
        "*": "@mul",
        "/": "@div",
        "//": "@idiv",
        "/.": "@idiv",
        "%": "@mod",
        "**": "@pow",
        "&&": "@land",
        "||": "@lor",
        "<": "@lt",
        "<=": "@lte",
        ">": "@gt",
        ">=": "@gte",
        "<<": "@shl",
        ">>": "@shr",
        "|": "@or",
        "&": "@and",
        "^": "@xor"
    }

    def __post_init__(self):
        self.fields: dict[str, Type] = {field: type_ for field, type_ in self.base.fields}

    def __str__(self):
        return f"{self.base.name if self.base.name else '?'}"

    def __eq__(self, other):
        return isinstance(other, StructInstanceType) and self.base == other.base

    def _fields(self, ctx: CompilationContext, value: Value):
        for field, _ in self.base.fields:
            yield value.getattr_req(ctx, False, field)

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        for field, _ in self.base.fields:
            value.getattr_req(ctx, False, field).assign(ctx, other.getattr_req(ctx, False, field))

    def assign_default(self, ctx: CompilationContext, value: Value):
        for field in self._fields(ctx, value):
            field.assign_default(ctx)

    def getattr(self, ctx: CompilationContext, value: Value, static: bool, name: str) -> Value | None:
        if static:
            if (val := self.base.get_static(value, name)) is not None:
                return val

        else:
            if (type_ := self.fields.get(name)) is not None:
                return Value(type_, ABI.attribute(value.value, name), value.const)

            elif (val := self.base.methods.get(name)) is not None:
                if value.const and not val[0]:
                    ctx.error(f"Cannot use non-const method '{name}' on const value of type '{self}'")
                return StructMethodType.create_value(value, val[1])

        return super(StructInstanceType, self).getattr(ctx, value, static, name)

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        strings = ["\"{\""]
        for field, _ in self.base.fields:
            if len(strings) > 1:
                strings.append(f"\", {field}: \"")
            else:
                strings.append(f"\"{field}: \"")
            strings += value.getattr_req(ctx, False, field).to_strings(ctx)
        strings.append("\"}\"")
        return strings

    def iterate(self, ctx: CompilationContext, value: Value) -> ValueIterator | None:
        iter_func = value.getattr_req(ctx, False, "@iter")
        if not iter_func.callable_with(ctx, []):
            ctx.error(f"Value of type '{iter_func.type}' does not have the correct function signature")
        return iter_func.call(ctx, []).iterate(ctx)

    @staticmethod
    def _get_call_func(ctx: CompilationContext, value: Value) -> Value:
        return value.getattr_req(ctx, False, "@call")

    def callable(self, ctx: CompilationContext, value: Value) -> bool:
        return value.getattr(ctx, False, "@call") is not None

    def call_with_suggestion(self, ctx: CompilationContext, value: Value) -> list[Type | None] | None:
        return self._get_call_func(ctx, value).call_with_suggestion(ctx)

    def callable_with(self, ctx: CompilationContext, value: Value, param_types: list[Type]) -> bool:
        return self._get_call_func(ctx, value).callable_with(ctx, param_types)

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        return self._get_call_func(ctx, value).call(ctx, params)

    @staticmethod
    def _get_index_func(ctx: CompilationContext, value: Value) -> Value:
        return value.getattr_req(ctx, False, "@index")

    def indexable(self, ctx: CompilationContext, value: Value) -> bool:
        return value.getattr(ctx, False, "@index") is not None

    def validate_index_types(self, ctx: CompilationContext, value: Value, indices: list[Type]) -> None | list[Type]:
        func = self._get_index_func(ctx, value)
        if func.callable_with(ctx, indices):
            return None
        else:
            if (suggested := func.call_with_suggestion(ctx)) is not None:
                return suggested
            else:
                return [EllipsisType()]

    def index(self, ctx: CompilationContext, value: Value, indices: list[Value]) -> Value:
        return self._get_index_func(ctx, value).call(ctx, indices)

    def unary_op(self, ctx: CompilationContext, value: Value, op: str) -> Value | None:
        if (name := self.UNARY_OP_NAMES.get(op)) is not None:
            if (func := value.getattr(ctx, False, name)) is not None:
                if not func.callable_with(ctx, []):
                    ctx.error(f"Value of type '{func.type}' does not have the correct function signature to implement \
unary operator '{op}'")
                return func.call(ctx, [])

        if op == "...":
            return Value.of_tuple(ctx, [value.getattr_req(ctx, False, field) for field, _ in self.base.fields])

        return super(StructInstanceType, self).unary_op(ctx, value, op)

    def binary_op(self, ctx: CompilationContext, value: Value, op: str, other: Value) -> Value | None:
        if (name := self.BINARY_OP_NAMES.get(op)) is not None:
            if (func := value.getattr(ctx, False, name)) is not None:
                if not func.callable_with(ctx, [other.type]):
                    ctx.error(f"Value of type '{func.type}' does not have the correct function signature to implement \
binary operator '{op}' with other value of type '{other.type}'")
                return func.call(ctx, [other])

        return super(StructInstanceType, self).binary_op(ctx, value, op, other)

    def binary_op_r(self, ctx: CompilationContext, other: Value, op: str, value: Value) -> Value | None:
        if (name := self.BINARY_OP_NAMES.get(op)) is not None:
            name = f"@r_{name[1:]}"
            if (func := value.getattr(ctx, False, name)) is not None:
                if not func.callable_with(ctx, [other.type]):
                    ctx.error(f"Value of type '{func.type}' does not have the correct function signature to implement \
reversed binary operator '{op}' with other value of type '{other.type}'")
                return func.call(ctx, [other])

        return super(StructInstanceType, self).binary_op_r(ctx, other, op, value)

    def into(self, ctx: CompilationContext, value: Value, type_: Type) -> Value | None:
        if (func := value.getattr(ctx, False, "@cast")) is not None:
            if not func.callable_with(ctx, [TypeType(type_)]):
                ctx.error(
                    f"Value of type '{func.type}' does not have the correct function signature to implement type casts")
            return func.call(ctx, [Value.of_type(type_)])

        return super(StructInstanceType, self).into(ctx, value, type_)

    def construct_from(self, ctx: CompilationContext, value: Value) -> Value | None:
        if (func := Value(self, "").getattr(ctx, False, "@from")) is not None:
            if not func.callable_with(ctx, [value.type]):
                ctx.error(
                    f"Value of type '{func.type}' does not have the correct function signature to implement type casts")
            return func.call(ctx, [value])

        return super(StructInstanceType, self).construct_from(ctx, value)

    def memcell_serializable(self, ctx: CompilationContext, value: Value) -> bool:
        return all(v.memcell_serializable(ctx) for v in self._fields(ctx, value))

    def memcell_size(self, ctx: CompilationContext, value: Value) -> int:
        return sum(v.memcell_size(ctx) for v in self._fields(ctx, value))

    def memcell_write(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [s for v in self._fields(ctx, value) for s in v.memcell_write(ctx)]

    def memcell_read(self, ctx: CompilationContext, value: Value, values: list[str]):
        i = 0
        for field in self._fields(ctx, value):
            size = field.memcell_size(ctx)
            field.memcell_read(ctx, values[i:i+size])
            i += size

    def table_copyable(self, ctx: CompilationContext, value: Value) -> bool:
        return True

    def table_size(self, ctx: CompilationContext, value: Value) -> int:
        return sum(v.memcell_size(ctx) for v in self._fields(ctx, value))

    def table_variables(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [s for v in self._fields(ctx, value) for s in v.table_variables(ctx)]

    def unpackable(self, ctx: CompilationContext, value: Value) -> bool:
        return value.getattr(ctx, False, "@unpack") is not None

    def unpack_count(self, ctx: CompilationContext, value: Value) -> int | None:
        return None

    def unpack(self, ctx: CompilationContext, value: Value) -> list[Value]:
        func = value.getattr_req(ctx, False, "@unpack")
        if not func.callable_with(ctx, []):
            ctx.error(
                f"Value of type '{func.type}' does not have the correct function signature to implement unpacking")
        return func.call(ctx, []).unpack_req(ctx)


@dataclass(slots=True)
class NamespaceType(Type):
    name: str | None
    variables: dict[str, Value]

    def __str__(self):
        return f"Namespace[{self.name if self.name else '?'}]"

    def __eq__(self, other):
        return isinstance(other, NamespaceType) and self.name == other.name and self.variables is other.variables

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        pass

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [_stringify(str(self))]

    def wrapped_type(self, ctx: CompilationContext) -> Type:
        return self

    def getattr(self, ctx: CompilationContext, value: Value, static: bool, name: str) -> Value | None:
        if static:
            if (val := self.variables.get(name)) is not None:
                return val

        return super(NamespaceType, self).getattr(ctx, value, static, name)

    def into(self, ctx: CompilationContext, value: Value, type_: Type) -> Value | None:
        if (func := value.getattr(ctx, True, "@cast")) is not None:
            if not func.callable_with(ctx, [TypeType(type_)]):
                ctx.error(
                    f"Value of type '{func.type}' does not have the correct function signature to implement type casts")
            return func.call(ctx, [Value.of_type(type_)])

        return super(NamespaceType, self).into(ctx, value, type_)

    def construct_from(self, ctx: CompilationContext, value: Value) -> Value | None:
        if (func := self.variables.get("@from")) is not None:
            if not func.callable_with(ctx, [value.type]):
                ctx.error(
                    f"Value of type '{func.type}' does not have the correct function signature to implement type casts")
            return func.call(ctx, [value])

        return super(NamespaceType, self).construct_from(ctx, value)

    def callable(self, ctx: CompilationContext, value: Value) -> bool:
        return True

    @staticmethod
    def _get_call_func(ctx: CompilationContext, value: Value) -> Value:
        return value.getattr(ctx, True, "@call")

    def call_with_suggestion(self, ctx: CompilationContext, value: Value) -> list[Type | None] | None:
        if (func := self._get_call_func(ctx, value)) is not None:
            return func.call_with_suggestion(ctx)

    def callable_with(self, ctx: CompilationContext, value: Value, param_types: list[Type]) -> bool:
        if (func := self._get_call_func(ctx, value)) is not None:
            return func.callable_with(ctx, param_types)

        return False

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        func = self._get_call_func(ctx, value)
        assert func is not None
        return func.call(ctx, params)

    @staticmethod
    def _get_index_func(ctx: CompilationContext, value: Value) -> Value:
        return value.getattr_req(ctx, True, "@index")

    def indexable(self, ctx: CompilationContext, value: Value) -> bool:
        return value.getattr(ctx, True, "@index") is not None

    def validate_index_types(self, ctx: CompilationContext, value: Value, indices: list[Type]) -> None | list[Type]:
        func = self._get_index_func(ctx, value)
        if func.callable_with(ctx, indices):
            return None
        else:
            if (suggested := func.call_with_suggestion(ctx)) is not None:
                return suggested
            else:
                return [EllipsisType()]

    def index(self, ctx: CompilationContext, value: Value, indices: list[Value]) -> Value:
        return self._get_index_func(ctx, value).call(ctx, indices)

    def unary_op(self, ctx: CompilationContext, value: Value, op: str) -> Value | None:
        if (name := StructBaseType.UNARY_OP_NAMES.get(op)) is not None:
            if (func := value.getattr(ctx, True, name)) is not None:
                if not func.callable_with(ctx, []):
                    ctx.error(f"Value of type '{func.type}' does not have the correct function signature to implement \
unary operator '{op}'")
                return func.call(ctx, [])

        return super(NamespaceType, self).unary_op(ctx, value, op)

    def binary_op(self, ctx: CompilationContext, value: Value, op: str, other: Value) -> Value | None:
        if (name := StructBaseType.BINARY_OP_NAMES.get(op)) is not None:
            if (func := value.getattr(ctx, True, name)) is not None:
                if not func.callable_with(ctx, [other.type]):
                    ctx.error(f"Value of type '{func.type}' does not have the correct function signature to implement \
binary operator '{op}' with other value of type '{other.type}'")
                return func.call(ctx, [other])

        return super(NamespaceType, self).binary_op(ctx, value, op, other)

    def binary_op_r(self, ctx: CompilationContext, other: Value, op: str, value: Value) -> Value | None:
        if (name := StructBaseType.BINARY_OP_NAMES.get(op)) is not None:
            name = f"@r_{name[1:]}"
            if (func := value.getattr(ctx, True, name)) is not None:
                if not func.callable_with(ctx, [other.type]):
                    ctx.error(f"Value of type '{func.type}' does not have the correct function signature to implement \
reversed binary operator '{op}' with other value of type '{other.type}'")
                return func.call(ctx, [other])

        return super(NamespaceType, self).binary_op_r(ctx, other, op, value)


@dataclass(slots=True)
class EnumBaseType(Type):
    name: str | None
    values_: dict[str, int]

    def __post_init__(self):
        self._instance_type: EnumInstanceType = EnumInstanceType(self)
        self.values: dict[str, Value] = {
            name: Value(self._instance_type, str(val))
            for name, val in self.values_.items()
        }
        self.bottom_values: dict[str, Value] = {
            "::" + name: val
            for name, val in self.values.items()
        }
        self.ordered_values = list(self.values_.items())
        self.ordered_values.sort(key=lambda x: x[1])

    def __str__(self):
        return f"Enum[{self.name if self.name else '?'}]"

    def __eq__(self, other):
        return isinstance(other, EnumBaseType) and self.name == other.name and self.values_ == other.values_

    def wrapped_type(self, ctx: CompilationContext) -> Type:
        return self._instance_type

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        pass

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [_stringify(str(self))]

    def getattr(self, ctx: CompilationContext, value: Value, static: bool, name: str) -> Value | None:
        if static:
            if (val := self.values.get(name)) is not None:
                return val

            elif val == "_len":
                return Value.of_number(len(self.values))

        return super(EnumBaseType, self).getattr(ctx, value, static, name)

    def callable(self, ctx: CompilationContext, value: Value) -> bool:
        return True

    def call_with_suggestion(self, ctx: CompilationContext, value: Value) -> list[Type | None] | None:
        return [NumberType()]

    def callable_with(self, ctx: CompilationContext, value: Value, param_types: list[Type]) -> bool:
        return len(param_types) == 1 and NumberType().contains(param_types[0])

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        return Value(self._instance_type, params[0].value)

    def bottom_scope(self) -> dict[str, Value] | None:
        return self.bottom_values


@dataclass(slots=True)
class EnumInstanceType(Type):
    base: EnumBaseType

    def __str__(self):
        return f"{self.base.name if self.base.name else '?'}"

    def __eq__(self, other):
        return isinstance(other, EnumInstanceType) and self.base == other.base

    def assign_default(self, ctx: CompilationContext, value: Value):
        ctx.emit(Instruction.set(value.value, "0"))

    def into(self, ctx: CompilationContext, value: Value, type_: Type) -> Value | None:
        if type_.contains(NumberType()):
            return Value(NumberType(), value.value, value.const)

        return super(EnumInstanceType, self).into(ctx, value, type_)

    def bottom_scope(self) -> dict[str, Value] | None:
        return self.base.bottom_scope()

    def memcell_serializable(self, ctx: CompilationContext, value: Value) -> bool:
        return True

    def _name_impl(self, ctx: CompilationContext, value: Value) -> Value:
        result = ctx.tmp()
        ctx.emit(Instruction.TableRead([result], [
            [f"\"{name}\""] for name, _ in self.base.ordered_values
        ], value.value))
        return Value(StringType(), result, False)

    def getattr(self, ctx: CompilationContext, value: Value, static: bool, name: str) -> Value | None:
        if not static:
            if name == "base":
                return Value(self.base, "")

            elif name == "name":
                return Value(SpecialFunctionType(
                    f"{self.base.name}.name",
                    [],
                    StringType(),
                    lambda ctx_, params: self._name_impl(ctx_, value)
                ), "")

        return super(self, EnumInstanceType).getattr(ctx, value, static, name)


class BuiltinEnumBaseType(Type):
    name: str
    values: dict[str, Value]
    bottom_values: dict[str, Value]
    copyable: bool
    _instance_type: BuiltinEnumInstanceType

    def __init__(self, name: str, values: set[str], content: bool, copyable: bool):
        self.name = name
        self._instance_type = BuiltinEnumInstanceType(self)
        self.values = {
            val.replace("-", "_"): Value(self._instance_type, ("@" if content else "") + val)
            for val in values
        }
        self.bottom_values = {
            "::" + name: val
            for name, val in self.values.items()
        }
        self.copyable = copyable

    def __str__(self):
        return f"Enum[{self.name}]"

    def __eq__(self, other):
        return isinstance(other, BuiltinEnumBaseType) and self.name == other.name

    def wrapped_type(self, ctx: CompilationContext) -> Type:
        return self._instance_type

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        pass

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [_stringify(str(self))]

    def getattr(self, ctx: CompilationContext, value: Value, static: bool, name: str) -> Value | None:
        if static:
            if (val := self.values.get(name)) is not None:
                return val

        return super(BuiltinEnumBaseType, self).getattr(ctx, value, static, name)

    def bottom_scope(self) -> dict[str, Value] | None:
        return self.bottom_values

    def table_copyable(self, ctx: CompilationContext, value: Value) -> bool:
        return True


@dataclass(slots=True)
class BuiltinEnumInstanceType(Type):
    base: BuiltinEnumBaseType

    def __str__(self):
        return self.base.name

    def __eq__(self, other):
        return isinstance(other, BuiltinEnumInstanceType) and self.base == other.base

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        if not self.base.copyable:
            ctx.error(f"Enum of type '{self.base.name}' is not copyable (must be passed by reference)")

        super(BuiltinEnumInstanceType, self).assign(ctx, value, other)

    def bottom_scope(self) -> dict[str, Value] | None:
        return self.base.bottom_scope()
