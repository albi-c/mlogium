from __future__ import annotations

import contextlib
from dataclasses import dataclass
from abc import abstractmethod, ABC
from typing import Callable, Iterable, Optional

from .compilation_context import CompilationContext
from .instruction import Instruction
from .abi import ABI
from .node import Node


@dataclass(slots=True)
class BaseValue(ABC):
    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def deref(self, ctx: CompilationContext) -> Value:
        raise NotImplementedError

    def deref_as(self, ctx: CompilationContext, type_: Type) -> Value | None:
        val = self.deref(ctx)
        if type_.contains(val.type):
            return val
        else:
            return None

    def deref_as_req(self, ctx: CompilationContext, type_: Type) -> Value:
        val = self.deref(ctx)
        if not type_.contains(val.type):
            ctx.error(f"Value {val} cannot be cast to type '{type_}'")
        return val

    @abstractmethod
    def assign(self, ctx: CompilationContext, value: Value):
        raise NotImplementedError


@dataclass(slots=True)
class Value(BaseValue):
    type: Type
    value: str
    const: bool = True
    no_discard: bool = False

    def __str__(self):
        return f"of type '{self.type}'"

    def deref(self, ctx: CompilationContext) -> Value:
        return self

    def assign(self, ctx: CompilationContext, value: Value):
        ctx.error(f"Assignment to constant value {self}")

    def with_no_discard(self, no_discard: bool = True) -> Value:
        return Value(self.type, self.value, self.const, no_discard)

    def copy(self, ctx: CompilationContext, to: str | None = None) -> VariableLValue:
        val = Value.var(ctx, self.type, to)
        val.assign(ctx, self)
        return val

    @staticmethod
    def null() -> Value:
        return Value(NullType(), "null")

    @staticmethod
    def of_number(val: int | float) -> Value:
        return Value(NumberType(), str(val))

    @staticmethod
    def of_string(string: str) -> Value:
        return Value(StringType(), string)

    @staticmethod
    def of_boolean(val: bool) -> Value:
        return Value(NumberType(), "1" if val else "0")

    @staticmethod
    def of_type(type_: Type) -> Value:
        return Value(TypeType(type_), "")

    @staticmethod
    def tmp(ctx: CompilationContext, type_: Type) -> VariableLValue:
        return VariableLValue(type_, ctx.tmp())

    @staticmethod
    def tmp_v(ctx: CompilationContext, type_: Type) -> Value:
        return Value(type_, ctx.tmp())

    @staticmethod
    def var(ctx: CompilationContext, type_: Type, name: str | None = None, const: bool = False) -> VariableLValue:
        name = name if name is not None else ctx.tmp()
        return VariableLValue(type_, name, const)

    def wrapped_type(self, ctx: CompilationContext) -> Type | None:
        return self.type.wrapped_type(ctx)

    def wrapped_type_req(self, ctx: CompilationContext) -> Type:
        type_ = self.wrapped_type(ctx)
        if type_ is None:
            ctx.error(f"Value {self} does not represent a type")
        return type_

    def cast(self, ctx: CompilationContext, type_: Type) -> Value | None:
        if type_.contains(self.type):
            return self

        elif isinstance(type_, UnionType):
            for t in type_.types:
                if (val := self.cast(ctx, t)) is not None:
                    return val

        if val := self.type.cast(ctx, self, type_):
            return val

        return Value.cast_from(ctx, type_, self)

    def cast_req(self, ctx: CompilationContext, type_: Type) -> Value:
        val = self.cast(ctx, type_)
        if val is None:
            ctx.error(f"Value {self} cannot be cast to type '{type_}'")
        return val

    @staticmethod
    def cast_from(ctx: CompilationContext, type_: Type, other: Value, to: str | None = None) -> VariableLValue | None:
        val = Value.var(ctx, type_, to)
        if not type_.cast_from(ctx, val.value, other):
            return None
        return val

    def to_print(self, ctx: CompilationContext) -> list[str]:
        return self.type.to_print(ctx, self)

    def to_condition(self, ctx: CompilationContext) -> str | None:
        return self.type.to_condition(ctx, self)

    def to_condition_req(self, ctx: CompilationContext) -> str:
        cond = self.to_condition(ctx)
        if cond is None:
            ctx.error(f"Value {self} is not usable as a condition")
        return cond

    def unpackable(self, ctx: CompilationContext) -> bool:
        return self.type.unpackable(ctx, self)

    def unpack_count(self, ctx: CompilationContext) -> int:
        return self.type.unpack_count(ctx, self)

    def unpack(self, ctx: CompilationContext) -> list[BaseValue]:
        return self.type.unpack(ctx, self)

    def getattr(self, ctx: CompilationContext, static: bool, name: str) -> BaseValue | None:
        return self.type.getattr(ctx, self, static, name)

    def getattr_req(self, ctx: CompilationContext, static: bool, name: str) -> BaseValue:
        attr = self.getattr(ctx, static, name)
        if attr is None:
            ctx.error(f"Value {self} has no {'static ' if static else ''}attribute '{name}'")
        return attr

    def callable(self, ctx: CompilationContext) -> bool:
        return self.type.callable(ctx, self)

    def call_signature(self, ctx: CompilationContext) -> FunctionSignature | None:
        return self.type.call_signature(ctx, self)

    def call_types(self, ctx: CompilationContext, param_count: int) -> list[Type] | None:
        return self.type.call_types(ctx, self, param_count)

    def callable_with(self, ctx: CompilationContext, param_types: list[Type]) -> bool:
        return self.type.callable_with(ctx, self, param_types)

    def call(self, ctx: CompilationContext, params: list[BaseValue]) -> BaseValue:
        return self.type.call(ctx, self, params)

    def index_signature(self, ctx: CompilationContext) -> FunctionSignature | None:
        return self.type.index_signature(ctx, self)

    def index_types(self, ctx: CompilationContext, param_count: int) -> list[Type] | None:
        return self.type.index_types(ctx, self, param_count)

    def indexable_with(self, ctx: CompilationContext, param_types: list[Type]) -> bool:
        return self.type.indexable_with(ctx, self, param_types)

    def index(self, ctx: CompilationContext, params: list[BaseValue]) -> BaseValue:
        return self.type.index(ctx, self, params)

    def binary_op(self, ctx: CompilationContext, op: str, right: BaseValue) -> BaseValue | None:
        return self.type.binary_op(ctx, self, op, right)

    def binary_op_r(self, ctx: CompilationContext, left: BaseValue, op: str) -> BaseValue | None:
        return self.type.binary_op_r(ctx, left, op, self)

    def unary_op(self, ctx: CompilationContext, op: str) -> BaseValue | None:
        return self.type.unary_op(ctx, self, op)

    def iterate(self, ctx: CompilationContext) -> ValueIterator | None:
        return self.type.iterate(ctx, self)

    def iterate_req(self, ctx: CompilationContext) -> ValueIterator:
        it = self.iterate(ctx)
        if it is None:
            ctx.error(f"Value {self} is not iterable")
        return it

    def mem_support(self, ctx: CompilationContext) -> int:
        return self.type.mem_support(ctx, self)

    def memcell_support(self, ctx: CompilationContext) -> bool:
        return self.type.memcell_support(ctx, self)

    def table_support(self, ctx: CompilationContext) -> bool:
        return self.type.table_support(ctx, self)

    def mem_size(self, ctx: CompilationContext) -> int:
        return self.type.mem_size(ctx, self)

    def mem_variables(self, ctx: CompilationContext) -> list[str]:
        return self.type.mem_variables(ctx, self)


@dataclass(slots=True)
class LValue(BaseValue, ABC):
    pass


@dataclass(slots=True)
class VariableLValue(LValue):
    type: Type
    value: str
    const: bool = True

    def __str__(self):
        return f"of type '{self.type}'"

    def deref(self, ctx: CompilationContext) -> Value:
        return Value(self.type, self.value)

    def assign(self, ctx: CompilationContext, value: Value):
        if self.const:
            ctx.error(f"Assignment to constant value {self}")
        self.type.assign(ctx, self.value, value)


@dataclass(slots=True)
class UnderscoreLValue(LValue):
    def __str__(self):
        return "underscore"

    def deref(self, ctx: CompilationContext) -> Value:
        ctx.error(f"Underscore can only be assigned to")
        return Value.null()

    def assign(self, ctx: CompilationContext, value: Value):
        pass


@dataclass(slots=True)
class MemcellRefLValue(LValue):
    cell: str
    index: Value

    def __str__(self):
        return "MemCellRef"

    def deref(self, ctx: CompilationContext) -> Value:
        result = Value.tmp_v(ctx, NumberType())
        ctx.emit(Instruction.read(result.value, self.cell, self.index.value))
        return result

    def assign(self, ctx: CompilationContext, value: Value):
        value = value.cast_req(ctx, NumberType())
        ctx.emit(Instruction.write(value.value, self.cell, self.index.value))


@dataclass(slots=True)
class TupleRefLValue(LValue):
    tup: Value
    index: Value

    def __str__(self):
        return "TupleRef"

    def deref(self, ctx: CompilationContext) -> Value:
        raise NotImplementedError  # TODO

    def assign(self, ctx: CompilationContext, value: Value):
        raise NotImplementedError  # TODO


@dataclass(slots=True)
class FunctionSignature:
    @dataclass(slots=True)
    class Param:
        name: str | None
        type: Type | None
        reference: bool

    params: list[Param]
    result: Type | None
    variadic: bool

    @classmethod
    def index(cls, types: Iterable[Type], result: Type | None) -> FunctionSignature:
        return cls([cls.Param(None, t, False) for t in types], result, False)

    def param_types(self) -> list[Optional[Type]]:
        raise NotImplementedError  # TODO

    def check_params(self, param_types: list[Type]) -> bool:
        raise NotImplementedError  # TODO


@dataclass(slots=True)
class ValueIterator(ABC):
    @abstractmethod
    def has_value(self, ctx: CompilationContext) -> str:
        raise NotImplementedError

    @abstractmethod
    def next_value(self, ctx: CompilationContext) -> BaseValue:
        raise NotImplementedError

    @abstractmethod
    def end_loop(self, ctx: CompilationContext):
        raise NotImplementedError


@dataclass(slots=True, eq=True)
class Type:
    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    def __or__(self, other):
        if isinstance(other, Type):
            if isinstance(self, UnionType):
                if isinstance(other, UnionType):
                    return UnionType(self.types + other.types)
                else:
                    return UnionType(self.types + [other])
            else:
                if isinstance(other, UnionType):
                    return UnionType([self] + other.types)
                else:
                    return UnionType([self, other])

        return NotImplemented

    def contains(self, other: Type) -> bool:
        return self == other

    def wrapped_type(self, ctx: CompilationContext) -> Type | None:
        return None

    def assign(self, ctx: CompilationContext, to: str, value: Value):
        raise NotImplementedError

    def cast(self, ctx: CompilationContext, value: Value, type_: Type) -> Value | None:
        return None

    def cast_from(self, ctx: CompilationContext, to: str, other: Value) -> bool:
        return False

    def to_print(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [value.value]

    def to_condition(self, ctx: CompilationContext, value: Value) -> str | None:
        return None

    def unpackable(self, ctx: CompilationContext, value: Value) -> bool:
        return False

    def unpack_count(self, ctx: CompilationContext, value: Value) -> int:
        raise NotImplementedError

    def unpack(self, ctx: CompilationContext, value: Value) -> list[BaseValue]:
        raise NotImplementedError

    def getattr(self, ctx: CompilationContext, value: Value, static: bool, name: str) -> BaseValue | None:
        return None

    def callable(self, ctx: CompilationContext, value: Value) -> bool:
        return self.call_signature(ctx, value) is not None

    def call_signature(self, ctx: CompilationContext, value: Value) -> FunctionSignature | None:
        return None

    def call_types(self, ctx: CompilationContext, value: Value, param_count: int) -> list[Type] | None:
        raise NotImplementedError  # TODO

    def callable_with(self, ctx: CompilationContext, value: Value, param_types: list[Type]) -> bool:
        if sig := self.call_signature(ctx, value):
            raise NotImplementedError  # TODO
        return False

    def call(self, ctx: CompilationContext, value: Value, params: list[BaseValue]) -> BaseValue:
        raise NotImplementedError

    def indexable(self, ctx: CompilationContext, value: Value) -> bool:
        return self.index_signature(ctx, value) is not None

    def index_signature(self, ctx: CompilationContext, value: Value) -> FunctionSignature | None:
        return None

    def index_types(self, ctx: CompilationContext, value: Value, param_count: int) -> list[Type] | None:
        raise NotImplementedError  # TODO

    def indexable_with(self, ctx: CompilationContext, value: Value, param_types: list[Type]) -> bool:
        if sig := self.index_signature(ctx, value):
            raise NotImplementedError  # TODO
        return False

    def index(self, ctx: CompilationContext, value: Value, params: list[BaseValue]) -> BaseValue:
        raise NotImplementedError

    def binary_op(self, ctx: CompilationContext, value: Value, op: str, right: BaseValue) -> BaseValue | None:
        right = right.deref(ctx)

        if op == "!==":
            return self.binary_op(ctx, value, "===", right).deref(ctx).unary_op(ctx, "!")

        if op in ("==", "!=", "==="):
            raise NotImplementedError  # TODO

        # TODO: +=, ...

        return right.type.binary_op_r(ctx, value, op, right)

    def binary_op_r(self, ctx: CompilationContext, left: BaseValue, op: str, value: Value) -> BaseValue | None:
        return None

    def unary_op(self, ctx: CompilationContext, value: Value, op: str) -> BaseValue | None:
        if op == "!":
            cond = value.to_condition_req(ctx)
            result = Value.tmp(ctx, NumberType())
            ctx.emit(Instruction.op("notEqual", result.value, cond, "0"))
            return result

        return None

    def iterate(self, ctx: CompilationContext, value: Value) -> ValueIterator | None:
        return None

    def mem_support(self, ctx: CompilationContext, value: Value) -> int:
        """
        0 - none
        1 - table
        2 - cell
        """
        return 0

    def memcell_support(self, ctx: CompilationContext, value: Value) -> bool:
        return self.mem_support(ctx, value) >= 2

    def table_support(self, ctx: CompilationContext, value: Value) -> bool:
        return self.mem_support(ctx, value) >= 1

    def mem_size(self, ctx: CompilationContext, value: Value) -> int:
        return 1

    def mem_variables(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [value.value]


@dataclass(slots=True, eq=True)
class UnionType(Type):
    types: list[Type]

    def __str__(self):
        return " | ".join(map(str, self.types))

    def contains(self, other: Type) -> bool:
        return any(t.contains(other) for t in self.types)

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        raise NotImplementedError

    def to_print(self, ctx: CompilationContext, value: Value) -> list[str]:
        raise NotImplementedError


@dataclass(slots=True, eq=True)
class NullType(Type):
    def __str__(self):
        return "null"


@dataclass(slots=True, eq=True)
class NumberType(Type):
    def __str__(self):
        return "num"


@dataclass(slots=True, eq=True)
class AnyType(Type):
    def __str__(self):
        return "?"

    def contains(self, other: Type) -> bool:
        return True

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        ctx.error("Cannot assign to value of unknown type")

    def to_print(self, ctx: CompilationContext, value: Value) -> list[str]:
        ctx.error("Cannot print value of unknown type")
        return []


@dataclass(slots=True, eq=True)
class AnyTrivialType(Type):
    def __str__(self):
        return "?"

    def contains(self, other: Type) -> bool:
        return UnionType([NullType(), NumberType(), StringType(), BlockType(),
                          UnitType(), BlockType(), ControllerType()]).contains(other) \
            or isinstance(other, EnumInstanceType) \
            or (isinstance(other, BuiltinEnumInstanceType) and other.base.copyable)

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        ctx.emit(
            Instruction.write(value.value, other.value)
        )

    def to_print(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [value.value]


@dataclass(slots=True, eq=True)
class EllipsisType(Type):
    def __str__(self):
        return "..."

    def contains(self, other: Type) -> bool:
        return True

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        raise NotImplementedError

    def to_print(self, ctx: CompilationContext, value: Value) -> list[str]:
        raise NotImplementedError


@dataclass(slots=True, eq=True)
class TypeType(Type):
    type: Type

    def __str__(self):
        return f"Type[{self.type}]"

    def wrapped_type(self, ctx: CompilationContext | None) -> Type:
        return self.type

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        pass

    def to_print(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [f"'{self}'"]

    def getattr(self, ctx: CompilationContext, value: Value, static: bool, name: str) -> Value | None:
        if not static:
            val = Value(self.type, "null")

            if name == "name":
                return Value.of_string(f"'{self.type}'")

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


@dataclass(slots=True, eq=True)
class GenericTypeType(Type):
    def __str__(self):
        return "Type"

    def contains(self, other: Type) -> bool:
        return True

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        pass

    def to_print(self, ctx: CompilationContext, value: Value) -> list[str]:
        return ["'Type'"]


@dataclass(slots=True, eq=True)
class TupleTypeSourceType(Type):
    def __str__(self):
        return "Tuple"

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        pass

    def to_print(self, ctx: CompilationContext, value: Value) -> list[str]:
        return ["'Tuple'"]

    def indexable(self, ctx: CompilationContext, value: Value) -> bool:
        return True

    def validate_index_types(self, ctx: CompilationContext, value: Value, indices: list[Type]) -> None | list[Type]:
        return None

    def index(self, ctx: CompilationContext, value: Value, indices: list[Value]) -> Value:
        types = [val.type.wrapped_type(ctx) for val in indices]
        return Value.of_type(TupleType(types))

    def callable(self, ctx: CompilationContext, value: Value) -> bool:
        return True

    def call_signature(self, ctx: CompilationContext, value: Value) -> FunctionSignature | None:
        return FunctionSignature([
            FunctionSignature.Param("type", GenericTypeType(), False),
            FunctionSignature.Param("count", NumberType(), False)
        ], GenericTypeType(), False)

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        count = 0
        try:
            count = int(params[1].value)
        except ValueError:
            ctx.error(f"Count is not a compile time known integer")
        if count < 0:
            ctx.error(f"Count cannot be negative")

        return Value.of_type(TupleType([params[0].type.wrapped_type(ctx)] * count))


@dataclass(slots=True, eq=True)
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

    def to_condition(self, ctx: CompilationContext, value: Value) -> str | None:
        return value.value

    def unary_op(self, ctx: CompilationContext, value: Value, op: str) -> Value | None:
        if op == "-":
            try:
                n = float(value.value)
            except ValueError:
                pass
            else:
                if n.is_integer():
                    n = int(n)
                return Value.of_number(-n)

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

    def mem_support(self, ctx: CompilationContext, value: Value) -> int:
        return 2

    def cast(self, ctx: CompilationContext, value: Value, type_: Type) -> Value | None:
        if isinstance(type_, FunctionRefType):
            return Value(type_, value.value, True)


@dataclass(slots=True, eq=True)
class StringType(Type):
    class ValueIterator(ValueIterator):
        string: Value
        index: Value
        length: Value

        def __init__(self, ctx: CompilationContext, string: Value):
            self.string = string
            self.index = Value(NumberType(), ctx.tmp(), False)
            self.index.assign(ctx, Value.of_number(0))
            self.length = string.getattr_req(ctx, False, "len").deref(ctx)

        def has_value(self, ctx: CompilationContext) -> Value:
            return self.index.binary_op(ctx, "<", self.length).deref(ctx)

        def next_value(self, ctx: CompilationContext) -> Value:
            return self.string.index(ctx, [self.index]).deref(ctx)

        def end_loop(self, ctx: CompilationContext):
            self.index.binary_op(ctx, "+=", Value.of_number(1))

    def __str__(self):
        return "str"

    def to_condition(self, ctx: CompilationContext, value: Value) -> str | None:
        return value.value

    def mem_support(self, ctx: CompilationContext, value: Value) -> int:
        return 1

    def indexable(self, ctx: CompilationContext, value: Value) -> bool:
        return True

    def index_signature(self, ctx: CompilationContext, value: Value) -> FunctionSignature | None:
        return FunctionSignature.index([NumberType()], NumberType())

    def getattr(self, ctx: CompilationContext, value: Value, static: bool, name: str) -> Value | None:
        if not static:
            if name in ("len", "length", "size"):
                result = Value(NumberType(), ctx.tmp())
                ctx.emit(Instruction.sensor_asm(result.value, value, "@size"))
                return result

        return super(StringType, self).getattr(ctx, value, static, name)

    def index(self, ctx: CompilationContext, value: Value, indices: list[Value]) -> Value:
        result = Value(NumberType(), ctx.tmp(), False)
        ctx.emit(Instruction.read(result.value, value.value, indices[0].value))
        return result

    def iterate(self, ctx: CompilationContext, value: Value) -> ValueIterator | None:
        return StringType.ValueIterator(ctx, value)


@dataclass(slots=True, eq=True)
class SoundBaseType(TypeType):
    def __init__(self):
        super().__init__(SoundType())

    def getattr(self, ctx: CompilationContext, value: Value, static: bool, name: str) -> Value | None:
        if static:
            return Value(SoundType(), "@sfx-" + name.replace("_", "-"))

        return super(SoundBaseType, self).getattr(ctx, value, static, name)


@dataclass(slots=True, eq=True)
class SoundType(Type):
    def __str__(self):
        return "Sound"

    def to_condition(self, ctx: CompilationContext, value: Value) -> str | None:
        return value.value

    def mem_support(self, ctx: CompilationContext, value: Value) -> int:
        return 1


@dataclass(slots=True, eq=True)
class BlockBaseType(TypeType):
    def __init__(self):
        super().__init__(BlockType())

    def getattr(self, ctx: CompilationContext, value: Value, static: bool, name: str) -> Value | None:
        if static:
            return Value(BlockType(), name)

        return super(BlockBaseType, self).getattr(ctx, value, static, name)


@dataclass(slots=True, eq=True)
class BlockType(Type):
    def __str__(self):
        return "Block"

    def to_condition(self, ctx: CompilationContext, value: Value) -> str | None:
        return value.value

    def indexable(self, ctx: CompilationContext, value: Value) -> bool:
        return True

    def index_signature(self, ctx: CompilationContext, value: Value) -> FunctionSignature | None:
        return FunctionSignature.index([NumberType()], AnyType())

    def index(self, ctx: CompilationContext, value: Value, indices: list[BaseValue]) -> BaseValue:
        # return Value(MemoryCellReferenceType(value, indices[0]), "", False)
        return MemcellRefLValue(value, indices[0].deref(ctx))

    @staticmethod
    def _memcell_write_impl(ctx: CompilationContext, params: list[BaseValue], value: Value) -> Value:
        val = params[1].deref(ctx)
        if not val.memcell_support(ctx):
            ctx.error(f"Value of type '{val.type}' is not serializable")

        next_index = params[0].deref(ctx)
        for v in val.mem_variables(ctx):
            ctx.emit(Instruction.write(v, value, next_index.value))
            next_index = next_index.binary_op(ctx, "+", Value.of_number(1)).deref(ctx)

        return Value.null()

    @staticmethod
    def _memcell_read_impl(ctx: CompilationContext, params: list[BaseValue], value: Value) -> Value:
        val = Value.tmp_v(ctx, params[1].deref(ctx).wrapped_type_req(ctx))
        if not val.memcell_support(ctx):
            ctx.error(f"Value of type '{val.type}' is not deserializable")

        next_index = params[0].deref(ctx)
        for var in val.mem_variables(ctx):
            ctx.emit(Instruction.read(var, value, next_index.value))
            next_index = next_index.binary_op(ctx, "+", Value.of_number(1)).deref(ctx)

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

    def mem_support(self, ctx: CompilationContext, value: Value) -> int:
        return 1


@dataclass(slots=True, eq=True)
class UnitType(Type):
    def __str__(self):
        return "Unit"

    def to_condition(self, ctx: CompilationContext, value: Value) -> str | None:
        return value.value

    def mem_support(self, ctx: CompilationContext, value: Value) -> int:
        return 1


@dataclass(slots=True, eq=True)
class ControllerType(Type):
    def __str__(self):
        return "Controller"

    def mem_support(self, ctx: CompilationContext, value: Value) -> int:
        return 1


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

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [ABI.attribute(value.value, "start"), "\"..\"", ABI.attribute(value.value, "end"), "\"..\"",
                ABI.attribute(value.value, "step")]

    def getattr(self, ctx: CompilationContext, value: Value, static: bool, name: str) -> BaseValue | None:
        if not static:
            if name in ("start", "end", "step"):
                return VariableLValue(NumberType(), ABI.attribute(value, name))

        return super(RangeWithStepType, self).getattr(ctx, value, static, name)

    def iterate(self, ctx: CompilationContext, value: Value) -> ValueIterator | None:
        return RangeWithStepType.ValueIterator(
            ctx,
            self._attr(value, "start", True),
            self._attr(value, "end", True),
            self._attr(value, "step", True)
        )

    def mem_support(self, ctx: CompilationContext, value: Value) -> int:
        return 2

    def mem_size(self, ctx: CompilationContext, value: Value) -> int:
        return 3

    def mem_variables(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [
            ABI.attribute(value, "start"),
            ABI.attribute(value, "end"),
            ABI.attribute(value, "step")
        ]
