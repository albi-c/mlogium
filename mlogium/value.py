from __future__ import annotations

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
        if NullType().contains(other.type):
            return self.assign_default(ctx)

        if self.const:
            ctx.error(f"Assignment to constant of type '{self.type}'")

        self.type.assign(ctx, self, other.into_req(ctx, self.assignable_type()))

        if self.const_on_write:
            self.const = True

    def assign_default(self, ctx: CompilationContext):
        if self.const:
            ctx.error(f"Assignment to constant of type '{self.type}'")

        self.type.assign_default(ctx, self)

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

    def bottom_scope(self) -> dict[str, Value] | None:
        return self.type.bottom_scope()


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

    def unpack_count(self) -> int:
        return -1

    def unpack(self, ctx: CompilationContext, value: Value) -> list[Value]:
        raise NotImplementedError

    def callable(self) -> bool:
        return False

    def call_with_suggestion(self) -> list[Type | None] | None:
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

    def bottom_scope(self) -> dict[str, Value] | None:
        return None


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


class GenericTypeType(Type):
    def __str__(self):
        return "Type"

    def __eq__(self, other):
        return isinstance(other, GenericTypeType)

    def contains(self, other: Type) -> bool:
        return self == other or isinstance(other, TypeType)

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

    def indexable(self) -> bool:
        return True

    def validate_index_count(self, count: int) -> int:
        return -1

    def index(self, ctx: CompilationContext, value: Value, indices: list[Value]) -> Value:
        types = [val.type.wrapped_type(ctx) for val in indices]
        return Value.of_type(TupleType(types))


class BlockSourceType(Type):
    def __str__(self):
        return "ExternBlock"

    def __eq__(self, other):
        return isinstance(other, BlockSourceType)

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        pass

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [_stringify(str(self))]

    def getattr(self, ctx: CompilationContext, value: Value, static: bool, name: str) -> Value | None:
        if static:
            return Value(BlockType(), name)

        return super(BlockSourceType, self).getattr(ctx, value, static, name)


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


class StringType(Type):
    def __str__(self):
        return "str"

    def __eq__(self, other):
        return isinstance(other, StringType)

    def assign_default(self, ctx: CompilationContext, value: Value):
        ctx.emit(Instruction.set(value.value, "\"\""))

    def to_condition(self, ctx: CompilationContext, value: Value) -> str | None:
        return value.value


class BlockType(Type):
    def __str__(self):
        return "Block"

    def __eq__(self, other):
        return isinstance(other, BlockType)

    def to_condition(self, ctx: CompilationContext, value: Value) -> str | None:
        return value.value


class UnitType(Type):
    def __str__(self):
        return "Unit"

    def __eq__(self, other):
        return isinstance(other, UnitType)

    def to_condition(self, ctx: CompilationContext, value: Value) -> str | None:
        return value.value


class ControllerType(Type):
    def __str__(self):
        return "Controller"

    def __eq__(self, other):
        return isinstance(other, ControllerType)


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

    def unpack_count(self) -> int:
        return len(self.types)

    def unpack(self, ctx: CompilationContext, value: Value) -> list[Value]:
        return [Value(t, ABI.attribute(value.value, i), value.const) for i, t in enumerate(self.types)]


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

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [ABI.attribute(value.value, "start"), "\"..\"", ABI.attribute(value.value, "end")]

    def getattr(self, ctx: CompilationContext, value: Value, static: bool, name: str) -> Value | None:
        if not static:
            if name in ("start", "end"):
                return self._attr(value, name, value.const)

        return super(RangeType, self).getattr(ctx, value, static, name)


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
        return isinstance(other, SpecialFunctionType) and self.name == other.name and self.params == other.params and \
            self.result == other.result

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

    def callable(self) -> bool:
        return True

    def call_with_suggestion(self) -> list[Type | None] | None:
        return self.input_params

    def callable_with(self, param_types: list[Type]) -> bool:
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

        def __str__(self):
            return f"{'&' if self.reference else ''}{self.name}{': ' + str(self.type) if self.type else ''}"

    name: str
    params: list[FunctionType.Param]
    result: Type | None
    code: Node

    def __str__(self):
        return f"fn({', '.join(map(str, self.params))}){' -> ' + str(self.result) if self.result else ''}"

    def __eq__(self, other):
        return isinstance(other, FunctionType) and self.params == other.params and self.result == other.result and \
            self.code == other.code

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        pass

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [_stringify(str(self))]

    def callable(self) -> bool:
        return True

    def call_with_suggestion(self) -> list[Type | None] | None:
        return [p.type for p in self.params]

    def callable_with(self, param_types: list[Type]) -> bool:
        return len(self.params) == len(param_types) and \
            all(p.type is None or p.type.contains(t) for p, t in zip(self.params, param_types))

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        with ctx.scope.function_call(ctx, f"$function_{self.name}:{ctx.tmp_num()}"):
            for dst, src in zip(self.params, params):
                if dst.reference:
                    ctx.scope.declare_special(dst.name, src)
                else:
                    ctx.scope.declare(dst.name, dst.type if dst.type is not None else src.type, False).assign(ctx, src)

            result = ctx.generate_node(self.code)

            return_val = Value(self.result if self.result is not None else result.type,
                               ABI.return_value(ctx.scope.get_function()), False)
            if NullType().contains(result.type):
                return_val.assign_default(ctx)
            else:
                return_val.assign(ctx, result)

            ctx.emit(Instruction.label(ABI.function_end(ctx.scope.get_function())))

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

    def callable(self) -> bool:
        return True

    def call_with_suggestion(self) -> list[Type | None] | None:
        return [p.type for p in self.params]

    def callable_with(self, param_types: list[Type]) -> bool:
        return len(self.params) == len(param_types) and \
            all(p.type is None or p.type.contains(t) for p, t in zip(self.params, param_types))

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        with ctx.scope.function_call(ctx, f"$lambda_{self.name}:{ctx.tmp_num()}"):
            for capture in self.captures:
                ctx.scope.declare_special(capture.name, capture.value)

            for dst, src in zip(self.params, params):
                if dst.reference:
                    ctx.scope.declare_special(dst.name, src)
                else:
                    ctx.scope.declare(dst.name, dst.type if dst.type is not None else src.type, False).assign(ctx, src)

            result = ctx.generate_node(self.code)

            return_val = Value(self.result if self.result is not None else result.type,
                               ABI.return_value(ctx.scope.get_function()), False)
            if NullType().contains(result.type):
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


@dataclass(slots=True)
class StructMethodType(Type):
    self_value: Value
    name: str
    params: list[FunctionType.Param]
    result: Type | None
    code: Node

    @classmethod
    def create_value(cls, self_value: Value, data: StructMethodData) -> Value:
        return Value(cls(self_value, data.name, data.params, data.result, data.code), "")

    def __str__(self):
        return f"fn({', '.join(map(str, self.params))}){' -> ' + str(self.result) if self.result else ''}"

    def __eq__(self, other):
        return isinstance(other, StructMethodType) and self.params == other.params and self.result == other.result and \
            self.code == other.code and self.self_value == other.self_value

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        pass

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [_stringify(str(self))]

    def callable(self) -> bool:
        return True

    def call_with_suggestion(self) -> list[Type | None] | None:
        return [p.type for p in self.params]

    def callable_with(self, param_types: list[Type]) -> bool:
        return len(self.params) == len(param_types) and \
            all(p.type is None or p.type.contains(t) for p, t in zip(self.params, param_types))

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        with ctx.scope.function_call(ctx, f"$method_{self.name}:{ctx.tmp_num()}"):
            ctx.scope.declare_special("self", self.self_value)

            for dst, src in zip(self.params, params):
                if dst.reference:
                    ctx.scope.declare_special(dst.name, src)
                else:
                    ctx.scope.declare(dst.name, dst.type if dst.type is not None else src.type, False).assign(ctx, src)

            result = ctx.generate_node(self.code)

            return_val = Value(self.result if self.result is not None else result.type,
                               ABI.return_value(ctx.scope.get_function()), False)
            if NullType().contains(result.type):
                return_val.assign_default(ctx)
            else:
                return_val.assign(ctx, result)

            ctx.emit(Instruction.label(ABI.function_end(ctx.scope.get_function())))

            return return_val


@dataclass(slots=True)
class StructBaseType(Type):
    name: str | None
    fields: list[tuple[str, Type]]
    static_fields: dict[str, Value]
    methods: dict[str, tuple[bool, StructMethodData]]
    static_methods: dict[str, StructMethodData]

    def __post_init__(self):
        self._instance_type: StructInstanceType = StructInstanceType(self)

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

    def callable(self) -> bool:
        return True

    def call_with_suggestion(self) -> list[Type | None] | None:
        return [t for _, t in self.fields]

    def callable_with(self, param_types: list[Type]) -> bool:
        return len(self.fields) == len(param_types) and all(f[1].contains(t) for f, t in zip(self.fields, param_types))

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        value = Value(self._instance_type, ctx.tmp(), False)

        for (field, _), param in zip(self.fields, params):
            value.getattr_req(ctx, False, field).assign(ctx, param)

        return value


@dataclass(slots=True)
class StructInstanceType(Type):
    base: StructBaseType

    def __post_init__(self):
        self.fields: dict[str, Type] = {field: type_ for field, type_ in self.base.fields}

    def __str__(self):
        return f"{self.base.name if self.base.name else '?'}"

    def __eq__(self, other):
        return isinstance(other, StructInstanceType) and self.base == other.base

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        for field, _ in self.base.fields:
            value.getattr_req(ctx, False, field).assign(ctx, other.getattr_req(ctx, False, field))

    def assign_default(self, ctx: CompilationContext, value: Value):
        for field, _ in self.base.fields:
            value.getattr_req(ctx, False, field).assign_default(ctx)

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

    def getattr(self, ctx: CompilationContext, value: Value, static: bool, name: str) -> Value | None:
        if static:
            if (val := self.variables.get(name)) is not None:
                return val

        return super(NamespaceType, self).getattr(ctx, value, static, name)


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

    def callable(self) -> bool:
        return True

    def call_with_suggestion(self) -> list[Type | None] | None:
        return [NumberType()]

    def callable_with(self, param_types: list[Type]) -> bool:
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


@dataclass(slots=True)
class BuiltinEnumInstanceType(Type):
    base: BuiltinEnumBaseType

    def __str__(self):
        return self.base.name

    def __eq__(self, other):
        return isinstance(other, BuiltinEnumInstanceType) and self.base == other.base

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        if not self.base.copyable:
            ctx.error(f"Enum of type '{self.base.name}' is not copyable")

        super(BuiltinEnumInstanceType, self).assign(ctx, value, other)

    def bottom_scope(self) -> dict[str, Value] | None:
        return self.base.bottom_scope()
