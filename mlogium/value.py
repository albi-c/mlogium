from __future__ import annotations


from .value_types import *
from .compilation_context import CompilationContext
from .instruction import InstructionInstance, Instruction
from .abi import ABI


class Value:
    type: Type
    impl: TypeImpl
    value: str
    const: bool
    const_on_write: bool

    def __init__(self, type_: Type, value: str, const: bool = True, *,
                 const_on_write: bool = False, impl: TypeImpl = None):
        self.type = type_
        self.impl = TypeImplRegistry.get_impl(self.type) if impl is None else impl
        self.value = value
        self.const = const
        self.const_on_write = const_on_write

    def __eq__(self, other):
        if isinstance(other, Value):
            return self.type == other.type and self.value == other.value and self.impl == other.impl

    def __bool__(self):
        return True

    def __str__(self):
        return self.impl.debug_str(self)

    @classmethod
    def null(cls) -> Value:
        return cls(Type.NULL, "null")

    @classmethod
    def number(cls, value: int | float) -> Value:
        return cls(Type.NUM, str(value))

    @classmethod
    def string(cls, value: str) -> Value:
        return cls(Type.STR, value)

    @classmethod
    def variable(cls, name: str, type_: Type, const: bool = False, *, const_on_write: bool = False) -> Value:
        return cls(type_, name, const, const_on_write=const_on_write)

    @classmethod
    def tuple(cls, ctx: CompilationContext, values: list[Value]) -> Value:
        type_ = TupleType([val.type for val in values])
        value = Value.variable(ctx.tmp(), type_)
        for a, b in zip(value.unpack(ctx), values):
            a.assign(ctx, b)
        return value

    def is_null(self) -> bool:
        return NullType().contains(self.type)

    def into(self, ctx: CompilationContext, type_: Type | None) -> Value | None:
        if type_ is None or type_.contains(self.type):
            return self

        return self.impl.into(ctx, self, type_)

    def unpack(self, ctx: CompilationContext) -> list[Value] | None:
        return self.impl.unpack(ctx, self)

    def assignable_type(self) -> Type:
        return self.impl.assignable_type(self)

    def assign(self, ctx: CompilationContext, other: Value):
        if self.const:
            raise TypeError("Assignment to constant")

        if self.type.is_opaque():
            ctx.error(f"Cannot assign to opaque type")

        assert other is not None
        assert self.assignable_type().contains(other.type)

        self.impl.assign(ctx, self, other)

        if self.const_on_write:
            self.const = True

    def assign_default(self, ctx: CompilationContext):
        if self.const:
            raise TypeError("Assignment to constant")

        self.impl.assign_default(ctx, self)

        if self.const_on_write:
            self.const = True

    def to_strings(self, ctx: CompilationContext) -> list[str]:
        return self.impl.to_strings(ctx, self)

    def callable(self) -> bool:
        return self.impl.callable(self)

    def params(self) -> list[Type]:
        assert self.callable()
        return self.impl.params(self)

    def call(self, ctx: CompilationContext, params: list[Value]) -> Value:
        assert self.callable()
        return self.impl.call(ctx, self, params)

    def getattr(self, ctx: CompilationContext, name: str, static: bool) -> Value | None:
        return self.impl.getattr(ctx, self, name, static)

    def unary_op(self, ctx: CompilationContext, op: str) -> Value | None:
        return self.impl.unary_op(ctx, self, op)

    def binary_op(self, ctx: CompilationContext, op: str, other: Value) -> Value | None:
        return self.impl.binary_op(ctx, self, op, other)


class TypeImpl:
    EQUALITY_OPS = {
        "==": "equal",
        "!=": "notEqual",
        "===": "strictEqual"
    }

    def debug_str(self, value: Value) -> str:
        return value.value

    def into(self, ctx: CompilationContext, value: Value, type_: Type) -> Value | None:
        return None

    def unpack(self, ctx: CompilationContext, value: Value) -> list[Value] | None:
        return None

    def assignable_type(self, value: Value) -> Type:
        return value.type

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        ctx.emit(Instruction.set(value.value, other.value))

    def assign_default(self, ctx: CompilationContext, value: Value):
        ctx.emit(Instruction.set(value.value, "null"))

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [value.value]

    def callable(self, value: Value) -> bool:
        return False

    def params(self, value: Value) -> list[Type]:
        raise NotImplementedError

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        raise NotImplementedError

    def getattr(self, ctx: CompilationContext, value: Value, name: str, static: bool) -> Value | None:
        return None

    def unary_op(self, ctx: CompilationContext, value: Value, op: str) -> Value | None:
        return None

    def binary_op(self, ctx: CompilationContext, value: Value, op: str, other: Value) -> Value | None:
        if op == "=":
            if (val := other.into(ctx, value.assignable_type())) is None:
                ctx.error(f"Incompatible types: {value.type}, {other.type}")
            value.assign(ctx, val)
            return value

        elif op in self.EQUALITY_OPS:
            tmp = Value.variable(ctx.tmp(), Type.NUM)
            ctx.emit(Instruction.op(self.EQUALITY_OPS[op], tmp.value, value.value, other.value))
            return tmp

        elif op == "!==":
            return value.binary_op(ctx, "===", other).unary_op(ctx, "!")

        elif op.endswith("="):
            value.assign(ctx, value.binary_op(ctx, op[:-1], other))
            return value

        return None


class NumberTypeImpl(TypeImpl):
    UNARY_OPS = {
        "-": lambda result, value: ("sub", result, "0", value),
        "!": lambda result, value: ("equal", result, value, "0"),
        "~": lambda result, value: ("flip", result, value, "_")
    }

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

    def unary_op(self, ctx: CompilationContext, value: Value, op: str) -> Value | None:
        if op in self.UNARY_OPS:
            tmp = Value.variable(ctx.tmp(), Type.NUM)
            ctx.emit(Instruction.op(*(self.UNARY_OPS[op](tmp.value, value.value))))
            return tmp

        return super().unary_op(ctx, value, op)

    def binary_op(self, ctx: CompilationContext, value: Value, op: str, other: Value) -> Value | None:
        if (other := other.into(ctx, Type.NUM)) is not None and op in self.BINARY_OPS:
            tmp = Value.variable(ctx.tmp(), Type.NUM)
            ctx.emit(Instruction.op(self.BINARY_OPS[op], tmp.value, value.value, other.value))
            return tmp

        return super().binary_op(ctx, value, op, other)


class AnyTypeImpl(TypeImpl):
    def into(self, ctx: CompilationContext, value: Value, type_: Type) -> Value | None:
        return Value.variable(value.value, type_, value.const, const_on_write=value.const_on_write)

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        Value.variable(value.value, other.type).assign(ctx, other)


class UnionTypeImpl(TypeImpl):
    def into(self, ctx: CompilationContext, value: Value, type_: Type) -> Value | None:
        union = value.type
        assert isinstance(union, UnionType)
        if union.contains(type_):
            return Value.variable(value.value, type_, value.const, const_on_write=value.const_on_write)

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        union = value.type
        assert isinstance(union, UnionType)
        if union.contains(value.type):
            Value.variable(value.value, other.type).assign(ctx, other)


class AnonymousFunctionTypeImpl(TypeImpl):
    def callable(self, value: Value) -> bool:
        return True

    def params(self, value: Value) -> list[Type]:
        type_ = value.type
        assert isinstance(type_, FunctionType)
        return type_.params

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        assert len(self.params(value)) == len(params)
        assert all(type_.contains(val.type) for type_, val in zip(self.params(value), params))

        for i, (type_, val) in enumerate(zip(self.params(value), params)):
            Value.variable(ABI.function_parameter(i), type_).assign(ctx, val)
        ctx.emit(
            Instruction.get_instruction_pointer_offset(ABI.function_return_address(), "1"),
            Instruction.jump_addr(value.value)
        )

        type_ = value.type
        assert isinstance(type_, FunctionType)
        if type_.ret is not None:
            ret = Value.variable(ctx.tmp(), type_.ret)
            ret.assign(ctx, Value.variable(ABI.function_return_value(), type_.ret))
            return ret
        else:
            return Value.null()


class ConcreteFunctionTypeImpl(TypeImpl):
    def into(self, ctx: CompilationContext, value: Value, type_: Type) -> Value | None:
        if isinstance(type_, FunctionType):
            val_type = value.type
            assert isinstance(val_type, ConcreteFunctionType)
            if type_.params == val_type.params and type_.ret == val_type.ret:
                return Value(type_, ABI.function_label(val_type.name), True)

    def callable(self, value: Value) -> bool:
        return True

    def params(self, value: Value) -> list[Type]:
        type_ = value.type
        assert isinstance(type_, ConcreteFunctionType)
        return type_.params

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        assert len(self.params(value)) == len(params)
        assert all(type_.contains(val.type) for type_, val in zip(self.params(value), params))

        type_ = value.type
        assert isinstance(type_, ConcreteFunctionType)

        with ctx.scope.function_call(f"{type_.name}:{ctx.tmp_num()}"):
            for (name, val_type), val in zip(type_.named_params, params):
                ctx.scope.declare(name, val_type, False).assign(ctx, val)

            result = ctx.generate_node(type_.attributes["code"])

            return_val = Value.variable(ABI.return_value(ctx.scope.get_function()), type_.ret)
            if return_val.type.contains(result.type):
                return_val.assign(ctx, result)
            else:
                return_val.assign_default(ctx)

            ctx.emit(Instruction.label(ABI.function_end(ctx.scope.get_function())))

        return return_val


class IntrinsicFunctionTypeImpl(TypeImpl):
    def callable(self, value: Value) -> bool:
        return True

    def params(self, value: Value) -> list[Type]:
        type_ = value.type
        assert isinstance(type_, IntrinsicFunctionType)
        return type_.input_types

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        assert len(self.params(value)) == len(params)
        assert all(type_.contains(val.type) for type_, val in zip(self.params(value), params))

        type_ = value.type
        assert isinstance(type_, IntrinsicFunctionType)

        params_all = []
        input_i = 0
        output_vars = []
        for val_type, output in type_.params:
            if output:
                var = Value.variable(ctx.tmp(), val_type)
                params_all.append(var)
                output_vars.append(var)
            else:
                params_all.append(params[input_i])
                input_i += 1

        type_.instruction_func(ctx, *params_all)

        if len(output_vars) == 0:
            return Value.null()
        elif len(output_vars) == 1:
            return output_vars[0]
        else:
            return Value.tuple(ctx, output_vars)


class IntrinsicSubcommandFunctionTypeImpl(TypeImpl):
    def getattr(self, ctx: CompilationContext, value: Value, name: str, static: bool) -> Value | None:
        if not static:
            type_ = value.type
            assert isinstance(type_, IntrinsicSubcommandFunctionType)
            func_type = type_.subcommands.get(name)
            return Value(func_type, f"{type_.name}.{name}")


class TupleTypeImpl(TypeImpl):
    def unpack(self, ctx: CompilationContext, value: Value) -> list[Value] | None:
        type_ = value.type
        assert isinstance(type_, TupleType)
        return [Value.variable(ABI.attribute(value.value, str(i)), t, value.const)
                for i, t in enumerate(type_.types)]

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        for a, b in zip(value.unpack(ctx), other.unpack(ctx)):
            a.assign(ctx, b)

    def assign_default(self, ctx: CompilationContext, value: Value):
        for val in self.unpack(ctx, value):
            val.assign_default(ctx)


class EnumBaseTypeImpl(TypeImpl):
    name: str
    values: set[str]
    has_prefix: bool
    is_opaque: bool

    def __init__(self, name: str, values: set[str], has_prefix: bool, is_opaque: bool):
        self.name = name
        self.values = values
        self.has_prefix = has_prefix
        self.is_opaque = is_opaque

    def getattr(self, ctx: CompilationContext, value: Value, name: str, static: bool) -> Value | None:
        if static:
            if name == "_len":
                return Value.number(len(self.values))

            elif name in self.values:
                return Value.variable(("@" if self.has_prefix else "") + name,
                                      BasicType(("$" if self.is_opaque else "") + self.name), True)


class ExternBlockTypeImpl(TypeImpl):
    def getattr(self, ctx: CompilationContext, value: Value, name: str, static: bool) -> Value | None:
        if static:
            return Value.variable(name, Type.BLOCK, True)


class TypeImplRegistry:
    _default_basic_type_implementations: dict[str, TypeImpl] = {}
    _basic_type_implementations: dict[str, TypeImpl] = {}
    _implementations: dict[type[Type], TypeImpl] = {}

    @classmethod
    def get_impl(cls, type_: Type) -> TypeImpl:
        if isinstance(type_, BasicType):
            if type_.name in cls._default_basic_type_implementations:
                return cls._default_basic_type_implementations[type_.name]
            elif type_.name in cls._basic_type_implementations:
                return cls._basic_type_implementations[type_.name]
        return cls._implementations[type(type_)]

    @classmethod
    def add_impls(cls, impls: dict[type[Type], TypeImpl]):
        cls._implementations |= impls

    @classmethod
    def reset_basic_type_impls(cls):
        cls._basic_type_implementations = {}

    @classmethod
    def add_basic_type_impl(cls, name: str, impl: TypeImpl):
        cls._basic_type_implementations[name] = impl

    @classmethod
    def add_default_basic_type_impls(cls, impls: dict[str, TypeImpl]):
        cls._default_basic_type_implementations |= impls


TypeImplRegistry.add_impls({
    AnyType: AnyTypeImpl(),
    BasicType: TypeImpl(),
    UnionTypeImpl: UnionTypeImpl(),
    FunctionType: AnonymousFunctionTypeImpl(),
    ConcreteFunctionType: ConcreteFunctionTypeImpl(),
    IntrinsicFunctionType: IntrinsicFunctionTypeImpl(),
    IntrinsicSubcommandFunctionType: IntrinsicSubcommandFunctionTypeImpl(),
    TupleType: TupleTypeImpl(),
    NullType: TypeImpl()
})

TypeImplRegistry.add_default_basic_type_impls({
    "num": NumberTypeImpl()
})
