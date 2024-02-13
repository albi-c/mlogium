from __future__ import annotations


from .value_types import *
from .compilation_context import CompilationContext
from .instruction import Instruction
from .abi import ABI


class Value:
    type: Type
    impl: TypeImpl
    value: str
    const: bool
    const_on_write: bool

    def __init__(self, type_: Type, value: str, const: bool = True, *,
                 const_on_write: bool = False):
        self.type = type_
        self.impl = TypeImplRegistry.get_impl(type(self.type))
        self.value = value
        self.const = const
        self.const_on_write = const_on_write

    def __eq__(self, other):
        if isinstance(other, Value):
            return self.type == other.type and self.value == other.value and self.impl == other.impl

    def __bool__(self):
        return True

    def __str__(self):
        return self.debug_str()

    @classmethod
    def null(cls) -> Value:
        return cls(NullType(), "null")

    @classmethod
    def number(cls, value: int | float) -> Value:
        return cls(BasicType("num"), str(value))

    @classmethod
    def string(cls, value: str) -> Value:
        return cls(BasicType("str"), value)

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


class TypeImpl:
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

        with ctx.scope.function_call(f"{self.name}:{ctx.tmp_num()}"):
            for (name, type_), val in zip(type_.named_params, params):
                ctx.scope.declare(name, type_, False).assign(ctx, val)

            ctx.generate_node(type_.attributes["code"])

            return_val = Value.variable(ABI.return_value(self.name), type_.ret)
            return_val.assign_default(ctx)

            ctx.emit(Instruction.label(ABI.function_end(type_.name)))

        return return_val


class IntrinsicFunctionTypeImpl(TypeImpl):
    pass


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


class TypeImplRegistry:
    _implementations: dict[type[Type], TypeImpl] = {}

    @classmethod
    def get_impl(cls, type_: type[Type]) -> TypeImpl:
        return cls._implementations[type_]

    @classmethod
    def add_impl(cls, type_: type[Type], impl: TypeImpl):
        cls._implementations[type_] = impl

    @classmethod
    def add_impls(cls, impls: dict[type[Type], TypeImpl]):
        cls._implementations |= impls


TypeImplRegistry.add_impls({
    AnyType: AnyTypeImpl(),
    BasicType: TypeImpl(),
    OpaqueType: TypeImpl(),
    UnionTypeImpl: UnionTypeImpl(),
    FunctionType: AnonymousFunctionTypeImpl(),
    ConcreteFunctionType: ConcreteFunctionTypeImpl(),
    IntrinsicFunctionType: IntrinsicFunctionTypeImpl(),
    TupleType: TupleTypeImpl(),
    NullType: TypeImpl()
})
