from __future__ import annotations

from .value_types import *
from .compilation_context import CompilationContext
from .instruction import Instruction
from .abi import ABI
from .optimizer import Optimizer


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

    def __repr__(self):
        return f"Value({self.type}, {self.value})"

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

    def into_req(self, ctx: CompilationContext, type_: Type | None) -> Value:
        if (value := self.into(ctx, type_)) is None:
            ctx.error(f"Incompatible types: {type_}, {self.type}")
        return value

    def unpack(self, ctx: CompilationContext) -> list[Value] | None:
        return self.impl.unpack(ctx, self)

    def assignable_type(self) -> Type:
        return self.impl.assignable_type(self)

    def assign(self, ctx: CompilationContext, other: Value):
        if self.const:
            raise TypeError("Assignment to constant")

        if self.type.is_opaque() and not self.assignable_type().contains(other.type):
            ctx.error(f"Cannot assign {other.type} to opaque type")

        assert other is not None

        self.impl.assign(ctx, self, other.into_req(ctx, self.assignable_type()))

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

    def needs_function_ref(self) -> bool:
        return self.impl.needs_function_ref(self)

    def params(self) -> list[Type | None]:
        assert self.callable()
        return self.impl.params_public(self)

    def call(self, ctx: CompilationContext, params: list[Value]) -> Value:
        assert self.callable()
        return self.impl.call(ctx, self, params)

    def getattr(self, ctx: CompilationContext, name: str, static: bool) -> Value | None:
        return self.impl.getattr(ctx, self, name, static)

    def unary_op(self, ctx: CompilationContext, op: str) -> Value | None:
        return self.impl.unary_op(ctx, self, op)

    def binary_op(self, ctx: CompilationContext, op: str, other: Value) -> Value | None:
        return self.impl.binary_op(ctx, self, op, other)

    def binary_op_r(self, ctx: CompilationContext, other: Value, op: str) -> Value | None:
        return self.impl.binary_op_r(ctx, other, op, self)

    def index_at(self, ctx: CompilationContext, index: Value) -> Value | None:
        return self.impl.index_at(ctx, self, index)

    def memcell_length(self, ctx: CompilationContext) -> int:
        return self.impl.memcell_length(ctx, self)

    def memcell_serialize(self, ctx: CompilationContext) -> list[str]:
        return self.impl.memcell_serialize(ctx, self)

    def memcell_deserialize(self, ctx: CompilationContext, values: list[str]):
        assert len(values) == self.memcell_length(ctx)
        return self.impl.memcell_deserialize(ctx, self, values)

    def iterate(self, ctx: CompilationContext) -> ValueIterator | None:
        return self.impl.iterate(ctx, self)

    def to_condition(self, ctx: CompilationContext) -> Value | None:
        return self.impl.to_condition(ctx, self)


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
        if value.type.is_opaque():
            return Type.NULL
        return value.type

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        ctx.emit(Instruction.set(value.value, other.value))

    def assign_default(self, ctx: CompilationContext, value: Value):
        ctx.emit(Instruction.set(value.value, "null"))

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [value.value]

    def callable(self, value: Value) -> bool:
        return False

    def needs_function_ref(self, value: Value) -> bool:
        return True

    def params(self, value: Value) -> list[Type | None]:
        raise NotImplementedError

    def params_public(self, value: Value) -> list[Type | None]:
        return self.params(value)

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        raise NotImplementedError

    def getattr(self, ctx: CompilationContext, value: Value, name: str, static: bool) -> Value | None:
        return None

    def unary_op(self, ctx: CompilationContext, value: Value, op: str) -> Value | None:
        return None

    def binary_op(self, ctx: CompilationContext, value: Value, op: str, other: Value) -> Value | None:
        if op == "=":
            if value.const:
                ctx.error(f"Assignment to constant")
            value.assign(ctx, other)
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

        return other.binary_op_r(ctx, value, op)

    def binary_op_r(self, ctx: CompilationContext, other: Value, op: str, value: Value) -> Value | None:
        return None

    def index_at(self, ctx: CompilationContext, value: Value, index: Value) -> Value | None:
        return None

    def memcell_length(self, ctx: CompilationContext, value: Value) -> int:
        return 1

    def memcell_serialize(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [value.value]

    def memcell_deserialize(self, ctx: CompilationContext, value: Value, values: list[str]):
        value.assign(ctx, Value.variable(values[0], value.assignable_type()))

    def iterate(self, ctx: CompilationContext, value: Value) -> ValueIterator | None:
        return None

    def to_condition(self, ctx: CompilationContext, value: Value) -> Value | None:
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

    def assign_default(self, ctx: CompilationContext, value: Value):
        self.assign(ctx, value, Value.number(0))

    @staticmethod
    def try_precalc_op(a: str, op: str, b: str) -> Value | None:
        try:
            a = float(a)
            if a.is_integer():
                a = int(a)

            try:
                b = float(b)
                if b.is_integer():
                    b = int(b)

                return Value.number(int(Optimizer.PRECALC[op](a, b)))

            except ValueError:
                try:
                    return Value.number(int(Optimizer.PRECALC[op](a, None)))

                except (ArithmeticError, ValueError, TypeError, KeyError):
                    return None

        except (ArithmeticError, ValueError, TypeError, KeyError):
            return None

    def unary_op(self, ctx: CompilationContext, value: Value, op: str) -> Value | None:
        if op in self.UNARY_OPS:
            tmp = Value.variable(ctx.tmp(), Type.NUM)
            params = self.UNARY_OPS[op](tmp.value, value.value)
            if (val := self.try_precalc_op(params[2], params[0], params[3])) is not None:
                return val
            ctx.emit(Instruction.op(*params))
            return tmp

        return super().unary_op(ctx, value, op)

    def binary_op(self, ctx: CompilationContext, value: Value, op: str, other: Value) -> Value | None:
        if op in ("==", "!="):
            operator = "equal" if op == "==" else "notEqual"
            if (val := self.try_precalc_op(value.value, operator, other.value)) is not None:
                return val

        if (other_num := other.into(ctx, Type.NUM)) is not None and op in self.BINARY_OPS:
            tmp = Value.variable(ctx.tmp(), Type.NUM)
            if (val := self.try_precalc_op(value.value, self.BINARY_OPS[op], other_num.value)) is not None:
                return val
            ctx.emit(Instruction.op(self.BINARY_OPS[op], tmp.value, value.value, other_num.value))
            return tmp

        return super().binary_op(ctx, value, op, other)

    def to_condition(self, ctx: CompilationContext, value: Value) -> Value | None:
        return value


class StringTypeImpl(TypeImpl):
    def assign_default(self, ctx: CompilationContext, value: Value):
        self.assign(ctx, value, Value.string("\"\""))

    def to_condition(self, ctx: CompilationContext, value: Value) -> Value | None:
        return value


class BlockTypeImpl(TypeImpl):
    def index_at(self, ctx: CompilationContext, value: Value, index: Value) -> Value | None:
        index = index.into_req(ctx, Type.NUM)
        return Value(BasicType("$CellRef"), value.value, False, impl=IndexReferenceTypeImpl(index.value))

    def to_condition(self, ctx: CompilationContext, value: Value) -> Value | None:
        return value


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

    def params(self, value: Value) -> list[Type | None]:
        type_ = value.type
        assert isinstance(type_, FunctionType)
        return type_.params

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        assert len(self.params(value)) == len(params)
        assert all(type_.contains(val.type) for type_, val in zip(self.params(value), params))

        with ctx.scope.anon_function(ctx):
            for i, (type_, val) in enumerate(zip(self.params(value), params)):
                Value.variable(ABI.function_parameter(i), type_).assign(ctx, val)
            ctx.emit(
                Instruction.prepare_return_address(),
                Instruction.jump_addr(value.value)
            )

            type_ = value.type
            assert isinstance(type_, FunctionType)
            # TODO: fix generic return
            if type_.ret is not None:
                ret = Value.variable(ctx.tmp(), type_.ret)
                ret.assign(ctx, Value.variable(ABI.function_return_value(), type_.ret))
                return ret
            else:
                return Value.null()


class ConcreteFunctionTypeImpl(TypeImpl):
    # def into(self, ctx: CompilationContext, value: Value, type_: Type) -> Value | None:
    #     if isinstance(type_, FunctionType):
    #         val_type = value.type
    #         assert isinstance(val_type, ConcreteFunctionType)
    #         if type_.params == val_type.params and type_.ret == val_type.ret:
    #             return Value(type_, ABI.label_var(ABI.function_label(val_type.name)), True)

    def callable(self, value: Value) -> bool:
        return True

    def needs_function_ref(self, value: Value) -> bool:
        return False

    def params(self, value: Value) -> list[Type | None]:
        type_ = value.type
        assert isinstance(type_, ConcreteFunctionType)
        return type_.params

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        type_ = value.type
        assert isinstance(type_, ConcreteFunctionType)

        assert len(type_.params) == len(params)
        assert all(type_ is None or type_.contains(val.type) for type_, val in zip(type_.params, params))

        with ctx.scope.function_call(ctx, f"{type_.name}:{ctx.tmp_num()}"):
            for i, ((name, val_type, ref), val) in enumerate(zip(type_.named_params, params)):
                if ref:
                    ctx.scope.declare_special(name, val)
                else:
                    ctx.scope.declare(name, val_type if val_type is not None else val.type, False).assign(ctx, val)

            result = ctx.generate_node(type_.attributes["code"])

            return_val = Value.variable(ABI.return_value(ctx.scope.get_function()),
                                        type_.ret if type_.ret is not None else result.type)
            if return_val.type.contains(result.type):
                return_val.assign(ctx, result)
            else:
                return_val.assign_default(ctx)

            ctx.emit(Instruction.label(ABI.function_end(ctx.scope.get_function())))

        return return_val


class LambdaTypeImpl(TypeImpl):
    def callable(self, value: Value) -> bool:
        return True

    def needs_function_ref(self, value: Value) -> bool:
        return False

    def params(self, value: Value) -> list[Type | None]:
        type_ = value.type
        assert isinstance(type_, LambdaType)
        return type_.params

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        type_ = value.type
        assert isinstance(type_, LambdaType)

        assert len(type_.params) == len(params)
        assert all(type_ is None or type_.contains(val.type) for type_, val in zip(type_.params, params))

        with ctx.scope.function_call(ctx, f"_lambda:{ctx.tmp_num()}"):
            for name, (value, ref) in type_.attributes["captures"].items():
                if ref:
                    ctx.scope.declare_special(name, value)
                else:
                    ctx.scope.declare(name, value.type, False).assign(ctx, value)

            for i, ((name, val_type, ref), val) in enumerate(zip(type_.named_params, params)):
                if ref:
                    ctx.scope.declare_special(name, val)
                else:
                    ctx.scope.declare(name, val_type if val_type is not None else val.type, False).assign(ctx, val)

            result = ctx.generate_node(type_.attributes["code"])

            return_val = Value.variable(ABI.return_value(ctx.scope.get_function()),
                                        type_.ret if type_.ret is not None else result.type)
            if return_val.type.contains(result.type):
                return_val.assign(ctx, result)
            else:
                return_val.assign_default(ctx)

            ctx.emit(Instruction.label(ABI.function_end(ctx.scope.get_function())))

        return return_val


class SpecialFunctionTypeImpl(TypeImpl):
    def callable(self, value: Value) -> bool:
        return True

    def params(self, value: Value) -> list[Type | None]:
        type_ = value.type
        assert isinstance(type_, SpecialFunctionType)
        return type_.params

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        type_ = value.type
        assert isinstance(type_, SpecialFunctionType)

        assert len(type_.params) == len(params)
        assert all(type_ is None or type_.contains(val.type) for type_, val in zip(type_.params, params))

        return type_.func(ctx, params)


class IntrinsicFunctionTypeImpl(TypeImpl):
    def callable(self, value: Value) -> bool:
        return True

    def params(self, value: Value) -> list[Type | None]:
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

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        strings = ['"("']
        for i, val in enumerate(value.unpack(ctx)):
            if i > 0:
                strings.append('","')
            strings += val.to_strings(ctx)
        strings.append('")"')
        return strings

    def getattr(self, ctx: CompilationContext, value: Value, name: str, static: bool) -> Value | None:
        type_ = value.type
        assert isinstance(type_, TupleType)

        if not static:
            if name == "reversed":
                return Value.tuple(ctx, value.unpack(ctx)[::-1])

            if name.startswith("_") and name.count("_") == 2:
                _, a, b = name.split("_")
                try:
                    a = int(a)
                    b = int(b)
                except ValueError:
                    return None
                else:
                    return Value.tuple(ctx, value.unpack(ctx)[a:b])

            try:
                index = int(name)
            except ValueError:
                return None
            else:
                if len(type_.types) <= index or index < 0:
                    return None
                return Value.variable(ABI.attribute(value.value, str(index)), type_.types[index], value.const)

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        for a, b in zip(value.unpack(ctx), other.unpack(ctx)):
            a.assign(ctx, b)

    def assign_default(self, ctx: CompilationContext, value: Value):
        for val in self.unpack(ctx, value):
            val.assign_default(ctx)

    def memcell_length(self, ctx: CompilationContext, value: Value) -> int:
        return sum(val.memcell_length(ctx) for val in value.unpack(ctx))

    def memcell_serialize(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [v for val in value.unpack(ctx) for v in val.memcell_serialize(ctx)]

    def memcell_deserialize(self, ctx: CompilationContext, value: Value, values: list[str]):
        i = 0
        for val in value.unpack(ctx):
            length = val.memcell_length(ctx)
            val.memcell_deserialize(ctx, values[i:i+length])
            i += length

    def unary_op(self, ctx: CompilationContext, value: Value, op: str) -> Value | None:
        results = [v.unary_op(ctx, op) for v in value.unpack(ctx)]
        if None in results:
            return None
        return Value.tuple(ctx, results)

    def binary_op(self, ctx: CompilationContext, value: Value, op: str, other: Value) -> Value | None:
        this_values = value.unpack(ctx)

        if (values := other.unpack(ctx)) is None:
            results = [v.binary_op(ctx, op, other) for v in this_values]
            if None in results:
                return None
            return Value.tuple(ctx, results)

        if op == "++":
            return Value.tuple(ctx, this_values + values)

        if len(this_values) != len(values):
            ctx.error(f"Tuple length mismatch: {len(this_values)} is not equal to {len(values)}")

        results = [a.binary_op(ctx, op, b) for a, b in zip(this_values, values)]
        if None in results:
            return None
        return Value.tuple(ctx, results)

    def binary_op_r(self, ctx: CompilationContext, other: Value, op: str, value: Value) -> Value | None:
        this_values = value.unpack(ctx)

        if (values := other.unpack(ctx)) is None:
            results = [other.binary_op(ctx, op, v) for v in this_values]
            if None in results:
                return None
            return Value.tuple(ctx, results)

        if op == "++":
            return Value.tuple(ctx, values + this_values)

        if len(this_values) != len(values):
            ctx.error(f"Tuple length mismatch: {len(this_values)} is not equal to {len(values)}")

        results = [a.binary_op(ctx, op, b) for a, b in zip(values, this_values)]
        if None in results:
            return None
        return Value.tuple(ctx, results)


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


class CustomEnumBaseTypeImpl(TypeImpl):
    name: str
    values: dict[str, int]

    def __init__(self, name: str, values: dict[str, int]):
        self.name = name
        self.values = values

    def getattr(self, ctx: CompilationContext, value: Value, name: str, static: bool) -> Value | None:
        if static:
            if name == "_len":
                return Value.number(len(self.values))

            elif name in self.values:
                return Value(BasicType(self.name), str(self.values[name]))

    def callable(self, value: Value) -> bool:
        return True

    def params(self, value: Value) -> list[Type | None]:
        return [Type.NUM]

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        assert len(params) == 1
        assert Type.NUM.contains(params[0].type)
        return Value(BasicType(self.name), params[0].value)


class CustomEnumInstanceTypeImpl(TypeImpl):
    def into(self, ctx: CompilationContext, value: Value, type_: Type) -> Value | None:
        if type_ == Type.NUM:
            return Value.number(int(value.value))


class ExternBlockTypeImpl(TypeImpl):
    def getattr(self, ctx: CompilationContext, value: Value, name: str, static: bool) -> Value | None:
        if static:
            return Value.variable(name, Type.BLOCK, True)


class IndexReferenceTypeImpl(TypeImpl):
    index: str

    def __init__(self, index: str):
        self.index = index

    def into(self, ctx: CompilationContext, value: Value, type_: Type) -> Value | None:
        val = Value.variable(ctx.tmp(), type_)
        values = []
        for _ in range(val.memcell_length(ctx)):
            v = ctx.tmp()
            new_index = ctx.tmp()
            ctx.emit(
                Instruction.read(v, value.value, self.index),
                Instruction.op("add", new_index, self.index, "1")
            )
            self.index = new_index
            values.append(v)
        val.memcell_deserialize(ctx, values)
        return val

    def assignable_type(self, value: Value) -> Type:
        return AnyType()

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        if other.type == BasicType("$CellRef"):
            other = other.into(ctx, Type.NUM)

        values = other.memcell_serialize(ctx)
        for val in values:
            new_index = ctx.tmp()
            ctx.emit(
                Instruction.write(val, value.value, self.index),
                Instruction.op("add", new_index, self.index, "1")
            )
            self.index = new_index


class StructBaseTypeImpl(TypeImpl):
    name: str
    fields: list[tuple[str, Type]]
    field_types: list[Type]
    methods: dict[str, tuple[bool, Value]]
    static_values: dict[str, Value]
    parent: StructBaseTypeImpl | None
    parents: list[StructBaseTypeImpl]
    instance_impl: StructInstanceTypeImpl

    def __init__(self, name: str, fields: list[tuple[str, Type]], methods: dict[str, tuple[bool, Value]],
                 static_values: dict[str, Value], parent: StructBaseTypeImpl | None):
        self.name = name
        self.fields = fields
        self.methods = methods
        self.static_values = static_values
        self.parent = parent
        self.parents = []
        self._get_parents(self.parents)

        self.rebuild()

    def _get_parents(self, parents: list[StructBaseTypeImpl]):
        if self.parent is not None:
            parents.append(self.parent)
            self.parent._get_parents(parents)

    def rebuild(self):
        self.field_types = [t for _, t in self.fields]
        self.instance_impl = StructInstanceTypeImpl(self, self.fields, self.methods, self.static_values,
                                                    [f for f, _ in self.fields], self.parents)
        TypeImplRegistry.add_basic_type_impl(self.name, self.instance_impl)

    def getattr(self, ctx: CompilationContext, value: Value, name: str, static: bool) -> Value | None:
        if static:
            return self.static_values.get(name)

    def callable(self, value: Value = None) -> bool:
        return True

    def params(self, value: Value = None) -> list[Type | None]:
        return self.field_types

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        assert len(self.params()) == len(params)
        assert all(type_.contains(val.type) for type_, val in zip(self.params(), params))

        value = Value.variable(ctx.tmp(), BasicType(self.name))
        for (name, _), val in zip(self.fields, params):
            attr = value.getattr(ctx, name, False)
            assert attr is not None
            attr.assign(ctx, val)

        return value

    def register_type(self):
        TypeImplRegistry.add_basic_type_impl(f"$StructBase_{self.name}", self)
        TypeImplRegistry.add_basic_type_impl(self.name, self.instance_impl)


class StructInstanceTypeImpl(TypeImpl):
    base: StructBaseTypeImpl
    fields: dict[str, Type]
    methods: dict[str, tuple[bool, Value]]
    static_values: dict[str, Value]
    field_list: list[str]
    parents: dict[str, StructInstanceTypeImpl]

    def __init__(self, base: StructBaseTypeImpl, fields: list[tuple[str, Type]], methods: dict[str, tuple[bool, Value]],
                 static_values: dict[str, Value], field_list: list[str], parents: list[StructBaseTypeImpl]):
        self.base = base
        self.fields = {k: v for k, v in fields}
        self.methods = methods
        self.static_values = static_values
        self.field_list = field_list
        self.parents = {impl.name: impl.instance_impl for impl in parents}

    def into(self, ctx: CompilationContext, value: Value, type_: Type) -> Value | None:
        if isinstance(type_, BasicType):
            if type_.name in self.parents:
                return Value(type_, value.value, value.const, const_on_write=value.const_on_write)

    def callable(self, value: Value) -> bool:
        return self.base.callable(value)

    def params(self, value: Value) -> list[Type | None]:
        return self.base.params(value)

    def params_public(self, value: Value) -> list[Type | None]:
        return self.base.params_public(value)

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        return self.base.call(ctx, value, params)

    def memcell_length(self, ctx: CompilationContext, value: Value) -> int:
        return sum(value.getattr(ctx, name, False).memcell_length(ctx) for name in self.field_list)

    def memcell_serialize(self, ctx: CompilationContext, value: Value) -> list[str]:
        return [val for name in self.field_list for val in value.getattr(ctx, name, False).memcell_serialize(ctx)]

    def memcell_deserialize(self, ctx: CompilationContext, value: Value, values: list[str]):
        i = 0
        for name in self.field_list:
            val = value.getattr(ctx, name, False)
            length = val.memcell_length(ctx)
            val.memcell_deserialize(ctx, values[i:i + length])
            i += length

    def to_strings(self, ctx: CompilationContext, value: Value) -> list[str]:
        strings = ['"{"']
        for i, name in enumerate(self.field_list):
            val = value.getattr(ctx, name, False)
            if i > 0:
                strings.append(f'", {name}: "')
            else:
                strings.append(f'"{name}: "')
            strings += val.to_strings(ctx)
        strings.append('"}"')
        return strings

    def getattr(self, ctx: CompilationContext, value: Value, name: str, static: bool) -> Value | None:
        if static:
            return self.static_values.get(name)

        else:
            if name in self.fields:
                return Value(self.fields[name], ABI.attribute(value.value, name), value.const)

            else:
                method = self.methods.get(name)
                if method is None:
                    return None

                if value.const and not method[0]:
                    ctx.error(f"Cannot call method '{name}' on const object")

                val = method[1]
                return Value(val.type, val.value, method[0], impl=StructMethodTypeImpl(value, method[0]))

    def assign(self, ctx: CompilationContext, value: Value, other: Value):
        for field in self.fields.keys():
            value.getattr(ctx, field, False).assign(ctx, other.getattr(ctx, field, False))

    def unary_op(self, ctx: CompilationContext, value: Value, op: str) -> Value | None:
        if op == "...":
            values = [value.getattr(ctx, name, False) for name in self.field_list]
            if None not in values:
                return Value.tuple(ctx, values)

        return super().unary_op(ctx, value, op)


class StructMethodTypeImpl(ConcreteFunctionTypeImpl):
    instance: Value
    const: bool

    def __init__(self, instance: Value, const: bool):
        self.instance = instance
        self.const = const

    def params(self, value: Value) -> list[Type | None]:
        return super().params(value)

    def params_public(self, value: Value) -> list[Type | None]:
        return super().params_public(value)[1:]

    def call(self, ctx: CompilationContext, value: Value, params: list[Value]) -> Value:
        return super().call(ctx, value, [self.instance] + params)

    def into(self, ctx: CompilationContext, value: Value, type_: Type) -> Value | None:
        return None


class TypeInfoTypeImpl(TypeImpl):
    value: Value

    def __init__(self, value: Value):
        self.value = value

    def into(self, ctx: CompilationContext, value: Value, type_: Type) -> Value | None:
        return None

    def _can_call(self, ctx: CompilationContext, params: list[Value]) -> Value:
        if not self.value.callable():
            ctx.error(f"Value of type '{self.value.type}' is not callable")

        if (values := params[0].unpack(ctx)) is None:
            ctx.error(f"Value of type '{params[0].type}' is not unpackable")

        if len(self.value.params()) != len(values):
            return Value.number(0)

        return Value.number(int(
            all(type_ is None or type_.contains(val.type) for type_, val in zip(self.value.params(), values))))

    def getattr(self, ctx: CompilationContext, value: Value, name: str, static: bool) -> Value | None:
        if not static:
            if name == "callable":
                return Value.number(int(self.value.callable()))

            elif name == "size":
                return Value.number(self.value.memcell_length(ctx))

            elif name == "unpackable":
                return Value.number(int(self.value.unpack(ctx) is not None))

            elif name == "same":
                return Value(
                    SpecialFunctionType(
                        [BasicType("$TypeInfo")],
                        lambda _, params: Value.number(int(self.value.type.contains(params[0].impl.value.type)))
                    ), "null")

            elif name == "same_as":
                return Value(
                    SpecialFunctionType(
                        [None],
                        lambda _, params: Value.number(int(self.value.type.contains(params[0].type)))
                    ), "null")

            elif name == "default":
                val = Value.variable(ctx.tmp(), self.value.type)
                val.assign_default(ctx)
                return val

            elif name == "can_call":
                return Value(SpecialFunctionType([None], self._can_call), "null")


class RangeTypeImpl(TypeImpl):
    def getattr(self, ctx: CompilationContext, value: Value, name: str, static: bool) -> Value | None:
        if not static:
            if name == "start":
                return Value.variable(ABI.attribute(value.value, "start"), Type.NUM, value.const)
            elif name == "end":
                return Value.variable(ABI.attribute(value.value, "end"), Type.NUM, value.const)

    def iterate(self, ctx: CompilationContext, value: Value) -> ValueIterator | None:
        return RangeValueIterator(ctx, value.getattr(ctx, "start", False), value.getattr(ctx, "end", False))


class ValueIterator:
    @abstractmethod
    def has_value(self, ctx: CompilationContext) -> Value:
        raise NotImplementedError

    @abstractmethod
    def get_current(self, ctx: CompilationContext) -> Value:
        raise NotImplementedError

    @abstractmethod
    def to_next(self, ctx: CompilationContext):
        raise NotImplementedError


class RangeValueIterator(ValueIterator):
    index: Value
    end_index: Value

    def __init__(self, ctx: CompilationContext, start_index: Value, end_index: Value):
        self.index = Value.variable(ctx.tmp(), Type.NUM)
        self.index.assign(ctx, start_index.into_req(ctx, self.index.assignable_type()))
        self.end_index = end_index

    def has_value(self, ctx: CompilationContext) -> Value:
        return self.index.binary_op(ctx, "<", self.end_index)

    def get_current(self, ctx: CompilationContext) -> Value:
        return self.index

    def to_next(self, ctx: CompilationContext):
        ctx.emit(Instruction.op("add", self.index.value, self.index.value, "1"))


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
    LambdaType: LambdaTypeImpl(),
    SpecialFunctionType: SpecialFunctionTypeImpl(),
    IntrinsicFunctionType: IntrinsicFunctionTypeImpl(),
    IntrinsicSubcommandFunctionType: IntrinsicSubcommandFunctionTypeImpl(),
    TupleType: TupleTypeImpl(),
    NullType: TypeImpl()
})

TypeImplRegistry.add_default_basic_type_impls({
    Type.NUM.name: NumberTypeImpl(),
    Type.STR.name: StringTypeImpl(),
    Type.BLOCK.name: BlockTypeImpl(),
    Type.RANGE.name: RangeTypeImpl()
})
