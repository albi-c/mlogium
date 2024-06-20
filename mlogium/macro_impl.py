import os

from .macro import *
from .parser import Parser
from .error import PositionedException
from .compiler import Compiler
from .value import Value, TypeInfoTypeImpl
from .node import *


class CastMacro(Macro):
    def __init__(self):
        super().__init__("cast")

    def inputs(self) -> tuple[Macro.Input, ...]:
        return MacroInput.TYPE, MacroInput.VALUE_NODE

    def invoke(self, ctx: MacroInvocationContext, compiler: Compiler, params: list) -> Value:
        return compiler.visit(params[1]).into_req(ctx.ctx, params[0])


class ImportMacro(Macro):
    IMPORTS: set[str] = set()

    def __init__(self):
        super().__init__("import")

    def inputs(self) -> tuple[Macro.Input, ...]:
        return (MacroInput.TOKEN,)

    def top_level_only(self) -> bool:
        return True

    def invoke(self, ctx: MacroInvocationContext, compiler: Compiler, params: list) -> Value:
        path = params[0]
        assert isinstance(path, Token)
        if path.type != TokenType.STRING:
            PositionedException.custom(path.pos, "Import macro requires a string")
        path = path.value
        if path.startswith("std:"):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stdlib", f"{path[4:]}.mu")
            display_path = params[0].value
        else:
            if ctx.pos.file not in ("<main>", "<clip>"):
                search_dir = os.path.dirname(os.path.abspath(ctx.pos.file))
                path = os.path.join(search_dir, path)
            else:
                path = os.path.abspath(path)
            display_path = path
        if not os.path.isfile(path):
            PositionedException.custom(ctx.pos, f"Can't import file '{display_path}'")

        if path in self.IMPORTS:
            PositionedException.custom(ctx.pos, f"Circular imports are not allowed: '{display_path}'")
        self.IMPORTS.add(path)

        code = open(path).read()
        tokens = Lexer().lex(code, path)
        node = Parser(tokens, ctx.registry).parse_block(False, True)

        result = compiler.visit(node)

        self.IMPORTS.remove(path)

        return result


class RepeatMacro(Macro):
    def __init__(self):
        super().__init__("repeat")

    def inputs(self) -> tuple[Macro.Input, ...]:
        return MacroInput.TOKEN, MacroInput.VALUE_NODE

    def invoke(self, ctx: MacroInvocationContext, compiler: Compiler, params: list) -> Value:
        try:
            n = int(params[0].value)
            return Value.tuple(ctx.ctx, [compiler.visit(params[1])] * n)
        except ValueError:
            PositionedException.custom(params[0].pos, "Repeat macro requires an integer")


class MapMacro(Macro):
    def __init__(self):
        super().__init__("map")

    def inputs(self) -> tuple[Macro.Input, ...]:
        return MacroInput.VALUE_NODE, MacroInput.VALUE_NODE

    def invoke(self, ctx: MacroInvocationContext, compiler: Compiler, params: list) -> Value:
        func = compiler.visit(params[0])
        if not func.callable():
            PositionedException.custom(params[0].pos, f"Value of type '{func.type}' is not callable")
        if len(func.params()) != 1:
            PositionedException.custom(params[0].pos, f"Value of type '{func.type}' has to take 1 parameter")

        tup = compiler.visit(params[1])
        if (values := tup.unpack(ctx.ctx)) is None:
            PositionedException.custom(params[1].pos, f"Value of type '{tup.type}' is not unpackable")

        param = func.params()[0]
        results = [func.call(ctx.ctx, [val.into_req(ctx.ctx, param)]) for val in values]

        return Value.tuple(ctx.ctx, results)


class UnpackMapMacro(Macro):
    def __init__(self):
        super().__init__("unpack_map")

    def inputs(self) -> tuple[Macro.Input, ...]:
        return MacroInput.VALUE_NODE, MacroInput.VALUE_NODE

    def invoke(self, ctx: MacroInvocationContext, compiler: Compiler, params: list) -> Value:
        func = compiler.visit(params[0])
        if not func.callable():
            PositionedException.custom(params[0].pos, f"Value of type '{func.type}' is not callable")
        func_params = func.params()

        tup = compiler.visit(params[1])
        if (values := tup.unpack(ctx.ctx)) is None:
            PositionedException.custom(params[1].pos, f"Value of type '{tup.type}' is not unpackable")

        results = []
        for val in values:
            if (unp := val.unpack(ctx.ctx)) is None:
                PositionedException.custom(params[1].pos, f"Value of type '{val.type}' is not unpackable")
            if len(unp) != len(func_params):
                PositionedException.custom(params[1].pos, f"({len(unp)} provided, {len(func_params)} expected)")
            results.append(func.call(ctx.ctx,
                                     [param.into_req(ctx.ctx, type_) for param, type_ in zip(unp, func_params)]))

        return Value.tuple(ctx.ctx, results)


class ZipMacro(Macro):
    def __init__(self):
        super().__init__("zip")

    def inputs(self) -> tuple[Macro.Input, ...]:
        return MacroInput.VALUE_NODE, RepeatMacroInput(MacroInput.VALUE_NODE)

    def invoke(self, ctx: MacroInvocationContext, compiler: Compiler, params: list) -> Value:
        unpacked = []
        for param in params:
            tup = compiler.visit(param)
            if (unp := tup.unpack(ctx.ctx)) is None:
                PositionedException.custom(param.pos, f"Value of type '{tup.type}' is not unpackable")
            if len(unpacked) > 0:
                if len(unp) != len(unpacked[0]):
                    PositionedException.custom(param.pos, "Zip macro requires all parameters to have the same length")
            unpacked.append(unp)

        return Value.tuple(ctx.ctx, [Value.tuple(ctx.ctx, list(tup)) for tup in zip(*unpacked)])


class UnpackableOperatorMacro(Macro, ABC):
    def inputs(self) -> tuple[Macro.Input, ...]:
        return (MacroInput.VALUE_NODE,)

    @abstractmethod
    def process(self, ctx: MacroInvocationContext, compiler: Compiler, values: list[Value]) -> Value:
        raise NotImplementedError

    def invoke(self, ctx: MacroInvocationContext, compiler: Compiler, params: list) -> Value:
        tup = compiler.visit(params[0])
        if (unp := tup.unpack(ctx.ctx)) is None:
            PositionedException.custom(params[0].pos, f"{self.name.capitalize()} macro requires an unpackable object")
        return self.process(ctx, compiler, unp)


class AllMacro(UnpackableOperatorMacro):
    def __init__(self):
        super().__init__("all")

    def process(self, ctx: MacroInvocationContext, compiler: Compiler, values: list[Value]) -> Value:
        if len(values) == 0:
            return Value.boolean(False)

        if (cond := values[0].to_condition(ctx.ctx)) is None:
            PositionedException.custom(ctx.pos, f"Value of type '{values[0].type}' is not usable as a condition")
        for val in values[1:]:
            if (c := val.to_condition(ctx.ctx)) is None:
                PositionedException.custom(ctx.pos, f"Value of type '{val.type}' is not usable as a condition")
            cond = cond.binary_op(ctx.ctx, "&&", c)

        return cond


class AnyMacro(UnpackableOperatorMacro):
    def __init__(self):
        super().__init__("any")

    def process(self, ctx: MacroInvocationContext, compiler: Compiler, values: list[Value]) -> Value:
        if len(values) == 0:
            return Value.boolean(False)

        if (cond := values[0].to_condition(ctx.ctx)) is None:
            PositionedException.custom(ctx.pos, f"Value of type '{values[0].type}' is not usable as a condition")
        for val in values[1:]:
            if (c := val.to_condition(ctx.ctx)) is None:
                PositionedException.custom(ctx.pos, f"Value of type '{val.type}' is not usable as a condition")
            cond = cond.binary_op(ctx.ctx, "||", c)

        return cond


class LenMacro(UnpackableOperatorMacro):
    def __init__(self):
        super().__init__("len")

    def process(self, ctx: MacroInvocationContext, compiler: Compiler, values: list[Value]) -> Value:
        return Value.number(len(values))


class SumMacro(UnpackableOperatorMacro):
    def __init__(self):
        super().__init__("sum")

    def process(self, ctx: MacroInvocationContext, compiler: Compiler, values: list[Value]) -> Value:
        if len(values) == 0:
            return Value.number(0)

        sum_ = values[0].into_req(ctx.ctx, Type.NUM)
        for val in values[1:]:
            sum_ = sum_.binary_op(ctx.ctx, "+", val.into_req(ctx.ctx, Type.NUM))

        return sum_


class ProdMacro(UnpackableOperatorMacro):
    def __init__(self):
        super().__init__("prod")

    def process(self, ctx: MacroInvocationContext, compiler: Compiler, values: list[Value]) -> Value:
        if len(values) == 0:
            return Value.number(1)

        sum_ = values[0].into_req(ctx.ctx, Type.NUM)
        for val in values[1:]:
            sum_ = sum_.binary_op(ctx.ctx, "*", val.into_req(ctx.ctx, Type.NUM))

        return sum_


class OperatorMacro(Macro):
    def __init__(self):
        super().__init__("op")

    def inputs(self) -> tuple[Macro.Input, ...]:
        return (MacroInput.TOKEN,)

    def invoke(self, ctx: MacroInvocationContext, compiler: Compiler, params: list) -> Value:
        if params[0].type != TokenType.OPERATOR:
            PositionedException.custom(params[0].pos, "Op macro requires an operator")
        op = params[0].value

        return compiler.register_function(
            ctx.ctx.tmp(),
            NamedParamFunctionType([("a", BasicType.NUM, False), ("b", BasicType.NUM, False)], BasicType.NUM),
            BinaryOpNode(ctx.pos, VariableValueNode(ctx.pos, "a"), op, VariableValueNode(ctx.pos, "b"))
        )


class TypeofMacro(Macro):
    def __init__(self):
        super().__init__("typeof")

    def is_type(self) -> bool:
        return True

    def inputs(self) -> tuple[Macro.Input, ...]:
        return (MacroInput.VALUE_NODE,)

    def type_invoke(self, ctx: MacroInvocationContext, compiler: Compiler, params: list) -> Type:
        return compiler.visit(params[0]).type


class ForeachMacro(Macro):
    def __init__(self):
        super().__init__("foreach")

    def inputs(self) -> tuple[Macro.Input, ...]:
        return MacroInput.VALUE_NODE, MacroInput.VALUE_NODE

    def invoke(self, ctx: MacroInvocationContext, compiler: Compiler, params: list) -> Value:
        func = compiler.visit(params[0])
        if not func.callable():
            PositionedException.custom(params[0].pos, f"Value of type '{func.type}' is not callable")
        if len(func.params()) != 1:
            PositionedException.custom(params[0].pos, f"Value of type '{func.type}' has to take 1 parameter")

        tup = compiler.visit(params[1])
        if (values := tup.unpack(ctx.ctx)) is None:
            PositionedException.custom(params[1].pos, f"Value of type '{tup.type}' is not unpackable")

        param = func.params()[0]
        for val in values:
            func.call(ctx.ctx, [val.into_req(ctx.ctx, param)])

        return Value.null()


class ReduceMacro(Macro):
    def __init__(self):
        super().__init__("reduce")

    def inputs(self) -> tuple[Macro.Input, ...]:
        return MacroInput.VALUE_NODE, MacroInput.VALUE_NODE, MacroInput.VALUE_NODE

    def invoke(self, ctx: MacroInvocationContext, compiler: Compiler, params: list) -> Value:
        func = compiler.visit(params[0])
        if not func.callable():
            PositionedException.custom(params[0].pos, f"Value of type '{func.type}' is not callable")
        if len(func.params()) != 2:
            PositionedException.custom(params[0].pos, f"Value of type '{func.type}' has to take 1 parameter")

        tup = compiler.visit(params[1])
        if (values := tup.unpack(ctx.ctx)) is None:
            PositionedException.custom(params[1].pos, f"Value of type '{tup.type}' is not unpackable")

        start = compiler.visit(params[2])
        current = Value.variable(ctx.ctx.tmp(), start.type)
        current.assign(ctx.ctx, start)

        for val in values:
            current = func.call(ctx.ctx, [current, val])

        return current


class TakeMacro(Macro):
    def __init__(self):
        super().__init__("take")

    def inputs(self) -> tuple[Macro.Input, ...]:
        return (MacroInput.VALUE_NODE,)

    def invoke(self, ctx: MacroInvocationContext, compiler, params: list) -> Value:
        tup = compiler.visit(params[0])
        if (values := tup.unpack(ctx.ctx)) is None:
            PositionedException.custom(params[0].pos, f"Value of type '{tup.type}' is not unpackable")
        if len(values) == 0:
            PositionedException.custom(params[0].pos, f"Not enough values to unpack")

        return Value.tuple(ctx.ctx, [values[0], Value.tuple(ctx.ctx, values[1:])])


class ReverseMacro(UnpackableOperatorMacro):
    def __init__(self):
        super().__init__("reverse")

    def process(self, ctx: MacroInvocationContext, compiler: Compiler, values: list[Value]) -> Value:
        return Value.tuple(ctx.ctx, values[::-1])


class RangeMacro(Macro):
    def __init__(self):
        super().__init__("range")

    def inputs(self) -> tuple[Macro.Input, ...]:
        return (MacroInput.TOKEN,)

    def invoke(self, ctx: MacroInvocationContext, compiler, params: list) -> Value:
        try:
            n = int(params[0].value)
            return Value.tuple(ctx.ctx, [Value.number(i) for i in range(n)])
        except ValueError:
            PositionedException.custom(params[0].pos, "Range macro requires an integer")


class GenerateMacro(Macro):
    def __init__(self):
        super().__init__("generate")

    def inputs(self) -> tuple[Macro.Input, ...]:
        return MacroInput.TOKEN, MacroInput.VALUE_NODE

    def invoke(self, ctx: MacroInvocationContext, compiler, params: list) -> Value:
        func = compiler.visit(params[1])
        if not func.callable():
            PositionedException.custom(params[1].pos, f"Value of type '{func.type}' is not callable")
        if len(func.params()) != 1:
            PositionedException.custom(params[1].pos, f"Value of type '{func.type}' has to take 1 parameter")

        try:
            n = int(params[0].value)
            return Value.tuple(ctx.ctx, [func.call(ctx.ctx, [Value.number(i)]) for i in range(n)])
        except ValueError:
            PositionedException.custom(params[0].pos, "Generate macro requires an integer")


class StaticAssertMacro(Macro):
    def __init__(self):
        super().__init__("static_assert")

    def inputs(self) -> tuple[Macro.Input, ...]:
        return MacroInput.VALUE_NODE, MacroInput.TOKEN

    def invoke(self, ctx: MacroInvocationContext, compiler, params: list) -> Value:
        if params[1].type != TokenType.STRING:
            PositionedException.custom(params[1].pos, "Static assert macro requires a string")
        msg = params[1].value

        cond_val = compiler.visit(params[0])
        if (cond := cond_val.to_condition(ctx.ctx)) is None:
            PositionedException.custom(params[0].pos,
                                       f"Value of type '{cond_val.type}' is not usable as a condition")

        try:
            num = int(cond.value)
        except ValueError:
            PositionedException.custom(params[0].pos, f"Value '{params[0]}' is not usable as a constant condition")
        else:
            if num == 0:
                PositionedException.custom(ctx.pos, f"Assertion failed: {msg}")

        return Value.null()


class TypeMacro(Macro):
    def __init__(self):
        super().__init__("type")

    def inputs(self) -> tuple[Macro.Input, ...]:
        return (MacroInput.VALUE_NODE,)

    def invoke(self, ctx: MacroInvocationContext, compiler, params: list) -> Value:
        return Value(BasicType("$TypeInfo"), "null", impl=TypeInfoTypeImpl(compiler.visit(params[0])))


class TypeiMacro(Macro):
    def __init__(self):
        super().__init__("typei")

    def inputs(self) -> tuple[Macro.Input, ...]:
        return (MacroInput.TYPE,)

    def invoke(self, ctx: MacroInvocationContext, compiler, params: list) -> Value:
        return Value(BasicType("$TypeInfo"), "null", impl=TypeInfoTypeImpl(Value(params[0], "null")))


class UnrollMacro(Macro):
    def __init__(self):
        super().__init__("unroll")

    def inputs(self) -> tuple[Macro.Input, ...]:
        return MacroInput.TOKEN, MacroInput.VALUE_NODE

    def invoke(self, ctx: MacroInvocationContext, compiler: Compiler, params: list) -> Value:
        func = compiler.visit(params[1])
        if not func.callable_with([Type.NUM]):
            PositionedException.custom(params[1].pos,
                                       f"Value of type '{func.type}' has to take 1 parameter of type 'num'")

        try:
            n = int(params[0].value)
        except ValueError:
            PositionedException.custom(params[0].pos, "Unroll macro requires an integer")
        else:
            for i in range(n):
                func.call(ctx.ctx, [Value.number(i)])

        return Value.null()


class EnumerateMacro(UnpackableOperatorMacro):
    def __init__(self):
        super().__init__("enumerate")

    def process(self, ctx: MacroInvocationContext, compiler: Compiler, values: list[Value]) -> Value:
        return Value.tuple(ctx.ctx, [Value.tuple(ctx.ctx, [Value.number(i), v]) for i, v in enumerate(values)])


MACROS: list[Macro] = [CastMacro(), ImportMacro(), RepeatMacro(), MapMacro(), UnpackMapMacro(), ZipMacro(), AllMacro(),
                       AnyMacro(), LenMacro(), SumMacro(), ProdMacro(), OperatorMacro(), TypeofMacro(), ForeachMacro(),
                       ReduceMacro(), TakeMacro(), ReverseMacro(), RangeMacro(), GenerateMacro(), StaticAssertMacro(),
                       TypeMacro(), TypeiMacro(), UnrollMacro(), EnumerateMacro()]
