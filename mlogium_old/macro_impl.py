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


MACROS: list[Macro] = [CastMacro(), ImportMacro(), RepeatMacro(), OperatorMacro(), TypeofMacro(), RangeMacro(),
                       GenerateMacro(), StaticAssertMacro(), TypeMacro(), TypeiMacro(), UnrollMacro()]
