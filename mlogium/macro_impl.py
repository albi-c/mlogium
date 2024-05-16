import os

from .macro import *
from .parser import Parser
from .error import PositionedException
from .compiler import Compiler
from .value import Value
from .value_types import Type


class Macro(BaseMacro, ABC):
    pass


class CastMacro(Macro):
    def __init__(self):
        super().__init__("cast")

    def inputs(self) -> tuple[BaseMacro.Input, ...]:
        return MacroInput.TYPE, MacroInput.VALUE_NODE

    def invoke(self, ctx: MacroInvocationContext, compiler: Compiler, params: list) -> Value:
        return compiler.visit(params[1]).into_req(ctx.ctx, params[0])


class ImportMacro(Macro):
    IMPORTS: set[str] = set()

    def __init__(self):
        super().__init__("import")

    def inputs(self) -> tuple[BaseMacro.Input, ...]:
        return (MacroInput.TOKEN,)

    def top_level_only(self) -> bool:
        return True

    def invoke(self, ctx: MacroInvocationContext, compiler: Compiler, params: list) -> Value:
        path = params[0]
        assert isinstance(path, Token)
        if path.type != TokenType.STRING:
            PositionedException.custom(path.pos, "Import macro requires a string")
        path = path.value
        if ctx.pos.file not in ("<main>", "<clip>"):
            search_dir = os.path.dirname(os.path.abspath(ctx.pos.file))
            path = os.path.join(search_dir, path)
        if not os.path.isfile(path):
            PositionedException.custom(ctx.pos, f"Can't import file '{path}'")

        if path in self.IMPORTS:
            PositionedException.custom(ctx.pos, f"Circular imports are not allowed: '{path}'")
        self.IMPORTS.add(path)

        code = open(path).read()
        tokens = Lexer().lex(code, path)
        node = Parser(tokens, ctx.registry).parse_block(False, True)

        self.IMPORTS.remove(path)

        return compiler.visit(node)


class RepeatMacro(Macro):
    def __init__(self):
        super().__init__("repeat")

    def inputs(self) -> tuple[BaseMacro.Input, ...]:
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

    def inputs(self) -> tuple[BaseMacro.Input, ...]:
        return MacroInput.VALUE_NODE, MacroInput.VALUE_NODE

    def invoke(self, ctx: MacroInvocationContext, compiler: Compiler, params: list) -> Value:
        func = compiler.visit(params[0])
        if not func.callable():
            PositionedException.custom(params[0].pos, "Map macro requires a function")
        if len(func.params()) != 1:
            PositionedException.custom(params[0].pos, "Map macro requires a function that takes 1 parameter")

        tup = compiler.visit(params[1])
        if (values := tup.unpack(ctx.ctx)) is None:
            PositionedException.custom(params[1].pos, "Map macro requires an unpackable object")

        param = func.params()[0]
        results = [func.call(ctx.ctx, [val.into_req(ctx.ctx, param)]) for val in values]

        return Value.tuple(ctx.ctx, results)


class ZipMacro(Macro):
    def __init__(self):
        super().__init__("zip")

    def inputs(self) -> tuple[BaseMacro.Input, ...]:
        return MacroInput.VALUE_NODE, RepeatMacroInput(MacroInput.VALUE_NODE)

    def invoke(self, ctx: MacroInvocationContext, compiler: Compiler, params: list) -> Value:
        unpacked = []
        for param in params:
            tup = compiler.visit(param)
            if (unp := tup.unpack(ctx.ctx)) is None:
                PositionedException.custom(param.pos, "Zip macro requires unpackable objects")
            if len(unpacked) > 0:
                if len(unp) != len(unpacked[0]):
                    PositionedException.custom(param.pos, "Zip macro requires all parameters to have the same length")
            unpacked.append(unp)

        return Value.tuple(ctx.ctx, [Value.tuple(ctx.ctx, list(tup)) for tup in zip(*unpacked)])


class UnpackableOperatorMacro(Macro, ABC):
    def inputs(self) -> tuple[BaseMacro.Input, ...]:
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
            return Value.number(0)

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
            return Value.number(0)

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


MACROS: list[Macro] = [CastMacro(), ImportMacro(), RepeatMacro(), MapMacro(), ZipMacro(), AllMacro(), AnyMacro(),
                       LenMacro(), SumMacro(), ProdMacro()]
