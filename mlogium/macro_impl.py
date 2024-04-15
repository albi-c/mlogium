import os

from .macro import *
from .parser import Parser
from .error import PositionedException


class Macro(BaseMacro, ABC):
    block_output: bool

    def __init__(self, name: str, block_output: bool = False):
        super().__init__(name)

        self.block_output = block_output

    @staticmethod
    def _lex(_: MacroInvocationContext, code: str) -> list[Token]:
        return Lexer().lex(code, "")

    def _parse(self, ctx: MacroInvocationContext, code: list[Token]) -> Node:
        parser = Parser(code, ctx.registry)
        if self.block_output:
            return parser.parse_block(False)
        else:
            return parser.parse_value()

    def invoke_to_str(self, ctx: MacroInvocationContext, params: list) -> str:
        raise NotImplementedError

    def invoke_to_tokens(self, ctx: MacroInvocationContext, params: list) -> list[Token]:
        return self._lex(ctx, self.invoke_to_str(ctx, params))

    def invoke(self, ctx: MacroInvocationContext, params: list) -> Node | Type:
        return self._parse(ctx, self.invoke_to_tokens(ctx, params))


class CastMacro(Macro):
    def __init__(self):
        super().__init__("cast")

    def inputs(self) -> tuple[MacroInput, ...]:
        return MacroInput.TYPE, MacroInput.VALUE_NODE

    def invoke(self, ctx: MacroInvocationContext, params: list) -> Node | Type:
        return BlockNode(ctx.pos, [
            DeclarationNode(ctx.pos, False, SingleAssignmentTarget("__tmp", params[0]), params[1]),
            VariableValueNode(ctx.pos, "__tmp")
        ], True)


class ImportMacro(Macro):
    IMPORTS: set[str] = set()

    def __init__(self):
        super().__init__("import")

    def inputs(self) -> tuple[MacroInput, ...]:
        return (MacroInput.TOKEN,)

    def top_level_only(self) -> bool:
        return True

    def invoke(self, ctx: MacroInvocationContext, params: list) -> Node | Type:
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

        return node


class RepeatMacro(Macro):
    def __init__(self):
        super().__init__("repeat")

    def inputs(self) -> tuple[MacroInput, ...]:
        return MacroInput.TOKEN, MacroInput.VALUE_NODE

    def invoke(self, ctx: MacroInvocationContext, params: list) -> Node | Type:
        try:
            n = int(params[0].value)
            return TupleValueNode(ctx.pos, [params[1]] * n)
        except ValueError:
            PositionedException.custom(params[0].pos, "Repeat macro requires an integer")


MACROS: list[Macro] = [CastMacro(), ImportMacro(), RepeatMacro()]
