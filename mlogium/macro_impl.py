from .macro import *
from .parser import Parser


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

    def invoke(self, ctx: MacroInvocationContext, params: list) -> Node:
        return self._parse(ctx, self.invoke_to_tokens(ctx, params))


class PassthroughMacro(Macro):
    def __init__(self):
        super().__init__("passthrough")

    def inputs(self) -> tuple[MacroInput, ...]:
        return (MacroInput.TOKEN,)

    def invoke_to_str(self, ctx: MacroInvocationContext, params: list) -> str:
        return f"""\
{params[0].value}"""


class CastMacro(Macro):
    def __init__(self):
        super().__init__("cast")

    def inputs(self) -> tuple[MacroInput, ...]:
        return MacroInput.TYPE, MacroInput.VALUE_NODE

    def invoke(self, ctx: MacroInvocationContext, params: list) -> Node:
        return BlockNode(ctx.pos, [
            DeclarationNode(ctx.pos, False, SingleAssignmentTarget("__tmp", params[0]), params[1]),
            VariableValueNode(ctx.pos, "__tmp")
        ], True)


class TypenameMacro(Macro):
    def __init__(self):
        super().__init__("typename")

    def inputs(self) -> tuple[MacroInput, ...]:
        return (MacroInput.TYPE,)

    def invoke(self, ctx: MacroInvocationContext, params: list) -> Node:
        return StringValueNode(ctx.pos, str(params[0]))


MACROS: list[Macro] = [PassthroughMacro(), CastMacro(), TypenameMacro()]
