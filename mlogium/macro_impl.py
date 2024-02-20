from .macro import *
from .parser import Parser


class Macro(BaseMacro, ABC):
    @staticmethod
    def _lex(_: MacroInvocationContext, code: str) -> list[Token]:
        return Lexer().lex(code, "")

    @staticmethod
    def _parse(ctx: MacroInvocationContext, code: list[Token]) -> Node:
        return Parser(code, ctx.registry).parse()

    def invoke_to_str(self, ctx: MacroInvocationContext, params: list) -> str:
        raise NotImplementedError

    def invoke_to_tokens(self, ctx: MacroInvocationContext, params: list) -> list[Token]:
        return self._lex(ctx, self.invoke_to_str(ctx, params))

    def invoke(self, ctx: MacroInvocationContext, params: list) -> Node:
        return self._parse(ctx, self.invoke_to_tokens(ctx, params))


MACROS: list[Macro] = []
