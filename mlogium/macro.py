from __future__ import annotations

from .tokens import Token, TokenType
from .base_parser import BaseParser
from .error import ParserError

from typing import Callable


class MacroRegistry:
    macros: dict[str, tuple[list[str], list[Token]]]

    TMP_INDEX: int = 0

    def __init__(self):
        self.macros = {}

    @staticmethod
    def _parse_comma_separated[T](self: BaseParser, func: Callable[[], T], start: TokenType | None = TokenType.LPAREN,
                                  end: TokenType = TokenType.RPAREN, end_val: str = None) -> list[T]:
        if start is not None:
            self.next(start)

        values = []

        prev = TokenType.LPAREN
        while self.has():
            tok = self.lookahead()
            if tok.type == TokenType.COMMA:
                self.next()
                if prev != TokenType.ID:
                    ParserError.unexpected_token(tok)
                prev = TokenType.COMMA
            elif tok.type == end and (end_val is None or tok.value == end_val):
                self.next()
                if prev == TokenType.COMMA:
                    ParserError.unexpected_token(tok, "value")
                break
            else:
                if prev == TokenType.ID:
                    ParserError.unexpected_token(tok, "comma or closing parenthesis")
                values.append(func())
                prev = TokenType.ID
        return values

    @staticmethod
    def _parse_until(self: BaseParser, type_: TokenType, take_end: bool, append_end: bool = False) -> list[Token]:
        tokens = []
        while True:
            if end := self.lookahead(type_, take_if_matches=take_end):
                if append_end:
                    tokens.append(end)
                return tokens
            tokens.append(self.next())

    @classmethod
    def _parse_macro_param(cls, self: BaseParser) -> list[Token]:
        if self.lookahead(TokenType.LBRACE, take_if_matches=False):
            return cls._parse_until(self, TokenType.RBRACE, True, True)
        return cls._parse_until(self, TokenType.COMMA | TokenType.RPAREN, False)

    def create(self, parser: BaseParser, name: Token):
        if name.value in self.macros:
            ParserError.custom(name.pos, f"Macro redefinition: '{name.value}'")
        if parser.lookahead(TokenType.LPAREN):
            params = self._parse_comma_separated(parser, lambda: parser.next(TokenType.ID).value, None)
        else:
            params = []
        tokens = self._parse_until(parser, TokenType.RBRACE, True, True)
        self.macros[name.value] = (params, tokens)

    def delete(self, name: Token):
        try:
            del self.macros[name.value]
        except KeyError:
            ParserError.custom(name.pos, f"Macro not found: '{name.value}'")

    def has(self, name: str):
        return name in self.macros

    @classmethod
    def _invoke[T](cls, self: BaseParser, name: Token, params: list[str], tokens: list[Token],
                   parse_func: Callable[[list[Token]], T]) -> T:
        if self.lookahead(TokenType.LPAREN):
            replacements = cls._parse_comma_separated(self, lambda: cls._parse_macro_param(self), None)
        else:
            replacements = []
        if len(params) != len(replacements):
            ParserError.custom(
                name.pos,
                f"Macro parameter count mismatch: got {len(replacements)}, expected {len(params)}")
        replacement_map = dict(zip(params, replacements))
        processed_tokens = []
        temp_variables = {}
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if (replacement := replacement_map.get(tok.value)) is not None:
                processed_tokens += replacement
            elif tok.value == "$":
                i += 1
                if i >= len(tokens):
                    ParserError.custom(tok.pos, "Temporary macro value missing index")
                index = tokens[i].value
                if (val := temp_variables.get(index)) is not None:
                    processed_tokens.append(val)
                else:
                    val = Token(TokenType.ID, f"__mac_tmp_{MacroRegistry.TMP_INDEX}", tok.pos)
                    temp_variables[index] = val
                    processed_tokens.append(val)
                    MacroRegistry.TMP_INDEX += 1
            else:
                processed_tokens.append(tok)
            i += 1
        return parse_func(processed_tokens)

    def invoke[T](self, parser: BaseParser, name: Token, parse_func: Callable[[list[Token]], T]) -> T:
        params, tokens = self.macros[name.value]
        return self._invoke(parser, name, params, tokens, parse_func)
