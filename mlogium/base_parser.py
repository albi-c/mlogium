from .tokens import TokenType, Token
from .error import ParserError
from .util import Position
from abc import ABC, abstractmethod


class BaseParser[T](ABC):
    i: int
    tokens: list[Token]

    def __init__(self, tokens: list[Token]):
        self.i = -1
        self.tokens = tokens

    def _current_pos(self) -> Position:
        if len(self.tokens) == 0 or self.i < 0:
            return Position(0, 0, 0, "", "")

        return self.tokens[self.i].pos

    def has(self, n: int = 1) -> bool:
        assert n >= 1

        return self.i < len(self.tokens) - n

    @staticmethod
    def _match_token(tok: Token, type_: TokenType, value: str | tuple[str, ...] = None) -> bool:
        if type_ is not None and tok.type not in type_:
            return False
        if (isinstance(value, str) and tok.value != value) or (isinstance(value, tuple) and tok.value not in value):
            return False

        return True

    def current(self) -> Token | None:
        if 0 <= self.i < len(self.tokens):
            return self.tokens[self.i]
        return None

    def next(self, type_: TokenType = None, value: str | tuple[str, ...] = None) -> Token:
        if not self.has():
            ParserError.unexpected_eof(self._current_pos(), type_.name if type_ is not None else None)

        self.i += 1
        tok = self.tokens[self.i]
        if not self._match_token(tok, type_, value):
            ParserError.unexpected_token(tok, type_.name if type_ is not None else None)

        return tok

    def lookahead(self, type_: TokenType = None, value: str | tuple[str, ...] = None,
                  n: int = 1, take_if_matches: bool = True) -> Token | None:
        assert n >= 1
        assert n == 1 or not take_if_matches or (type_ is None and value is None)

        if not self.has(n):
            return None

        tok = self.tokens[self.i + n]
        if not self._match_token(tok, type_, value):
            return None

        if take_if_matches and (type_ is not None or value is not None):
            self.i += 1

        return tok

    def skip(self, n: int = 1):
        assert n >= 1

        if not self.has(n):
            ParserError.unexpected_eof(self._current_pos())

        self.i += n

    @abstractmethod
    def parse(self) -> T:
        raise NotImplementedError
