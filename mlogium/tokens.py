import enum
from dataclasses import dataclass

from .util import Position


class TokenType(enum.Flag):
    ID = enum.auto()

    KW_LET = enum.auto()
    KW_CONST = enum.auto()
    KW_FN = enum.auto()
    KW_RETURN = enum.auto()
    KW_IF = enum.auto()
    KW_ELSE = enum.auto()
    KW_WHILE = enum.auto()
    KW_FOR = enum.auto()
    KW_BREAK = enum.auto()
    KW_CONTINUE = enum.auto()
    KW_IN = enum.auto()
    KW_STRUCT = enum.auto()
    # KW_MATCH = enum.auto()
    KW_ENUM = enum.auto()
    KW_STATIC = enum.auto()
    KW_SCOPE = enum.auto()
    KW_AS = enum.auto()

    STRING = enum.auto()
    NUMBER = enum.auto()
    COLOR = enum.auto()

    OPERATOR = enum.auto()
    ASSIGNMENT = enum.auto()

    ARROW = enum.auto()

    SEMICOLON = enum.auto()
    COLON = enum.auto()
    DOUBLE_COLON = enum.auto()
    DOT = enum.auto()
    DOUBLE_DOT = enum.auto()
    COMMA = enum.auto()
    HASH = enum.auto()
    ELLIPSIS = enum.auto()
    QUESTION = enum.auto()

    LPAREN = enum.auto()
    RPAREN = enum.auto()
    LBRACE = enum.auto()
    RBRACE = enum.auto()
    LBRACK = enum.auto()
    RBRACK = enum.auto()


@dataclass(repr=False)
class Token:
    type: TokenType
    value: str
    pos: Position

    def __repr__(self):
        return f"Token({self.type}, \"{self.value}\", {self.pos})"

    def __str__(self):
        return f"[{self.type}, \"{self.value}\"]"

    def __bool__(self):
        return True
