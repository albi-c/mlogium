from .util import Position
from .tokens import Token


class NonPositionedException(Exception):
    msg: str

    def __init__(self, msg: str):
        super().__init__(msg)

        self.msg = msg

    @staticmethod
    def custom(msg: str):
        raise NonPositionedException(msg)


class LinkerError(NonPositionedException):
    pass


class PositionedException(Exception):
    msg: str
    pos: Position

    def __init__(self, msg: str, pos: Position):
        super().__init__(msg, pos)

        self.msg = msg
        self.pos = pos

    @staticmethod
    def custom(pos: Position, msg: str):
        raise PositionedException(msg, pos)


class LexerError(PositionedException):
    @staticmethod
    def unexpected_character(pos: Position, ch: str):
        raise LexerError(f"Unexpected character: '{ch}'", pos)

    @staticmethod
    def unexpected_eof(pos: Position):
        raise LexerError("Unexpected EOF", pos)


class ParserError(PositionedException):
    @staticmethod
    def custom(pos: Position, msg: str):
        raise ParserError(msg, pos)

    @staticmethod
    def unexpected_token(tok: Token, expected: str = None):
        if expected is not None:
            raise ParserError(f"Unexpected token: '{tok.value}' (expected {expected})", tok.pos)
        else:
            raise ParserError(f"Unexpected token: '{tok.value}'", tok.pos)

    @staticmethod
    def unexpected_eof(pos: Position, expected: str = None):
        if expected is not None:
            raise ParserError(f"Unexpected EOF (expected {expected})", pos)
        else:
            raise ParserError("Unexpected EOF", pos)


class CompilerError(PositionedException):
    @staticmethod
    def custom(pos: Position, msg: str):
        raise CompilerError(msg, pos)


class InterpreterError(PositionedException):
    @staticmethod
    def custom(pos: Position, msg: str):
        raise CompilerError(msg, pos)
