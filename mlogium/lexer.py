import string
from typing import Callable

from .tokens import Token, TokenType
from .util import Position
from .error import LexerError


class Lexer:
    i: int
    line: int
    p_line: int
    p_char: int
    char: int
    code: str
    code_lines: list[str]
    p_code_lines: list[str]
    filename: str

    CH_INTEGER = string.digits
    CH_ID_START = string.ascii_letters + "_@"
    CH_HEX_DIGIT = string.hexdigits
    CH_ID = CH_ID_START + string.digits
    CH_OP_DOUBLE = "*<>&|+"
    CH_OP = CH_OP_DOUBLE + "+-!~^/%"
    CH_STR_PREFIXES = "f"

    OP_WITH_ASSIGNMENT = ("+", "-", "*", "**", "/", "%", "<<", ">>", "&", "|", "^", "&&", "||")

    KEYWORDS: dict[str, TokenType] = {
        "let": TokenType.KW_LET,
        "const": TokenType.KW_CONST,
        "fn": TokenType.KW_FN,
        "return": TokenType.KW_RETURN,
        "if": TokenType.KW_IF,
        "else": TokenType.KW_ELSE,
        "while": TokenType.KW_WHILE,
        "for": TokenType.KW_FOR,
        "break": TokenType.KW_BREAK,
        "continue": TokenType.KW_CONTINUE,
        "in": TokenType.KW_IN,
        "struct": TokenType.KW_STRUCT,
        # "match": TokenType.KW_MATCH,
        "enum": TokenType.KW_ENUM,
        "static": TokenType.KW_STATIC,
        "scope": TokenType.KW_SCOPE,
        "as": TokenType.KW_AS
    }

    def _reset(self, code: str, filename: str, start_pos: tuple[int, int, list[str] | None] = (0, 0, None)):
        self.i = -1
        self.line = 0
        self.p_line = start_pos[0]
        self.char = 0
        self.p_char = start_pos[1]
        self.code = code
        self.code_lines = code.splitlines()
        self.p_code_lines = start_pos[2] if start_pos[2] is not None else self.code_lines
        self.filename = filename

    def make_pos(self, length: int) -> Position:
        return Position(self.p_line, self.p_char - length, self.p_char - 1,
                        self.p_code_lines[self.p_line], self.filename)

    def make_token(self, type_: TokenType, value: str) -> Token:
        return Token(type_, value, self.make_pos(len(value)))

    def has(self, n: int = 1) -> bool:
        assert n >= 1
        return self.i < len(self.code) - n

    def _increase_index(self, n: int = 1):
        assert n >= 1

        if n > 1:
            for _ in range(n):
                self._increase_index()
            return

        self.i += 1
        self.char += 1
        self.p_char += 1
        if self.char > len(self.code_lines[self.line]):
            self.line += 1
            self.p_line += 1
            self.char = 0
            self.p_char = 0

    def next(self, expected: str = None) -> str:
        if not self.has():
            LexerError.unexpected_eof(self.make_pos(0))

        self._increase_index()
        ch = self.code[self.i]
        if expected is not None and ch not in expected:
            LexerError.unexpected_character(self.make_pos(1), ch)

        return ch

    def current(self) -> str:
        assert self.i >= 0
        return self.code[self.i]

    def lookahead(self, expected: str = None, n: int = 1, take_if_matches: bool = True) -> str:
        assert n >= 1

        if not self.has(n):
            return ""

        ch = self.code[self.i + n]
        if expected is not None:
            if ch not in expected:
                return ""

            if take_if_matches:
                self._increase_index()

        return ch

    def lookahead_str(self, expected: str, take_if_matches: bool = True) -> bool:
        assert len(expected) > 0

        if not self.has(len(expected)):
            return False

        if all(ch == self.lookahead(n=i) for i, ch in enumerate(expected, start=1)):
            if take_if_matches:
                self._increase_index(len(expected))

            return True

        return False

    @staticmethod
    def _lex_simple_match(ch: str, tokens: list[Token], matches: list[tuple[str, Callable[[], Token | None]]]) -> bool:
        for match, func in matches:
            if ch in match:
                if (tok := func()) is not None:
                    tokens.append(tok)
                    return True

        return False

    def _lex_single_double(self, ch: str, tokens: list[Token],
                           matches: list[tuple[str, TokenType] | tuple[str, TokenType, TokenType]]) -> bool:

        for match in matches:
            # single character match
            if len(match) == 2:
                if ch == match[0]:
                    self.next()
                    tokens.append(self.make_token(match[1], ch))
                    return True

            # single or double character match
            elif len(match) == 3:
                if ch == match[0]:
                    self.next()
                    if self.lookahead(ch):
                        tokens.append(self.make_token(match[2], ch + ch))
                    else:
                        tokens.append(self.make_token(match[1], ch))
                    return True

            else:
                raise ValueError("Invalid size of tuple")

        return False

    def lex(self, code: str, filename: str, start_pos: tuple[int, int, list[str] | None] = (0, 0, None)) -> list[Token]:
        self._reset(code, filename, start_pos)

        tokens = []

        while self.has():
            ch = self.lookahead()

            # skip whitespace
            if not ch.strip():
                self.next()
                continue

            # skip comments
            if self.lookahead_str("//"):
                while self.has() and self.next() != "\n":
                    pass
                continue

            if self.lookahead_str("->"):
                tokens.append(self.make_token(TokenType.ARROW, "->"))

            elif self._lex_simple_match(ch, tokens, [
                (Lexer.CH_INTEGER, self.lex_number),
                ("\"", self.lex_string),
                (Lexer.CH_ID_START, self.lex_id),
                ("=", self.lex_assignment),
                (Lexer.CH_OP, self.lex_operator),
                (".", self.lex_dot)
            ]):
                pass

            elif self._lex_single_double(ch, tokens, [
                (";", TokenType.SEMICOLON),
                (":", TokenType.COLON, TokenType.DOUBLE_COLON),
                (",", TokenType.COMMA),
                ("#", TokenType.HASH),
                ("(", TokenType.LPAREN),
                (")", TokenType.RPAREN),
                ("{", TokenType.LBRACE),
                ("}", TokenType.RBRACE),
                ("[", TokenType.LBRACK),
                ("]", TokenType.RBRACK),
                ("?", TokenType.QUESTION)
            ]):
                pass

            else:
                LexerError.unexpected_character(self.make_pos(1), ch)

        return tokens

    def lex_number(self) -> Token:
        val = self.next()

        while ch := self.lookahead(Lexer.CH_INTEGER):
            val += ch

        # is a float
        if self.lookahead(".", take_if_matches=False) and self.lookahead(Lexer.CH_INTEGER, 2, False):
            val += self.next()
            while ch := self.lookahead(Lexer.CH_INTEGER):
                val += ch

        return self.make_token(TokenType.NUMBER, val)

    def lex_string(self) -> Token:
        if self.lookahead_str("\"\"\""):
            triple_quotes = True
        else:
            self.next("\"")
            triple_quotes = False

        val = ""
        prev = ""
        while ch := self.next():
            if ch == "\"" and prev != "\\" and (not triple_quotes or self.lookahead_str("\"\"")):
                break

            val += ch
            prev = ch

        return self.make_token(TokenType.STRING, val)

    def lex_id(self) -> Token:
        val = self.next()

        while ch := self.lookahead(Lexer.CH_ID):
            val += ch

        return self.make_token(Lexer.KEYWORDS.get(val, TokenType.ID), val)

    def lex_assignment(self) -> Token:
        self.next()

        if self.lookahead("="):
            if self.lookahead("="):
                return self.make_token(TokenType.OPERATOR, "===")
            return self.make_token(TokenType.OPERATOR, "==")

        return self.make_token(TokenType.ASSIGNMENT, "=")

    def lex_operator(self) -> Token:
        val = self.next()

        if val == "%":
            if ch := self.lookahead(Lexer.CH_HEX_DIGIT):
                val += ch
                for _ in range(5):
                    if ch := self.lookahead(Lexer.CH_HEX_DIGIT):
                        val += ch
                    else:
                        break
                return self.make_token(TokenType.COLOR, val)

        if val == "!" and self.lookahead("="):
            if self.lookahead("="):
                return self.make_token(TokenType.OPERATOR, "!==")
            return self.make_token(TokenType.OPERATOR, "!=")

        # ++, **, ...
        if val in Lexer.CH_OP_DOUBLE:
            if self.lookahead(val):
                val += val

        # <=, >=
        if val in ("<", ">"):
            if self.lookahead("="):
                val += "="

        # +=, **=, ...
        elif val in Lexer.OP_WITH_ASSIGNMENT:
            if self.lookahead("="):
                val += "="
                return self.make_token(TokenType.ASSIGNMENT, val)

        return self.make_token(TokenType.OPERATOR, val)

    def lex_dot(self) -> Token:
        self.next()

        if self.lookahead("."):
            if self.lookahead("."):
                return self.make_token(TokenType.ELLIPSIS, "...")
            return self.make_token(TokenType.DOUBLE_DOT, "..")

        return self.make_token(TokenType.DOT, ".")
