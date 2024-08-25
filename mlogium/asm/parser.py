from ..tokens import Token, TokenType
from ..base_parser import BaseParser
from ..lexer import Lexer
from ..error import ParserError
from .node import *
from typing import Callable
import copy


class AsmParser(BaseParser[AsmNode]):
    def __init__(self, tokens: list[Token]):
        super().__init__(self.modify_tokens(tokens))

    @staticmethod
    def modify_tokens(tokens: list[Token]) -> list[Token]:
        tokens = copy.deepcopy(tokens)
        keyword_types = set(Lexer.KEYWORDS.values())
        for tok in tokens:
            if tok.type in keyword_types:
                tok.type = TokenType.ID
        return tokens

    def _parse_comma_separated[T](self, func: Callable[[], T], start: TokenType | None = TokenType.LPAREN,
                                  end: TokenType = TokenType.RPAREN, end_val: str = None) -> tuple[list[T], Token]:
        if start is not None:
            self.next(start)

        values = []
        prev = TokenType.LPAREN
        while True:
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
                return values, tok

            else:
                if prev == TokenType.ID:
                    ParserError.unexpected_token(tok, "comma or closing parenthesis")
                values.append(func())
                prev = TokenType.ID

    def parse(self) -> AsmNode:
        statements = []
        while self.has():
            statements.append(self.parse_statement())

        if len(statements) > 0:
            return RootAsmNode(statements[0].pos + statements[-1].pos, statements)
        else:
            return RootAsmNode(self._current_pos(), statements)

    def parse_statement(self) -> AsmNode:
        if tok := self.lookahead(TokenType.ARROW):
            name = self.next(TokenType.ID)
            if self.lookahead(TokenType.LPAREN):
                left = self.next(TokenType.ID).value
                op = self.next(TokenType.OPERATOR, ("<", "<=", ">", ">=", "==", "!=", "===")).value
                right = self.next(TokenType.ID).value
                close = self.next(TokenType.RPAREN)
                return JumpAsmNode(tok.pos + close.pos, name.value, JumpAsmNode.Condition(left, op, right))
            return JumpAsmNode(tok.pos + name.pos, name.value, None)

        elif tok := self.lookahead(TokenType.ID):
            if colon := self.lookahead(TokenType.COLON):
                return LabelAsmNode(tok.pos + colon.pos, tok.value)

            elif self.lookahead(TokenType.ASSIGNMENT, "="):
                if op := self.lookahead(TokenType.OPERATOR, ("~", "!")):
                    value = self.parse_primitive_value()
                    return UnaryOpAsmNode(tok.pos + value.pos, tok.value, op.value, value.value)

                elif (self.lookahead(TokenType.OPERATOR, "-", take_if_matches=False) and
                      (value := self.lookahead(TokenType.ID, n=2, take_if_matches=False))):
                    self.skip(2)
                    return UnaryOpAsmNode(tok.pos + value.pos, tok.value, "-", value.value)

                if value := self.lookahead(TokenType.ID, take_if_matches=False):
                    if self.lookahead(TokenType.DOT, n=2, take_if_matches=False):
                        self.skip(2)
                        prop = self.next(TokenType.ID)
                        return PropertyReadAsmNode(tok.pos + prop.pos, tok.value, value.value, prop.value)

                    elif self.lookahead(TokenType.LPAREN, n=2, take_if_matches=False):
                        self.skip(2)
                        params, rparen = self._parse_comma_separated(self.parse_primitive_value_str, None)
                        return CallAsmNode(tok.pos + rparen.pos, tok.value, value.value, params)

                    elif self.lookahead(TokenType.LBRACK, n=2, take_if_matches=False):
                        self.skip(2)
                        index = self.parse_primitive_value()
                        rbrack = self.next(TokenType.RBRACK)
                        return IndexReadAsmNode(tok.pos + rbrack.pos, tok.value, value.value, index.value)

                value = self.parse_primitive_value()
                if op := self.lookahead(TokenType.OPERATOR):
                    value2 = self.parse_primitive_value()
                    return BinaryOpAsmNode(tok.pos + value2.pos, tok.value, value.value, op.value, value2.value)

                return AssignmentAsmNode(tok.pos + value.pos, tok.value, value.value)

            elif op := self.lookahead(TokenType.ASSIGNMENT):
                value = self.parse_primitive_value()
                return ModifyAsmNode(tok.pos + value.pos, tok.value, op.value, value.value)

            elif self.lookahead(TokenType.DOT):
                prop = self.next()
                self.next(TokenType.ASSIGNMENT, "=")
                value = self.parse_primitive_value()
                return PropertyWriteAsmNode(tok.pos + value.pos, tok.value, prop.value, value.value)

            elif self.lookahead(TokenType.LPAREN):
                params, rparen = self._parse_comma_separated(self.parse_primitive_value_str, None)
                return CallAsmNode(tok.pos + rparen.pos, None, tok.value, params)

            elif self.lookahead(TokenType.LBRACK):
                index = self.parse_primitive_value()
                self.next(TokenType.RBRACK)
                self.next(TokenType.ASSIGNMENT, "=")
                value = self.parse_primitive_value()
                return IndexWriteAsmNode(tok.pos + value.pos, tok.value, index.value, value.value)

        elif tok := self.lookahead(TokenType.LBRACK):
            results, _ = self._parse_comma_separated(lambda: self.next(TokenType.ID).value, None, TokenType.RBRACK)
            self.next(TokenType.ASSIGNMENT, "=")
            value = self.next(TokenType.ID)
            params, rparen = self._parse_comma_separated(self.parse_primitive_value_str)
            return UnpackCallAsmNode(tok.pos + rparen.pos, results, value.value, params)

        else:
            ParserError.unexpected_token(self.next())

    def parse_primitive_value_str(self) -> str:
        return self.parse_primitive_value().value

    def parse_primitive_value(self) -> Token:
        if neg := self.lookahead(TokenType.OPERATOR, "-"):
            value = self.next(TokenType.ID | TokenType.FLOAT)
            return Token(value.type, f"-{value.value}", neg.pos + value.pos)
        elif tok := self.lookahead(TokenType.STRING):
            return Token(tok.type, f"\"{tok.value}\"", tok.pos)
        return self.next(TokenType.ID | TokenType.INTEGER | TokenType.FLOAT)
