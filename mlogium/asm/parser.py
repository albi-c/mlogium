from ..tokens import Token, TokenType
from ..base_parser import BaseParser
from ..lexer import Lexer
from ..error import ParserError
from .node import *
from typing import Callable
import copy


class AsmParser(BaseParser[AsmNode]):
    macros: dict[str, tuple[list[str], list[Token]]]

    TMP_INDEX: int = 0

    def __init__(self, tokens: list[Token], macros: dict[str, tuple[list[str], list[Token]]] = None):
        super().__init__(self.modify_tokens(tokens))

        self.macros = macros if macros is not None else {}

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
            if (node := self.parse_statement()) is not None:
                statements.append(node)

        if len(statements) > 0:
            return RootAsmNode(statements[0].pos + statements[-1].pos, statements)
        else:
            return RootAsmNode(self._current_pos(), statements)

    def _parse_call_param(self) -> str:
        if tok := self.lookahead(TokenType.OPERATOR, "-"):
            return tok.value
        elif self.lookahead(TokenType.OPERATOR, "&"):
            output = self.next(TokenType.ID)
            return f"&{output.value}"
        return self.parse_primitive_value().value

    def _parse_label_with_offset(self) -> tuple[str, Position]:
        if tok := self.lookahead(TokenType.DOT):
            return "+0", tok.pos
        elif value := self.lookahead(TokenType.INTEGER):
            return value.value, value.pos
        elif op := self.lookahead(TokenType.OPERATOR, ("+", "-")):
            offset = self.next(TokenType.INTEGER)
            return f"{op.value}{offset.value}", offset.pos
        elif value := self.next(TokenType.ID):
            if op := self.lookahead(TokenType.OPERATOR, ("+", "-")):
                offset = self.next(TokenType.INTEGER)
                return f"{value.value}{op.value}{offset.value}", offset.pos
            else:
                return value.value, value.pos
        else:
            ParserError.unexpected_token(tok, str(TokenType.INTEGER | TokenType.OPERATOR))

    def _parse_until(self, type_: TokenType, take_end: bool, append_end: bool = False) -> list[Token]:
        tokens = []
        while True:
            if end := self.lookahead(type_, take_if_matches=take_end):
                if append_end:
                    tokens.append(end)
                return tokens
            tokens.append(self.next())

    def _parse_macro_param(self) -> list[Token]:
        if self.lookahead(TokenType.LBRACE, take_if_matches=False):
            return self._parse_until(TokenType.RBRACE, True, True)
        return self._parse_until(TokenType.COMMA | TokenType.RPAREN, False)

    def parse_statement(self) -> AsmNode | None:
        if tok := self.lookahead(TokenType.LBRACE):
            nodes = []
            while True:
                if end := self.lookahead(TokenType.RBRACE):
                    return BlockAsmNode(tok.pos + end.pos, nodes)
                if (node := self.parse_statement()) is not None:
                    nodes.append(node)

        elif self.lookahead(TokenType.HASH):
            name = self.next(TokenType.ID)
            if name.value == "def":
                macro_name = self.next(TokenType.ID)
                if macro_name.value in ("def", "undef"):
                    ParserError.custom(macro_name.pos, f"Reserved macro name: '{macro_name.value}'")
                if macro_name.value in self.macros:
                    ParserError.custom(macro_name.pos, f"Macro redefinition: '{macro_name.value}'")
                if self.lookahead(TokenType.LPAREN):
                    params, _ = self._parse_comma_separated(lambda: self.next(TokenType.ID).value, None)
                else:
                    params = []
                tokens = self._parse_until(TokenType.RBRACE, True, True)
                self.macros[macro_name.value] = params, tokens
            elif name.value == "undef":
                macro_name = self.next(TokenType.ID)
                if macro_name.value in self.macros:
                    del self.macros[macro_name.value]
                else:
                    ParserError.custom(macro_name.pos, f"Macro not found: '{macro_name.value}'")
            else:
                if (macro := self.macros.get(name.value)) is not None:
                    params: list[str] = macro[0]
                    tokens: list[Token] = macro[1]
                    if self.lookahead(TokenType.LPAREN):
                        replacements, _ = self._parse_comma_separated(self._parse_macro_param, None)
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
                                val = Token(TokenType.ID, f"__asm_tmp_{AsmParser.TMP_INDEX}", tok.pos)
                                temp_variables[index] = val
                                processed_tokens.append(val)
                                AsmParser.TMP_INDEX += 1
                        else:
                            processed_tokens.append(tok)
                        i += 1
                    return AsmParser(processed_tokens, self.macros).parse_statement()
                else:
                    ParserError.custom(name.pos, f"Macro not found: '{name.value}'")

        elif tok := self.lookahead(TokenType.ARROW):
            dest, dest_pos = self._parse_label_with_offset()
            if self.lookahead(TokenType.LPAREN):
                left = self.next(TokenType.ID).value
                op = self.next(TokenType.OPERATOR, ("<", "<=", ">", ">=", "==", "!=", "===")).value
                right = self.next(TokenType.ID).value
                close = self.next(TokenType.RPAREN)
                return JumpAsmNode(tok.pos + close.pos, dest, JumpAsmNode.Condition(left, op, right))
            return JumpAsmNode(tok.pos + dest_pos, dest, None)

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
                        params, rparen = self._parse_comma_separated(self._parse_call_param, None)
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
                params, rparen = self._parse_comma_separated(self._parse_call_param, None)
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
            params, rparen = self._parse_comma_separated(self._parse_call_param)
            return UnpackCallAsmNode(tok.pos + rparen.pos, results, value.value, params)

        else:
            ParserError.unexpected_token(self.next())

    def parse_primitive_value(self) -> Token:
        if neg := self.lookahead(TokenType.OPERATOR, "-"):
            value = self.next(TokenType.ID | TokenType.FLOAT)
            return Token(value.type, f"-{value.value}", neg.pos + value.pos)
        elif tok := self.lookahead(TokenType.STRING):
            return Token(tok.type, f"\"{tok.value}\"", tok.pos)
        elif tok := self.lookahead(TokenType.DOLLAR):
            dest, dest_pos = self._parse_label_with_offset()
            return Token(tok.type, f"${dest}", tok.pos + dest_pos)
        return self.next(TokenType.ID | TokenType.INTEGER | TokenType.FLOAT)
