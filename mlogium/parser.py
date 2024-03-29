from .tokens import Token, TokenType
from .error import ParserError
from .node import *
from .macro import MacroInput, CustomMacroInput, MacroInvocationContext, MacroRegistry

from typing import Callable


class Parser:
    i: int
    tokens: list[Token]
    macro_registry: MacroRegistry

    def __init__(self, tokens: list[Token], macro_registry: MacroRegistry):
        self.i = -1
        self.tokens = tokens
        self.macro_registry = macro_registry

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

    def next(self, type_: TokenType = None, value: str | tuple[str, ...] = None) -> Token:
        if not self.has():
            ParserError.unexpected_eof(self._current_pos())

        self.i += 1
        tok = self.tokens[self.i]
        if not self._match_token(tok, type_, value):
            ParserError.unexpected_token(tok)

        return tok

    def lookahead(self, type_: TokenType = None, value: str | tuple[str, ...] = None,
                  n: int = 1, take_if_matches: bool = True) -> Token | None:
        assert n >= 1

        if not self.has(n):
            return None

        tok = self.tokens[self.i + n]
        if not self._match_token(tok, type_, value):
            return None

        if take_if_matches and (type_ is not None or value is not None):
            assert n == 1
            self.i += 1

        return tok

    def parse(self) -> Node:
        return self.parse_block(False, True)

    def parse_block(self, end_on_rbrace: bool, top_level: bool = False) -> Node:
        code = []
        returns_last = True

        while self.has():
            if end_on_rbrace and self.lookahead(TokenType.RBRACE):
                returns_last = True
                break

            elif end_on_rbrace and self.lookahead(TokenType.SEMICOLON, take_if_matches=False) \
                    and self.lookahead(TokenType.RBRACE, n=2, take_if_matches=False):
                self.next()
                self.next()
                returns_last = False
                break

            self.lookahead(TokenType.SEMICOLON)

            if not self.has():
                break

            code.append(self.parse_top_level_statement() if top_level else self.parse_statement())

        return BlockNode(self._current_pos() if not code else code[0].pos + code[-1].pos,
                         code, returns_last and not top_level)

    def _parse_struct_inner(self) -> tuple[list[SingleAssignmentTarget], list[tuple[bool, SingleAssignmentTarget, Node]], list[tuple[bool, str, NamedParamFunctionType, Node]], list[tuple[str, NamedParamFunctionType, Node]]]:
        fields: list[SingleAssignmentTarget] = []
        static_fields: list[tuple[bool, SingleAssignmentTarget, Node]] = []
        methods: list[tuple[bool, str, NamedParamFunctionType, Node]] = []
        static_methods: list[tuple[str, NamedParamFunctionType, Node]] = []

        while self.has() and not self.lookahead(TokenType.RBRACE, take_if_matches=False):
            if self.lookahead(TokenType.KW_STATIC):
                if (const := bool(self.lookahead(TokenType.KW_CONST))) or self.lookahead(TokenType.KW_LET):
                    name = self.next(TokenType.ID).value
                    if self.lookahead(TokenType.COLON):
                        type_ = self.parse_type()
                    else:
                        type_ = None
                    self.next(TokenType.ASSIGNMENT, "=")
                    value = self.parse_value()
                    static_fields.append((const, SingleAssignmentTarget(name, type_), value))

                else:
                    self.next(TokenType.KW_FN)
                    name = self.next(TokenType.ID).value
                    type_ = self._parse_named_func_type()
                    self.next(TokenType.LBRACE)
                    code = self.parse_block(True)
                    static_methods.append((name, type_, code))

            else:
                if self.lookahead(TokenType.KW_LET):
                    name = self.next(TokenType.ID).value
                    if self.lookahead(TokenType.COLON):
                        type_ = self.parse_type()
                    else:
                        type_ = None
                    fields.append(SingleAssignmentTarget(name, type_))

                else:
                    const = bool(self.lookahead(TokenType.KW_CONST))
                    self.next(TokenType.KW_FN)
                    name = self.next(TokenType.ID).value
                    type_ = self._parse_named_func_type()
                    self.next(TokenType.LBRACE)
                    code = self.parse_block(True)
                    methods.append((const, name, type_, code))

            while self.lookahead(TokenType.SEMICOLON):
                pass

        self.next(TokenType.RBRACE)

        return fields, static_fields, methods, static_methods

    def _parse_macro(self, top_level: bool) -> Node:
        name = self.next(TokenType.ID)
        if (macro := self.macro_registry.get(name.value)) is None:
            ParserError.custom(name.pos, f"Macro not found: '{name.value}'")
        if macro.top_level_only() and not top_level:
            ParserError.custom(name.pos, f"Macro must be called in the top level")

        self.next(TokenType.LPAREN)
        params = []
        for i, inp in enumerate(macro.inputs()):
            if i > 0:
                self.next(TokenType.COMMA)

            match inp:
                case MacroInput.TYPE:
                    params.append(self.parse_type())
                case MacroInput.TOKEN:
                    params.append(self.next())
                case MacroInput.VALUE_NODE:
                    params.append(self.parse_value())
                case MacroInput.BLOCK_NODE:
                    self.next(TokenType.LBRACE)
                    params.append(self.parse_block(True))
                case CustomMacroInput(func):
                    params.append(func(self))
                case _:
                    raise ValueError("Unknown macro parameter type")
        self.next(TokenType.RPAREN)

        return macro.invoke(MacroInvocationContext(name.pos, self.macro_registry), params)

    def _parse_top_level_statement_or_none(self) -> Node | None:
        tok = self.lookahead()

        if tok.type == TokenType.KW_FN:
            self.next()
            name = self.next(TokenType.ID).value
            type_ = self._parse_named_func_type()
            self.next(TokenType.LBRACE)
            code = self.parse_block(True)
            return FunctionNode(tok.pos + code.pos, name, type_, code)

        elif tok.type in TokenType.KW_STRUCT:
            self.next()
            name = self.next(TokenType.ID).value
            self.next(TokenType.LBRACE)

            fields, static_fields, methods, static_methods = self._parse_struct_inner()

            return StructNode(tok.pos, name, fields, static_fields, methods, static_methods)

        elif tok.type == TokenType.KW_ENUM:
            self.next()
            name = self.next(TokenType.ID).value
            options = self._parse_comma_separated(lambda: self.next(TokenType.ID).value, TokenType.LBRACE,
                                                  TokenType.RBRACE)
            return EnumNode(tok.pos, name, options)

        elif tok.type == TokenType.HASH:
            self.next()
            return self._parse_macro(True)

        return None

    def parse_top_level_statement(self) -> Node:
        if (node := self._parse_top_level_statement_or_none()) is not None:
            return node

        return self.parse_statement()

    def _parse_optional_type(self) -> Type | None:
        if self.lookahead(TokenType.ID, "_"):
            return None

        return self.parse_type()

    def _parse_unpack(self, pos: Position) -> list[str | tuple[str, Type]]:
        names = self._parse_comma_separated(lambda: self.next(TokenType.ID).value)
        if self.lookahead(TokenType.COLON):
            types = self._parse_comma_separated(self._parse_optional_type)
        else:
            types = None

        if types is None:
            return names

        if len(names) != len(types):
            ParserError.custom(pos, f"Unpack count doesn't match")
        return list(zip(names, types))

    def _parse_assignment_target(self, pos: Position) -> AssignmentTarget:
        if self.lookahead(TokenType.LPAREN, take_if_matches=False):
            unpack = self._parse_unpack(pos)
            return UnpackAssignmentTarget(unpack)

        name = self.next().value
        if self.lookahead(TokenType.COLON):
            type_ = self.parse_type()
        else:
            type_ = None
        return SingleAssignmentTarget(name, type_)

    def parse_statement(self) -> Node:
        if (node := self._parse_top_level_statement_or_none()) is not None:
            return node

        tok = self.lookahead()

        if tok.type in TokenType.KW_LET | TokenType.KW_CONST:
            const = tok.type == TokenType.KW_CONST

            self.next()

            target = self._parse_assignment_target(tok.pos)
            self.next(TokenType.ASSIGNMENT, "=")
            val = self.parse_value()
            return DeclarationNode(tok.pos + val.pos, const, target, val)

        elif tok.type == TokenType.KW_WHILE:
            self.next()
            cond = self.parse_value()
            code = self.parse_statement()
            return WhileNode(tok.pos + code.pos, cond, code)

        elif tok.type == TokenType.KW_FOR:
            self.next()
            self.next(TokenType.LPAREN)

            target = self._parse_assignment_target(tok.pos)
            self.next(TokenType.KW_IN)
            iterator = self.parse_value()

            self.next(TokenType.RPAREN)

            code = self.parse_statement()

            return ForNode(tok.pos + code.pos, target, iterator, code)

        elif tok.type == TokenType.KW_RETURN:
            self.next()

            if self.lookahead(TokenType.SEMICOLON, take_if_matches=False):
                return ReturnNode(tok.pos, None)

            val = self.parse_value()
            return ReturnNode(tok.pos + val.pos, val)

        elif tok.type == TokenType.KW_BREAK:
            self.next()
            return BreakNode(tok.pos)

        elif tok.type == TokenType.KW_CONTINUE:
            self.next()
            return ContinueNode(tok.pos)

        return self.parse_value()

    def _parse_comma_separated[T](self, func: Callable[[], T], start: TokenType | None = TokenType.LPAREN,
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
                    ParserError.unexpected_token(tok)

                break

            else:
                if prev == TokenType.ID:
                    ParserError.unexpected_token(tok)

                values.append(func())

                prev = TokenType.ID

        return values

    def _parse_name_with_type(self) -> tuple[str, Type]:
        name = self.next(TokenType.ID).value
        self.next(TokenType.COLON)
        type_ = self.parse_type()
        return name, type_

    def _parse_named_func_type(self) -> NamedParamFunctionType:
        params = self._parse_comma_separated(self._parse_name_with_type)

        if self.lookahead(TokenType.ARROW):
            return NamedParamFunctionType(params, self.parse_type())

        return NamedParamFunctionType(params, NullType())

    def _parse_func_type(self) -> FunctionType:
        params = self._parse_comma_separated(self.parse_type)

        if self.lookahead(TokenType.ARROW):
            return FunctionType(params, self.parse_type())

        return FunctionType(params, NullType())

    def _parse_tuple_type(self) -> TupleType:
        return TupleType(self._parse_comma_separated(self.parse_type))

    def _parse_basic_type(self) -> BasicType:
        name = self.next(TokenType.ID).value

        return BasicType(name)

    def parse_type(self) -> Type:
        if self.lookahead(TokenType.KW_FN):
            type_ = self._parse_func_type()

        elif self.lookahead(TokenType.LPAREN, take_if_matches=False):
            type_ = self._parse_tuple_type()

        else:
            type_ = self._parse_basic_type()

        return type_

    def parse_value(self) -> Node:
        return self.parse_assignment()

    def _parse_binary_op(self, values: tuple[str, ...] | None, func: Callable[[], Node],
                         type_: TokenType = TokenType.OPERATOR, single: bool = False) -> Node:

        node = func()

        while self.has():
            if (op := self.lookahead(type_, values)) is not None:
                right = func()
                node = BinaryOpNode(node.pos + right.pos, node, op.value, right)

                if single:
                    break

            else:
                break

        return node

    def _parse_unary_op(self, values: tuple[str, ...], func: Callable[[], Node]) -> Node:
        if (op := self.lookahead(TokenType.OPERATOR, values)) is not None:
            value = self._parse_unary_op(values, func)
            return UnaryOpNode(op.pos + value.pos, op.value, value)

        return func()

    def parse_assignment(self) -> Node:
        return self._parse_binary_op(None, self.parse_range, TokenType.ASSIGNMENT, True)

    def parse_range(self) -> Node:
        val = self.parse_logical_or()

        if self.lookahead(TokenType.DOUBLE_DOT):
            end = self.parse_logical_or()
            return RangeValueNode(val.pos + end.pos, val, end)

        return val

    def parse_logical_or(self) -> Node:
        return self._parse_binary_op(("||",), self.parse_logical_and)

    def parse_logical_and(self) -> Node:
        return self._parse_binary_op(("&&",), self.parse_logical_not)

    def parse_logical_not(self) -> Node:
        return self._parse_unary_op(("!",), self.parse_comparison)

    # def parse_in(self) -> Node:
    #     return self._parse_binary_op(("in",), self.parse_comparison, TokenType.KW_IN, True)

    def parse_comparison(self) -> Node:
        return self._parse_binary_op(("<", ">", "<=", ">=", "==", "!=", "===", "!=="), self.parse_bitwise_binary)

    def parse_bitwise_binary(self) -> Node:
        return self._parse_binary_op(("|", "&", "^"), self.parse_bitwise_shift)

    def parse_bitwise_shift(self) -> Node:
        return self._parse_binary_op(("<<", ">>"), self.parse_arith)

    def parse_arith(self) -> Node:
        return self._parse_binary_op(("+", "-"), self.parse_term)

    def parse_term(self) -> Node:
        return self._parse_binary_op(("*", "/", "%"), self.parse_factor)

    def parse_factor(self) -> Node:
        return self._parse_unary_op(("-", "~"), self.parse_power)

    def parse_power(self) -> Node:
        return self._parse_binary_op(("**",), self.parse_call_index_attr)

    def parse_call_index_attr(self) -> Node:
        node = self.parse_atom()

        while self.has():
            if self.lookahead(TokenType.LPAREN, take_if_matches=False):
                params = self._parse_comma_separated(self.parse_value)
                node = CallNode(node.pos, node, params)

            elif self.lookahead(TokenType.LBRACK):
                index = self.parse_value()
                self.next(TokenType.RBRACK)
                node = IndexNode(node.pos, node, index)

            elif tok := self.lookahead(TokenType.DOT | TokenType.DOUBLE_COLON):
                attr = self.next(TokenType.ID | TokenType.NUMBER)
                if attr.type == TokenType.NUMBER and "." in attr.value:
                    ParserError.unexpected_token(attr)
                node = AttributeNode(node.pos, node, attr.value, tok.type == TokenType.DOUBLE_COLON)

            else:
                break

        return node

    # def _parse_pattern(self) -> Pattern:
    #     tok = self.lookahead()
    #
    #     if tok.type == TokenType.INTEGER:
    #         self.next()
    #         return IntegerPattern(int(tok.value))
    #
    #     elif tok.type == TokenType.FLOAT:
    #         self.next()
    #         return FloatPattern(float(tok.value))
    #
    #     elif tok.type == TokenType.STRING:
    #         self.next()
    #         return StringPattern(tok.value)
    #
    #     elif tok.type == TokenType.LPAREN:
    #         return TuplePattern(self._parse_comma_separated(self._parse_pattern, TokenType.LPAREN))
    #
    #     elif tok.type == TokenType.LBRACK:
    #         return ListPattern(self._parse_comma_separated(self._parse_pattern, TokenType.LBRACK, TokenType.RBRACK))
    #
    #     name = self.next(TokenType.ID).value
    #
    #     if name == "_":
    #         return AnyPattern()
    #
    #     if self.lookahead(TokenType.DOUBLE_COLON):
    #         names = [name, self.next(TokenType.ID).value]
    #         while self.lookahead(TokenType.DOUBLE_COLON):
    #             names.append(self.next(TokenType.ID).value)
    #         if self.lookahead(TokenType.LPAREN):
    #             return NamedPattern(names, self._parse_comma_separated(self._parse_pattern, None))
    #         return NamedPattern(names, None)
    #
    #     return VariablePattern(name)

    def parse_atom(self) -> Node:
        tok = self.next()

        if tok.type == TokenType.KW_IF:
            cond = self.parse_value()
            code_if = self.parse_statement()
            pos = tok.pos + code_if.pos
            if self.lookahead(TokenType.KW_ELSE):
                code_else = self.parse_statement()
                pos = pos + code_else.pos
            else:
                code_else = None
            return IfNode(pos, cond, code_if, code_else)

        elif tok.type == TokenType.LBRACE:
            return self.parse_block(True)

        elif tok.type == TokenType.LPAREN:
            if self.lookahead(TokenType.RPAREN):
                return TupleValueNode(tok.pos, [])
            val = self.parse_value()
            if self.lookahead(TokenType.COMMA):
                return TupleValueNode(tok.pos, [val] + self._parse_comma_separated(self.parse_value, None))
            else:
                self.next(TokenType.RPAREN)
                return val

        elif tok.type == TokenType.NUMBER:
            number = float(tok.value)
            if number.is_integer():
                number = int(number)
            return NumberValueNode(tok.pos, number)

        elif tok.type == TokenType.STRING:
            return StringValueNode(tok.pos, tok.value)

        elif tok.type == TokenType.LPAREN:
            value = self.parse_value()
            self.next(TokenType.RPAREN)
            return value

        elif tok.type == TokenType.ID:
            return VariableValueNode(tok.pos, tok.value)

        elif tok.type == TokenType.OPERATOR and tok.value == "|":
            params = self._parse_comma_separated(self._parse_name_with_type, None, TokenType.OPERATOR, "|")
            if self.lookahead(TokenType.ARROW):
                ret = self.parse_type()
            else:
                ret = NullType()
            self.next(TokenType.LBRACE)
            code = self.parse_block(True)
            return LambdaNode(tok.pos + code.pos, NamedParamFunctionType(params, ret), code)

        elif tok.type == TokenType.OPERATOR and tok.value == "||":
            if self.lookahead(TokenType.ARROW):
                ret = self.parse_type()
            else:
                ret = NullType()
            self.next(TokenType.LBRACE)
            code = self.parse_block(True)
            return LambdaNode(tok.pos + code.pos, NamedParamFunctionType([], ret), code)

        elif tok.type == TokenType.HASH:
            return self._parse_macro(False)

        ParserError.unexpected_token(tok)
