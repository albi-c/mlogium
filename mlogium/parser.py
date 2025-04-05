from .tokens import TokenType, Token
from .error import ParserError
from .node import *
from .base_parser import BaseParser
from .macro import MacroRegistry

from typing import Callable


class Parser(BaseParser[Node]):
    macro_reg: MacroRegistry
    is_comptime: bool

    TMP_INDEX: int = 0

    def __init__(self, tokens: list[Token], macro_reg: MacroRegistry = None):
        super().__init__(tokens)

        self.macro_reg = macro_reg if macro_reg is not None else MacroRegistry()
        self.is_comptime = False

    def parse(self) -> Node:
        return self.parse_block(False, False)

    def parse_block(self, end_on_rbrace: bool, can_return_last: bool) -> Node:
        code = []
        returns_last = False

        while self.has():
            if end_on_rbrace and self.lookahead(TokenType.RBRACE):
                returns_last = False
                break

            statement = self.parse_statement()
            code.append(statement)

            if can_return_last and end_on_rbrace and self.lookahead(TokenType.RBRACE):
                returns_last = True
                break

            else:
                if (curr := self.current()) and curr.type == TokenType.RBRACE:
                    self.lookahead(TokenType.SEMICOLON)
                else:
                    self.next(TokenType.SEMICOLON)
                if end_on_rbrace and self.lookahead(TokenType.RBRACE):
                    returns_last = False
                    break

        assert can_return_last or not returns_last
        returns_last = returns_last and len(code) > 0

        return BlockNode(self._current_pos() if not code else code[0].pos + code[-1].pos,
                         code, returns_last)

    def _parse_int(self) -> int:
        neg = -1 if self.lookahead(TokenType.OPERATOR, "-") else 1
        return neg * int(self.next(TokenType.INTEGER).value)

    def parse_statement(self) -> Node:
        tok = self.lookahead()

        if tok.type == TokenType.KW_WHILE:
            self.next()
            cond = self.parse_value()
            self.next(TokenType.LBRACE)
            code = self.parse_block(True, False)
            return WhileNode(tok.pos, cond, code)

        elif tok.type == TokenType.KW_FOR:
            self.next()
            target = self._parse_assignment_target(False)
            self.next(TokenType.KW_IN)
            iterable = self.parse_value()
            self.next(TokenType.LBRACE)
            code = self.parse_block(True, False)
            return ForNode(tok.pos, target, iterable, code)

        elif tok.type == TokenType.KW_UNROLL:
            self.next()
            variable = self.next(TokenType.ID).value
            self.next(TokenType.KW_IN)
            start = self._parse_int()
            self.next(TokenType.DOUBLE_DOT)
            end = self._parse_int()
            if self.lookahead(TokenType.DOUBLE_DOT):
                step = self._parse_int()
                range_obj = range(start, end, step)
            else:
                step = None
                range_obj = range(start, end)
            self.next(TokenType.LBRACE)
            code = self.parse_block(True, False)
            return UnrollNode(tok.pos, variable, (start, end, step), code, range_obj)

        elif tok.type == TokenType.KW_RETURN:
            self.next()
            if self.lookahead(TokenType.SEMICOLON | TokenType.RBRACE, take_if_matches=False):
                return ReturnNode(tok.pos, None)
            else:
                return ReturnNode(tok.pos, self.parse_value())

        elif tok.type == TokenType.KW_BREAK:
            self.next()
            return BreakNode(tok.pos)

        elif tok.type == TokenType.KW_CONTINUE:
            self.next()
            return ContinueNode(tok.pos)

        return self._parse_binary_op(None, self.parse_value, TokenType.ASSIGNMENT, True)

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
                    ParserError.unexpected_token(tok, "value")

                break

            else:
                if prev == TokenType.ID:
                    ParserError.unexpected_token(tok, "comma or closing parenthesis")

                values.append(func())

                prev = TokenType.ID

        return values

    def parse_type(self) -> Node:
        if tok := self.lookahead(TokenType.LPAREN):
            types = self._parse_comma_separated(self.parse_type, None)
            return TupleTypeNode(tok.pos, types)

        elif tok := self.lookahead(TokenType.LBRACK):
            n = int(self.next(TokenType.INTEGER).value)
            self.next(TokenType.RBRACK)
            type_ = self.parse_type()
            return TupleTypeNode(tok.pos, [type_] * n)

        elif tok := self.lookahead(TokenType.KW_FN):
            params = self._parse_comma_separated(
                lambda: None if self.lookahead(TokenType.QUESTION) else self.parse_type())
            self.next(TokenType.ARROW)
            if q_tok := self.lookahead(TokenType.QUESTION):
                result = None
                end_pos = q_tok.pos
            else:
                result = self.parse_type()
                end_pos = result.pos
            return FunctionTypeNode(tok.pos + end_pos, params, result)

        return self.parse_call_index_attr()

    def _parse_assignment_target(self, const: bool) -> AssignmentTarget:
        if self.lookahead(TokenType.LPAREN):
            values = self._parse_comma_separated(lambda: self._parse_assignment_target(const), None)
            return UnpackAssignmentTarget(const, values)

        name = self.next(TokenType.ID).value
        if self.lookahead(TokenType.COLON):
            type_ = self.parse_type()
        else:
            type_ = None
        return SingleAssignmentTarget(const, name, type_)

    def parse_value(self) -> Node:
        tok = self.lookahead()

        if tok.type in (TokenType.KW_LET, TokenType.KW_CONST):
            self.next()
            const = tok.type == TokenType.KW_CONST

            target = self._parse_assignment_target(const)
            self.next(TokenType.ASSIGNMENT, "=")
            val = self.parse_value()
            return DeclarationNode(tok.pos, target, val)

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

    def _parse_unary_op(self, values: tuple[str, ...], func: Callable[[], Node],
                        token_type: TokenType = TokenType.OPERATOR) -> Node:
        if (op := self.lookahead(token_type, values)) is not None:
            value = self._parse_unary_op(values, func)
            return UnaryOpNode(op.pos + value.pos, op.value, value)

        return func()

    def parse_assignment(self) -> Node:
        return self._parse_binary_op(None, self.parse_cast, TokenType.WALRUS, True)

    def parse_cast(self) -> Node:
        val = self.parse_range()
        if self.lookahead(TokenType.KW_AS):
            type_ = self.parse_type()
            return CastNode(val.pos + type_.pos, val, type_)

        return val

    def parse_range(self) -> Node:
        val = self.parse_logical_or()

        if self.lookahead(TokenType.DOUBLE_DOT):
            end = self.parse_logical_or()

            if self.lookahead(TokenType.DOUBLE_DOT):
                step = self.parse_logical_or()
                return RangeValueNode(val.pos + step.pos, val, end, step)

            return RangeValueNode(val.pos + end.pos, val, end, None)

        return val

    def parse_logical_or(self) -> Node:
        return self._parse_binary_op(("||",), self.parse_logical_and)

    def parse_logical_and(self) -> Node:
        return self._parse_binary_op(("&&",), self.parse_logical_not)

    def parse_logical_not(self) -> Node:
        return self._parse_unary_op(("!",), self.parse_comparison)

    def parse_comparison(self) -> Node:
        return self._parse_binary_op(("<", ">", "<=", ">=", "==", "!=", "===", "!=="), self.parse_bitwise_binary)

    def parse_bitwise_binary(self) -> Node:
        return self._parse_binary_op(("|", "&", "^"), self.parse_bitwise_shift)

    def parse_bitwise_shift(self) -> Node:
        return self._parse_binary_op(("<<", ">>"), self.parse_tuple_join)

    def parse_tuple_join(self) -> Node:
        return self._parse_binary_op(("++",), self.parse_arith)

    def parse_arith(self) -> Node:
        return self._parse_binary_op(("+", "-"), self.parse_term)

    def parse_term(self) -> Node:
        return self._parse_binary_op(("*", "/", "/.", "%"), self.parse_factor)

    def parse_factor(self) -> Node:
        return self._parse_unary_op(("*", "-", "~"), self.parse_power)

    def parse_power(self) -> Node:
        return self._parse_binary_op(("**",), self.parse_to_tuple)

    def parse_to_tuple(self) -> Node:
        return self._parse_unary_op(("...",), self.parse_call_index_attr, TokenType.ELLIPSIS)

    def _parse_call_param(self) -> tuple[Node, bool]:
        value = self.parse_value()
        if self.lookahead(TokenType.ELLIPSIS):
            return value, True
        return value, False

    def parse_call_index_attr(self) -> Node:
        node = self.parse_atom()

        while self.has():
            if self.lookahead(TokenType.LPAREN, take_if_matches=False):
                params = self._parse_comma_separated(self._parse_call_param)
                node = CallNode(node.pos, node, params)

            elif self.lookahead(TokenType.LBRACK):
                indices = self._parse_comma_separated(self.parse_value, None, TokenType.RBRACK)
                node = IndexNode(node.pos, node, indices)

            elif tok := self.lookahead(TokenType.DOT | TokenType.DOUBLE_COLON):
                attr = self.next(TokenType.ID | TokenType.INTEGER)
                node = AttributeNode(node.pos, node, attr.value, tok.type == TokenType.DOUBLE_COLON)

            else:
                break

        return node

    def _parse_function_param(self) -> FunctionParam:
        ref = bool(self.lookahead(TokenType.OPERATOR, "&"))
        name = self.next(TokenType.ID).value
        variadic = self.lookahead(TokenType.ELLIPSIS)
        if variadic and ref:
            ParserError.custom(variadic.pos, "Variadic parameters cannot be references")
        if self.lookahead(TokenType.COLON):
            type_ = self.parse_type()
        else:
            type_ = None
        return FunctionParam(name, ref, type_, bool(variadic), variadic.pos if variadic is not None else None)

    def _parse_function_signature(self) -> tuple[list[FunctionParam], Node | None]:
        params = self._parse_comma_separated(self._parse_function_param)
        for i, p in enumerate(params):
            if p.variadic and i != len(params) - 1:
                ParserError.custom(p.debug_variadic_pos, "Variadic function parameter must be last")
        if self.lookahead(TokenType.ARROW):
            type_ = self.parse_type()
        else:
            type_ = None
        return params, type_

    def _parse_function_declaration(self, require_name: bool) -> FunctionDeclaration:
        if self.lookahead(TokenType.LBRACK):
            attributes = set(self._parse_comma_separated(
                lambda: self.next(TokenType.ID).value, None, TokenType.RBRACK))
        else:
            attributes = set()
        if require_name:
            name = self.next(TokenType.ID).value
        else:
            if name_tok := self.lookahead(TokenType.ID):
                name = name_tok.value
            else:
                name = None
        params, result = self._parse_function_signature()
        self.next(TokenType.LBRACE)
        code = self.parse_block(True, True)
        return FunctionDeclaration(name, params, result, code, attributes)

    def _parse_capture(self) -> LambdaCapture:
        ref = bool(self.lookahead(TokenType.OPERATOR, "&"))
        name = self.next(TokenType.ID).value
        if self.lookahead(TokenType.ASSIGNMENT, "="):
            value = self.parse_value()
        else:
            value = None
        return LambdaCapture(name, ref, value)

    def _parse_tuple_value(self) -> tuple[Node, bool]:
        value = self.parse_value()
        unpack = bool(self.lookahead(TokenType.ELLIPSIS))
        return value, unpack

    def parse_atom(self) -> Node:
        tok = self.next()

        if tok.type == TokenType.KW_IF:
            const = bool(self.lookahead(TokenType.KW_CONST))
            cond = self.parse_value()
            code_if = self.parse_statement()
            pos = tok.pos + code_if.pos
            if self.lookahead(TokenType.KW_ELSE):
                code_else = self.parse_statement()
                pos = pos + code_else.pos
            else:
                code_else = None
            return IfNode(pos, const, cond, code_if, code_else)

        elif tok.type == TokenType.KW_FN:
            func = self._parse_function_declaration(False)
            return FunctionNode(tok.pos, func.name, func.params, func.result, func.code, func.attributes)

        elif tok.type == TokenType.KW_COMPTIME:
            old_is_comptime = self.is_comptime
            self.is_comptime = True
            value = self.parse_value()
            self.is_comptime = old_is_comptime
            return ComptimeNode(tok.pos + value.pos, value)

        elif tok.type == TokenType.KW_STRUCT:
            if name_tok := self.lookahead(TokenType.ID):
                name = name_tok.value
            else:
                name = None
            if self.lookahead(TokenType.COLON):
                parent = self.parse_type()
            else:
                parent = None

            self.next(TokenType.LBRACE)
            fields = []
            static_fields = []
            methods = []
            static_methods = []
            while not self.lookahead(TokenType.RBRACE):
                if self.lookahead(TokenType.KW_STATIC):
                    if var_tok := self.lookahead(TokenType.KW_LET | TokenType.KW_CONST):
                        const = var_tok.type == TokenType.KW_CONST
                        name_ = self.next(TokenType.ID).value
                        if self.lookahead(TokenType.COLON):
                            type_ = self.parse_type()
                        else:
                            type_ = None
                        self.next(TokenType.ASSIGNMENT, "=")
                        val = self.parse_value()
                        static_fields.append((SingleAssignmentTarget(const, name_, type_), val))
                        self.next(TokenType.SEMICOLON)

                    else:
                        self.next(TokenType.KW_FN)
                        static_methods.append(self._parse_function_declaration(True))

                elif self.lookahead(TokenType.KW_CONST):
                    self.next(TokenType.KW_FN)
                    methods.append((True, self._parse_function_declaration(True)))

                elif self.lookahead(TokenType.KW_LET):
                    name_ = self.next(TokenType.ID).value
                    self.next(TokenType.COLON)
                    type_ = self.parse_type()
                    fields.append(SingleAssignmentTarget(False, name_, type_))
                    self.next(TokenType.SEMICOLON)

                else:
                    self.next(TokenType.KW_FN)
                    methods.append((False, self._parse_function_declaration(True)))

            return StructNode(tok.pos, name, parent, fields, static_fields, methods, static_methods)

        elif tok.type == TokenType.KW_ENUM:
            if name_tok := self.lookahead(TokenType.ID):
                name = name_tok.value
            else:
                name = None
            options = self._parse_comma_separated(lambda: self.next(TokenType.ID).value,
                                                  TokenType.LBRACE, TokenType.RBRACE)
            option_set = set(options)
            if len(option_set) != len(options):
                for opt in option_set:
                    if options.count(opt) > 1:
                        ParserError.custom(tok.pos, f"Duplicate option in enum: '{opt}'")
            return EnumNode(tok.pos, name, options)

        elif tok.type == TokenType.KW_NAMESPACE:
            if name_tok := self.lookahead(TokenType.ID):
                name = name_tok.value
            else:
                name = None
            self.next(TokenType.LBRACE)
            code = self.parse_block(True, False)
            return NamespaceNode(tok.pos, name, code)

        elif tok.type == TokenType.LBRACE:
            return self.parse_block(True, True)

        elif tok.type == TokenType.LPAREN:
            if self.lookahead(TokenType.RPAREN):
                return TupleValueNode(tok.pos, [])
            val = self.parse_value()
            unpack = bool(self.lookahead(TokenType.ELLIPSIS))
            if self.lookahead(TokenType.COMMA):
                return TupleValueNode(tok.pos, [(val, unpack)] + self._parse_comma_separated(
                    self._parse_tuple_value, None))
            elif unpack:
                return TupleValueNode(tok.pos, [(val, unpack)])
            elif self.lookahead(TokenType.KW_FOR):
                target = self._parse_assignment_target(False)
                self.next(TokenType.KW_IN)
                iterable = self.parse_value()
                end = self.next(TokenType.RPAREN)
                return ComprehensionNode(tok.pos + end.pos, val, target, iterable)
            else:
                self.next(TokenType.RPAREN)
                return val

        elif tok.type == TokenType.INTEGER:
            return NumberValueNode(tok.pos, int(tok.value))

        elif tok.type == TokenType.FLOAT:
            return NumberValueNode(tok.pos, float(tok.value))

        elif tok.type == TokenType.STRING:
            return StringValueNode(tok.pos, tok.value)

        elif tok.type == TokenType.COLOR:
            return ColorValueNode(tok.pos, tok.value)

        elif tok.type == TokenType.ID:
            return VariableValueNode(tok.pos, tok.value)

        elif tok.type == TokenType.OPERATOR and tok.value == "|":
            params = self._parse_comma_separated(self._parse_function_param, None, TokenType.OPERATOR, "|")

            if self.lookahead(TokenType.LBRACK):
                captures = self._parse_comma_separated(self._parse_capture, None, TokenType.RBRACK)
            else:
                captures = []

            if self.lookahead(TokenType.ARROW):
                ret = self.parse_type()
            else:
                ret = None

            if self.lookahead(TokenType.LBRACE):
                code = self.parse_block(True, True)
            else:
                code = self.parse_value()

            return LambdaNode(tok.pos + code.pos, params, captures, ret, code)

        elif tok.type == TokenType.OPERATOR and tok.value == "||":
            if self.lookahead(TokenType.LBRACK):
                captures = self._parse_comma_separated(self._parse_capture, None, TokenType.RBRACK)
            else:
                captures = []

            if self.lookahead(TokenType.ARROW):
                ret = self.parse_type()
            else:
                ret = None

            if self.lookahead(TokenType.LBRACE):
                code = self.parse_block(True, True)
            else:
                code = self.parse_value()

            return LambdaNode(tok.pos + code.pos, [], captures, ret, code)

        elif tok.type == TokenType.HASH:
            name = self.next(TokenType.ID)
            if name.value == "def":
                macro_name = self.next(TokenType.ID)
                if macro_name.value in ("def", "undef", "import", "use", "static_assert"):
                    ParserError.custom(macro_name.pos, f"Reserved macro name: '{macro_name.value}'")
                self.macro_reg.create(self, macro_name)
                return NullValueNode(tok.pos + macro_name.pos)
            elif name.value == "undef":
                macro_name = self.next(TokenType.ID)
                self.macro_reg.delete(macro_name)
                return NullValueNode(tok.pos + macro_name.pos)
            elif self.macro_reg.has(name.value):
                return self.macro_reg.invoke(self, name,
                                             lambda tokens: Parser(tokens, self.macro_reg).parse_statement())

            return VariableValueNode(tok.pos + name.pos, tok.value + name.value)

        elif tok.type == TokenType.DOUBLE_COLON:
            name = self.next(TokenType.ID)
            return VariableValueNode(tok.pos + name.pos, tok.value + name.value)

        ParserError.unexpected_token(tok)
