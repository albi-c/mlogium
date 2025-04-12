from __future__ import annotations

from .node import *
from .error import InterpreterError
from .comptime_value import *
from .comptime_scope import ComptimeScopeStack


BUILTIN_TYPES = [
    ("num", NumberCType()),
    ("str", StringCType()),
    ("Opaque", OpaqueCType()),
    ("Range", RangeCType()),
    ("RangeWithStep", RangeWithStepCType())
]


def make_builtins() -> dict[str, VariableCLValue]:
    builtins = {}

    for name, type_ in BUILTIN_TYPES:
        builtins[name] = VariableCLValue(TypeCValue(type_), True)

    builtins["Tuple"] = VariableCLValue(TupleSourceCValue(), True)
    builtins["Block"] = VariableCLValue(BlockSourceCValue(), True)

    builtins["debug"] = VariableCLValue(FunctionCValue("name", [None], ["value"], NullCType(),
                                                       lambda p: (print(p[0]), NullCValue())[1]), True)

    builtins["true"] = VariableCLValue(BooleanCValue(True), True)
    builtins["false"] = VariableCLValue(BooleanCValue(False), True)
    builtins["null"] = VariableCLValue(NullCValue(), True)

    return builtins


class ComptimeInterpreter(AstVisitor[BaseCValue]):
    class Context(ComptimeInterpreterContext):
        interpreter: ComptimeInterpreter
        tmp_num_provider: Callable[[], int]

        def __init__(self, interpreter: ComptimeInterpreter, tmp_num_provider: Callable[[], int], note_context: ErrorContext):
            super().__init__(interpreter.scope, note_context)

            self.interpreter = interpreter
            self.tmp_num_provider = tmp_num_provider

        def error(self, msg: str, pos: Position | None = None):
            self.interpreter.error(msg, pos)

        def exc_guard[T](self, func: Callable[[], T]) -> T:
            return self.interpreter.exc_guard(func)

        def interpret(self, node: Node) -> BaseCValue:
            return self.interpreter.visit(node)

        def tmp_num(self) -> int:
            return self.tmp_num_provider()

    class Break(Exception):
        pos: Position

        def __init__(self, pos: Position):
            self.pos = pos

    class Continue(Exception):
        pos: Position

        def __init__(self, pos: Position):
            self.pos = pos

    scope: ComptimeScopeStack
    ctx: ComptimeInterpreterContext

    def __init__(self, tmp_num_provider: Callable[[], int], note_context: ErrorContext):
        super().__init__()

        self.scope = ComptimeScopeStack()
        self.ctx = ComptimeInterpreter.Context(self, tmp_num_provider, note_context)

        self.scope.push("<builtins>", make_builtins())
        self.scope.push("<main>")

    def exc_guard[T](self, func: Callable[[], T]) -> T:
        try:
            return func()
        except Return as e:
            self.error("Return statement used outside of function", e.pos)
        except self.Break as e:
            self.error("Break statement used outside of function", e.pos)
        except self.Continue as e:
            self.error("Continue statement used outside of function", e.pos)
        except KeyboardInterrupt:
            self.error("Comptime execution stopped", self.current_pos)

    def interpret(self, node: Node) -> BaseCValue:
        return self.exc_guard(lambda: self.visit(node))

    def error(self, msg: str, pos: Position = None):
        InterpreterError.custom(self.current_pos if pos is None else pos, msg)

    def _var_get(self, name: str) -> BaseCValue:
        return self._var_capture(name, True)

    def _var_set(self, name: str, value: CValue):
        self._var_capture(name, True).assign(self.ctx, value)

    def _var_declare(self, name: str, value: CValue, type_: CType = None, const: bool = False) -> CValue:
        if type_ is not None:
            value = value.into_req(self.ctx, type_)
        if not self.scope.declare(name, value, const):
            self.error(f"Variable already defined: '{name}'")
        return value

    def _var_capture(self, name: str, ref: bool) -> CLValue:
        if (var := self.scope.capture(name, ref)) is None:
            self.error(f"Variable not found: '{name}'")
        return var

    def _resolve_type_opt(self, node: Node | None) -> CType | None:
        if node is None:
            return None

        return self.visit(node).deref().get_wrapped_type(self.ctx)

    def _resolve_type(self, node: Node) -> CType:
        return self.visit(node).deref().get_wrapped_type_req(self.ctx)

    def _declare_target_inner(self, target: AssignmentTarget, value: CValue, const_override: bool) -> BaseCValue:
        if isinstance(target, SingleAssignmentTarget):
            self._var_declare(target.name, value, self._resolve_type_opt(target.type), target.const | const_override)

        elif isinstance(target, UnpackAssignmentTarget):
            values = value.unpack_req(self.ctx)
            if len(values) != len(target.values):
                self.error(
                    f"Value of type '{value.type}' unpacks into {len(values)} values, expected {len(target.values)}")
            for i, (dst, src) in enumerate(zip(target.values, values)):
                self._declare_target_inner(dst, src, target.const | const_override)

        else:
            raise TypeError(f"Invalid assignment target: {target}")

        return value

    def _declare_target(self, target: AssignmentTarget, value: CValue) -> BaseCValue:
        return self._declare_target_inner(target, value, target.const)

    def visit_block_node(self, node: BlockNode) -> BaseCValue:
        last_value = CValue.null()
        for n in node.code:
            last_value = self.visit(n)
        if node.returns_last:
            return last_value
        return CValue.null()

    def visit_declaration_node(self, node: DeclarationNode) -> BaseCValue:
        return self._declare_target(node.target, self.visit(node.value).deref())

    def visit_comptime_node(self, node: ComptimeNode) -> BaseCValue:
        return self.visit(node.value)

    def _resolve_function_params(self, param_data: list[FunctionParam]) -> tuple[list[CType | None], list[str]]:
        params = []
        param_names = []
        for param in param_data:
            if param.variadic:
                self.error(f"Variadic parameters are not supported", param.debug_variadic_pos)
            if param.reference:
                self.error(f"Reference parameters are not supported")
            params.append(self._resolve_type_opt(param.type))
            param_names.append(param.name)
        return params, param_names

    def visit_function_node(self, node: FunctionNode) -> BaseCValue:
        params, param_names = self._resolve_function_params(node.params)
        value = FunctionCValue(node.name if node.name is not None else "_", params, param_names,
                               self._resolve_type_opt(node.result), lambda _: self.visit(node.code).deref())
        if node.name is not None:
            self._var_declare(node.name, value)
        return value

    def visit_lambda_node(self, node: LambdaNode) -> BaseCValue:
        params, param_names = self._resolve_function_params(node.params)
        value = LambdaCValue(
            params, param_names,
            {
                capture.name: VariableCLValue(self.visit(capture.value).deref(), False)
                if capture.value is not None else
                self._var_capture(capture.name, capture.reference)
                for capture in node.captures
            },
            self._resolve_type_opt(node.result),
            lambda _: self.visit(node.code).deref()
        )
        return value

    def visit_return_node(self, node: ReturnNode) -> BaseCValue:
        raise Return(node.pos, self.visit(node.value).deref() if node.value is not None else None)

    def visit_struct_node(self, node: StructNode) -> BaseCValue:
        raise NotImplementedError

    def visit_if_node(self, node: IfNode) -> BaseCValue:
        if node.const:
            self.error(f"If statement is always constant in comptime context")

        if self.visit(node.cond).deref().is_true_req(self.ctx):
            return self.visit(node.code_if)

        elif node.code_else is not None:
            return self.visit(node.code_else)

        else:
            return CValue.null()

    def visit_namespace_node(self, node: NamespaceNode) -> BaseCValue:
        raise NotImplementedError

    def visit_enum_node(self, node: EnumNode) -> BaseCValue:
        raise NotImplementedError

    def visit_while_node(self, node: WhileNode) -> BaseCValue:
        while self.visit(node.cond).deref().is_true_req(self.ctx):
            try:
                self.visit(node.code)
            except self.Break:
                break
            except self.Continue:
                continue

        return CValue.null()

    def visit_for_node(self, node: ForNode) -> BaseCValue:
        value = self.visit(node.iterable).deref()
        if not value.iterable():
            self.error(f"Value of type '{value.type}' is not iterable", node.iterable.pos)

        for val in value.iterate(self.ctx):
            with self.scope(self.ctx.tmp()):
                self._declare_target(node.target, val)
                try:
                    self.visit(node.code)
                except self.Break:
                    break
                except self.Continue:
                    continue

        return CValue.null()

    def visit_unroll_node(self, node: UnrollNode) -> BaseCValue:
        self.error("Unrolled loops are not supported in comptime")
        return CValue.null()

    def visit_comprehension_node(self, node: ComprehensionNode) -> BaseCValue:
        value = self.visit(node.iterable).deref()
        if not value.iterable():
            self.error(f"Value of type '{value.type}' is not iterable", node.iterable.pos)

        values = []
        for val in value.iterate(self.ctx):
            with self.scope(self.ctx.tmp()):
                self._declare_target(node.target, val)
                values.append(self.visit(node.expr).deref())

        return CValue.of_tuple(values)

    def visit_break_node(self, node: BreakNode) -> BaseCValue:
        raise self.Break(node.pos)

    def visit_continue_node(self, node: ContinueNode) -> BaseCValue:
        raise self.Continue(node.pos)

    def visit_cast_node(self, node: CastNode) -> BaseCValue:
        return self.visit(node.value).deref().into_req(self.ctx, self._resolve_type(node.type))

    def _do_binary_op(self, left: CValue, op: str, right: CValue) -> CValue:
        if (val := left.binary_op(self.ctx, op, right)) is None:
            if (val_r := right.binary_op_r(self.ctx, left, op)) is None:
                self.error(f"Operator {op} is not supported for types '{left.type}' and '{right.type}'")
            return val_r
        return val

    def visit_binary_op_node(self, node: BinaryOpNode) -> BaseCValue:
        left = self.visit(node.left)
        right = self.visit(node.right).deref()

        if node.op in ("=", ":="):
            if not left.assign(self.ctx, right):
                self.error(f"Assignment to constant of type '{left.deref().type}'")
            return right

        elif node.op.endswith("=") and node.op not in (">=", "<=", "=="):
            if (result := left.deref().binary_op(self.ctx, node.op[:-1], right)) is None:
                self.error(f"Operator {node.op} is not supported for values of types '{left.deref().type}' and '{right.type}'")
            if not left.assign(self.ctx, result):
                self.error(f"Assignment to constant of type '{left.deref().type}'")
            return CValue.null()

        return left.deref().do_binary_op(self.ctx, node.op, right)

    def visit_unary_op_node(self, node: UnaryOpNode) -> BaseCValue:
        value = self.visit(node.value).deref()
        if (val := value.unary_op(self.ctx, node.op)) is None:
            self.error(f"Operator {node.op} is not supported for value of type '{value.type}'")
        return val

    def visit_call_node(self, node: CallNode) -> BaseCValue:
        func = self.visit(node.value).deref()
        if not func.callable():
            self.error(f"Value of type '{func.type}' is not callable")

        params = []
        for p, u in node.params:
            if u:
                params += self.visit(p).deref().unpack_req(self.ctx)
            else:
                params.append(self.visit(p).deref())

        if (exp := func.callable_with([p.type for p in params])) is not None:
            if len(params) == len(exp):
                converted = []
                for param, type_ in zip(params, exp):
                    converted.append(param.into_req(self.ctx, type_))
                if func.callable_with([p.type for p in converted]) is None:
                    return func.call(self.ctx, converted)

            self.error(f"Cannot call function with parameters of types ({', '.join(f'\'{p.type}\'' for p in params)}), \
expected ({', '.join(f'\'{t}\'' for t in exp)})")

        return func.call(self.ctx, params)

    def visit_index_node(self, node: IndexNode) -> BaseCValue:
        value = self.visit(node.value).deref()
        return value.index_req(self.ctx, [self.visit(i).deref() for i in node.indices])

    def visit_attribute_node(self, node: AttributeNode) -> BaseCValue:
        return self.visit(node.value).deref().getattr_req(self.ctx, node.attr, node.static)

    def visit_number_value_node(self, node: NumberValueNode) -> BaseCValue:
        return CValue.of_number(node.value)

    def visit_string_value_node(self, node: StringValueNode) -> BaseCValue:
        return CValue.of_string(node.value)

    def visit_color_value_node(self, node: ColorValueNode) -> BaseCValue:
        self.error(f"Color values are not supported in comptime contexts")
        return CValue.null()

    def visit_variable_value_node(self, node: VariableValueNode) -> BaseCValue:
        return self._var_get(node.name)

    def visit_tuple_value_node(self, node: TupleValueNode) -> BaseCValue:
        values = []
        for n, u in node.values:
            if u:
                values += self.visit(n).deref().unpack_req(self.ctx)
            else:
                values.append(self.visit(n).deref())
        return CValue.of_tuple(values)

    def visit_range_value_node(self, node: RangeValueNode) -> BaseCValue:
        start = self.visit(node.start).deref().into_req(self.ctx, NumberCType())
        end = self.visit(node.end).deref().into_req(self.ctx, NumberCType())
        assert isinstance(start, NumberCValue)
        assert isinstance(end, NumberCValue)
        if node.step is not None:
            step = self.visit(node.step).deref().into_req(self.ctx, NumberCType())
            assert isinstance(step, NumberCValue)
            return RangeWithStepCValue(start.as_int_or_float(), end.as_int_or_float(), step.as_int_or_float())
        else:
            return RangeCValue(start.as_int_or_float(), end.as_int_or_float())

    def visit_tuple_type_node(self, node: TupleTypeNode) -> BaseCValue:
        return TypeCValue(TupleCType([self._resolve_type(n) for n in node.types]))

    def visit_function_type_node(self, node: FunctionTypeNode) -> BaseCValue:
        return TypeCValue(FunctionCType([self._resolve_type_opt(n) for n in node.params],
                                        self._resolve_type_opt(node.result)))

    def visit_null_value_node(self, node: NullValueNode) -> BaseCValue:
        return CValue.null()
