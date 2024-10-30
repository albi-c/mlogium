from __future__ import annotations

from .node import *
from .error import InterpreterError
from .comptime_value import *
from .comptime_scope import ComptimeScopeStack


class ComptimeInterpreter(AstVisitor[CValue]):
    class Context(ComptimeInterpreterContext):
        interpreter: ComptimeInterpreter
        tmp_num_provider: Callable[[], int]

        def __init__(self, interpreter: ComptimeInterpreter, tmp_num_provider: Callable[[], int]):
            super().__init__(interpreter.scope)

            self.interpreter = interpreter
            self.tmp_num_provider = tmp_num_provider

        def error(self, msg: str):
            self.interpreter.error(msg)

        def interpret(self, node: Node) -> CValue:
            return self.interpreter.visit(node)

        def tmp_num(self) -> int:
            return self.tmp_num_provider()

    class Return(Exception):
        value: CValue | None

        def __init__(self, value: CValue | None):
            self.value = value

    class Break(Exception):
        pass

    class Continue(Exception):
        pass

    scope: ComptimeScopeStack
    ctx: ComptimeInterpreterContext

    def __init__(self, tmp_num_provider: Callable[[], int]):
        super().__init__()

        self.scope = ComptimeScopeStack()
        self.ctx = ComptimeInterpreter.Context(self, tmp_num_provider)

        # TODO: builtins
        self.scope.push("<builtins>", {})
        self.scope.push("<main>")

    def interpret(self, node: Node) -> CValue:
        # TODO: catch return, break and continue
        return self.visit(node)

    def error(self, msg: str, pos: Position = None):
        InterpreterError.custom(self.current_pos if pos is None else pos, msg)

    def _var_get(self, name: str) -> CValue:
        if (var := self.scope.get(name)) is None:
            self.error(f"Variable not found: '{name}'")
        return var

    def _var_set(self, name: str, value: CValue):
        var = self._var_capture(name, True)
        var.set(value.into_req(self.ctx, var.get().type))

    def _var_declare(self, name: str, value: CValue, type_: CType = None, const: bool = False) -> CValue:
        # TODO: constants
        if type_ is not None:
            value = value.into_req(self.ctx, type_)
        if not self.scope.declare(name, value):
            self.error(f"Variable already defined: '{name}'")
        return value

    def _var_capture(self, name: str, ref: bool) -> Cell[CValue]:
        if (var := self.scope.capture(name, ref)) is None:
            self.error(f"Variable not found: '{name}'")
        return var

    def _resolve_type_opt(self, node: Node | None) -> CType | None:
        if node is None:
            return None

        return self.visit(node).get_wrapped_type(self.ctx)

    def _resolve_type(self, node: Node) -> CType:
        return self.visit(node).get_wrapped_type_req(self.ctx)

    def _declare_target(self, target: AssignmentTarget, value: CValue) -> CValue:
        if isinstance(target, SingleAssignmentTarget):
            self._var_declare(target.name, value, self._resolve_type_opt(target.type), target.const)

        elif isinstance(target, UnpackAssignmentTarget):
            values = value.unpack_req(self.ctx)
            if len(values) != len(target.values):
                self.error(
                    f"Value of type '{value.type}' unpacks into {len(values)} values, expected {len(target.values)}")
            for i, (dst, src) in enumerate(zip(target.values, values)):
                type_ = self._resolve_type_opt(target.types[i] if target.types is not None else None)
                self._var_declare(dst, src, type_, target.const)

        else:
            raise TypeError(f"Invalid assignment target: {target}")

        return value

    def visit_block_node(self, node: BlockNode) -> CValue:
        last_value = CValue.null()
        for n in node.code:
            last_value = self.visit(n)
        if node.returns_last:
            return last_value
        return CValue.null()

    def visit_declaration_node(self, node: DeclarationNode) -> CValue:
        return self._declare_target(node.target, self.visit(node.value))

    def visit_comptime_node(self, node: ComptimeNode) -> CValue:
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

    def visit_function_node(self, node: FunctionNode) -> CValue:
        params, param_names = self._resolve_function_params(node.params)
        value = FunctionCValue(node.name if node.name is not None else "_", params, param_names,
                               self._resolve_type_opt(node.result), lambda _: self.interpret(node.code))
        if node.name is not None:
            self._var_declare(node.name, value)
        return value

    def visit_lambda_node(self, node: LambdaNode) -> CValue:
        params, param_names = self._resolve_function_params(node.params)
        value = LambdaCValue(
            params, param_names,
            {
                capture.name: Cell(self.visit(capture.value))
                if capture.value is not None else
                self._var_capture(capture.name, capture.reference)
                for capture in node.captures
            },
            self._resolve_type_opt(node.result),
            lambda _: self.interpret(node.code)
        )
        return value

    def visit_return_node(self, node: ReturnNode) -> CValue:
        raise self.Return(self.visit(node.value) if node.value is not None else None)

    def visit_struct_node(self, node: StructNode) -> CValue:
        raise NotImplementedError

    def visit_if_node(self, node: IfNode) -> CValue:
        if node.const:
            self.error(f"If statement is always constant in comptime context")

        if self.visit(node.cond).is_true_req(self.ctx):
            return self.visit(node.code_if)

        elif node.code_else is not None:
            return self.visit(node.code_else)

        else:
            return CValue.null()

    def visit_namespace_node(self, node: NamespaceNode) -> CValue:
        raise NotImplementedError

    def visit_enum_node(self, node: EnumNode) -> CValue:
        raise NotImplementedError

    def visit_while_node(self, node: WhileNode) -> CValue:
        while self.visit(node.cond).is_true_req(self.ctx):
            try:
                self.visit(node.code)
            except self.Break:
                break
            except self.Continue:
                continue

        return CValue.null()

    def visit_for_node(self, node: ForNode) -> CValue:
        value = self.visit(node.iterable)
        if not value.iterable():
            self.error(f"Value of type {value.type} is not iterable", node.iterable.pos)

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

    def visit_comprehension_node(self, node: ComprehensionNode) -> CValue:
        value = self.visit(node.iterable)
        if not value.iterable():
            self.error(f"Value of type {value.type} is not iterable", node.iterable.pos)

        values = []
        for val in value.iterate(self.ctx):
            with self.scope(self.ctx.tmp()):
                self._declare_target(node.target, val)
                values.append(self.visit(node.expr))

        return CValue.of_tuple(values)

    def visit_break_node(self, node: BreakNode) -> CValue:
        raise self.Break()

    def visit_continue_node(self, node: ContinueNode) -> CValue:
        raise self.Continue()

    def visit_cast_node(self, node: CastNode) -> CValue:
        return self.visit(node.value).into_req(self.ctx, self._resolve_type(node.type))

    def visit_binary_op_node(self, node: BinaryOpNode) -> CValue:
        # TODO
        raise NotImplementedError

    def visit_unary_op_node(self, node: UnaryOpNode) -> CValue:
        # TODO
        raise NotImplementedError

    def visit_call_node(self, node: CallNode) -> CValue:
        func = self.visit(node.value)
        if not func.callable():
            self.error(f"Value of type '{func.type}' is not callable")

        params = []
        for p, u in node.params:
            if u:
                params += self.visit(p).unpack_req(self.ctx)
            else:
                params.append(self.visit(p))

        if (exp := func.callable_with([p.type for p in params])) is not None:
            self.error(f"Cannot call function with parameters of types ({', '.join(f'\'{p.type}\'' for p in params)}), \
expected ({', '.join(f'\'{t}\'' for t in exp)})")

        return func.call(self.ctx, params)

    def visit_index_node(self, node: IndexNode) -> CValue:
        # TODO
        raise NotImplementedError

    def visit_attribute_node(self, node: AttributeNode) -> CValue:
        # TODO
        raise NotImplementedError

    def visit_number_value_node(self, node: NumberValueNode) -> CValue:
        return CValue.of_number(node.value)

    def visit_string_value_node(self, node: StringValueNode) -> CValue:
        return CValue.of_string(node.value)

    def visit_color_value_node(self, node: ColorValueNode) -> CValue:
        self.error(f"Colors values are not supported in comptime contexts")
        return CValue.null()

    def visit_variable_value_node(self, node: VariableValueNode) -> CValue:
        return self._var_get(node.name)

    def visit_tuple_value_node(self, node: TupleValueNode) -> CValue:
        values = []
        for n, u in node.values:
            if u:
                values += self.visit(n).unpack_req(self.ctx)
            else:
                values.append(self.visit(n))
        return CValue.of_tuple(values)

    def visit_range_value_node(self, node: RangeValueNode) -> CValue:
        raise NotImplementedError

    def visit_tuple_type_node(self, node: TupleTypeNode) -> CValue:
        return TypeCValue(TupleCType([self._resolve_type(n) for n in node.types]))

    def visit_null_value_node(self, node: NullValueNode) -> CValue:
        return CValue.null()
