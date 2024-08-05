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

    def _var_declare(self, name: str, value: CValue, type_: CType = None, const: bool = False) -> CValue:
        # TODO: constants, type checking
        if not self.scope.declare(name, value):
            self.error(f"Variable already defined: '{name}'")
        return value

    def _declare_target(self, target: AssignmentTarget, value: CValue) -> CValue:
        if isinstance(target, SingleAssignmentTarget):
            self._var_declare(target.name, value, target.type, target.const)

        elif isinstance(target, UnpackAssignmentTarget):
            values = value.unpack_req(self.ctx)
            if len(values) != len(target.values):
                self.error(
                    f"Value of type '{value.type}' unpacks into {len(values)} values, expected {len(target.values)}")
            for i, (dst, src) in enumerate(zip(target.values, values)):
                type_ = target.types[i] if target.types is not None else None
                self._var_declare(dst, src, type_, target.const)

        else:
            raise TypeError(f"Invalid assignment target: {target}")

        return value

    def _create_function(self, func: FunctionDeclaration) -> CValue:
        return FunctionCValue(func.name if func.name is not None else "_",
                              [],
                              None,
                              lambda params: self.interpret(func.code))

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

    def visit_function_node(self, node: FunctionNode) -> CValue:
        value = self._create_function(node)
        if node.name is not None:
            self._var_declare(node.name, value)
        return value

    def visit_lambda_node(self, node: LambdaNode) -> CValue:
        pass

    def visit_return_node(self, node: ReturnNode) -> CValue:
        pass

    def visit_struct_node(self, node: StructNode) -> CValue:
        pass

    def visit_if_node(self, node: IfNode) -> CValue:
        pass

    def visit_namespace_node(self, node: NamespaceNode) -> CValue:
        pass

    def visit_enum_node(self, node: EnumNode) -> CValue:
        pass

    def visit_while_node(self, node: WhileNode) -> CValue:
        pass

    def visit_for_node(self, node: ForNode) -> CValue:
        pass

    def visit_comprehension_node(self, node: ComprehensionNode) -> CValue:
        pass

    def visit_break_node(self, node: BreakNode) -> CValue:
        pass

    def visit_continue_node(self, node: ContinueNode) -> CValue:
        pass

    def visit_cast_node(self, node: CastNode) -> CValue:
        pass

    def visit_binary_op_node(self, node: BinaryOpNode) -> CValue:
        pass

    def visit_unary_op_node(self, node: UnaryOpNode) -> CValue:
        pass

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
        pass

    def visit_attribute_node(self, node: AttributeNode) -> CValue:
        pass

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
        pass

    def visit_tuple_type_node(self, node: TupleTypeNode) -> CValue:
        pass
