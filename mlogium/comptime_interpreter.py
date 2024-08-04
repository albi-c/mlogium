from __future__ import annotations

from .node import *
from .error import InterpreterError
from .comptime_value import *
from .comptime_scope import ComptimeScopeStack


class ComptimeInterpreter(AstVisitor[CValue]):
    class Context(ComptimeInterpreterContext):
        interpreter: ComptimeInterpreter

        def __init__(self, interpreter: ComptimeInterpreter):
            self.interpreter = interpreter

        def error(self, msg: str):
            self.interpreter.error(msg)

    ctx: ComptimeInterpreterContext
    scope: ComptimeScopeStack

    def __init__(self):
        super().__init__()

        self.ctx = ComptimeInterpreter.Context(self)
        self.scope = ComptimeScopeStack()

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

    def _var_declare(self, name: str, value: CValue) -> CValue:
        if not self.scope.declare(name, value):
            self.error(f"Variable already defined: '{name}'")
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
        pass

    def visit_function_node(self, node: FunctionNode) -> CValue:
        pass

    def visit_comptime_function_node(self, node: ComptimeFunctionNode) -> CValue:
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
        pass

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
        pass

    def visit_tuple_value_node(self, node: TupleValueNode) -> CValue:
        values = []
        for n, u in node.values:
            if u:
                values += self.visit(n).unpack_req(self.ctx)
        return CValue.of_tuple(values)

    def visit_range_value_node(self, node: RangeValueNode) -> CValue:
        pass

    def visit_tuple_type_node(self, node: TupleTypeNode) -> CValue:
        pass
