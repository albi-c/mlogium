from __future__ import annotations

from .node import *
from .value import *
from .instruction import Instruction
from .compilation_context import CompilationContext
from .error import CompilerError
from .scope import ScopeStack
from . import builtins


class Compiler(AstVisitor[Value]):
    class CompilationContext(CompilationContext):
        scope: ScopeStack
        compiler: Compiler

        def __init__(self, compiler: Compiler, scope: ScopeStack):
            super().__init__(scope)
            self.compiler = compiler

        def generate_node(self, node: Node) -> Value:
            return self.compiler.visit(node)

    ctx: CompilationContext
    scope: ScopeStack

    def __init__(self):
        super().__init__()

        self.scope = ScopeStack()
        self.ctx = Compiler.CompilationContext(self, self.scope)
        self.functions = {}

        self.scope.scopes.append(ScopeStack.Scope("<builtins>"))
        for name, value in builtins.BUILTINS.items():
            self.scope.scopes[-1].variables[name] = value
        self.scope.scopes.append(ScopeStack.Scope("<main>"))

    def emit(self, *instructions: Instruction):
        self.ctx.emit(*instructions)

    def _error(self, msg: str):
        CompilerError.custom(self.current_node.pos, msg)

    def _var_get(self, name: str) -> Value:
        if (var := self.scope.get(name)) is None:
            self._error(f"Variable not found: '{name}'")
        return var

    def _var_declare(self, name: str, type_: Type | None, value: Value, const: bool) -> Value:
        if type_ is None:
            type_ = value.type
        if (var := self.scope.declare(name, type_, const)) is None:
            self._error(f"Variable already exists: '{name}'")
        val = value.into(self.ctx, type_)
        if val is None:
            self._error(f"Incompatible types: {type_}, {value.type}")
        var.assign(self.ctx, val)
        return var

    def _var_declare_special(self, name: str, value: Value) -> Value:
        if not self.scope.declare_special(name, value):
            self._error(f"Variable already exists: '{name}'")
        return value

    def _var_assign(self, name: str, value: Value) -> Value:
        if (var := self.scope.assign(self.ctx, name, value)) is None:
            self._error(f"Variable not found: '{name}'")
        var.assign(self.ctx, value)
        return var

    def visit_block_node(self, node: BlockNode) -> Value:
        last_value = None
        for n in node.code:
            last_value = self.visit(n)
        if node.returns_last:
            return last_value
        return Value.null()

    def _declare_target(self, target: AssignmentTarget, value: Value, const: bool) -> Value:
        if isinstance(target, SingleAssignmentTarget):
            self._var_declare(target.name, target.type, value, const)

        elif isinstance(target, UnpackAssignmentTarget):
            values = value.unpack(self.ctx)
            if values is None:
                self._error(f"Value '{value}' of type {value.type} is not unpackable")
            if len(values) != len(target.values):
                self._error(f"Value '{value}' of type {value.type} unpacks into {len(values)} values, {len(target.values)} expected")
            for el, val in zip(target.values, values):
                if isinstance(el, str):
                    name = el
                    type_ = None
                else:
                    name, type_ = el

                self._var_declare(name, type_, val, const)

        else:
            raise TypeError("Invalid assignment target")

        return value

    def visit_declaration_node(self, node: DeclarationNode) -> Value:
        return self._declare_target(node.target, self.visit(node.value), False)

    def visit_function_node(self, node: FunctionNode) -> Value:
        return self._var_declare_special(
            node.name,
            Value(
                ConcreteFunctionType(node.name, node.type.named_params, node.type.ret, {
                    "code": node.code
                }),
                node.name,
                True
            )
        )

    def visit_lambda_node(self, node: LambdaNode) -> Value:
        pass

    def visit_return_node(self, node: ReturnNode) -> Value:
        if (func := self.scope.get_function()) is None:
            self._error(f"Return must be in a function")

        val = self.visit(node.value)
        Value.variable(ABI.return_value(func), val.type).assign(self.ctx, val)
        self.emit(Instruction.jump_always(ABI.function_end(func)))

        return Value.null()

    def visit_struct_node(self, node: StructNode) -> Value:
        pass

    def visit_if_node(self, node: IfNode) -> Value:
        pass

    def visit_enum_node(self, node: EnumNode) -> Value:
        pass

    def visit_while_node(self, node: WhileNode) -> Value:
        pass

    def visit_for_node(self, node: ForNode) -> Value:
        pass

    def visit_break_node(self, node: BreakNode) -> Value:
        pass

    def visit_continue_node(self, node: ContinueNode) -> Value:
        pass

    def visit_binary_op_node(self, node: BinaryOpNode) -> Value:
        pass

    def visit_unary_op_node(self, node: UnaryOpNode) -> Value:
        pass

    def visit_call_node(self, node: CallNode) -> Value:
        func = self.visit(node.value)
        if not func.callable():
            self._error(f"Not callable: '{func}'")

        param_types = func.params()
        if len(param_types) != len(node.params):
            self._error(f"Invalid number of parameters to function ({len(node.params)} provided, {len(param_types)} expected)")
        params = []
        for type_, param in zip(param_types, node.params):
            value = self.visit(param)
            if (val := value.into(self.ctx, type_)) is None:
                self._error(f"Incompatible types: {type_}, {value.type}")
            params.append(val)

        return func.call(self.ctx, params)

    def visit_index_node(self, node: IndexNode) -> Value:
        pass

    def visit_attribute_node(self, node: AttributeNode) -> Value:
        pass

    def visit_number_value_node(self, node: NumberValueNode) -> Value:
        return Value.number(node.value)

    def visit_string_value_node(self, node: StringValueNode) -> Value:
        return Value.string(f"\"{node.value}\"")

    def visit_variable_value_node(self, node: VariableValueNode) -> Value:
        return self._var_get(node.name)

    def visit_tuple_value_node(self, node: TupleValueNode) -> Value:
        return Value.tuple(self.ctx, [self.visit(val) for val in node.values])

    def visit_range_value_node(self, node: RangeValueNode) -> Value:
        pass
