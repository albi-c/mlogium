from __future__ import annotations

from .node import *
from .value import *
from .compilation_context import CompilationContext
from .instruction import Instruction, InstructionInstance
from .error import CompilerError
from .scope import ScopeStack
from .builtins import construct_builtins


class Compiler(AstVisitor[Value]):
    class CompilationContext(CompilationContext):
        compiler: Compiler

        def __init__(self, compiler: Compiler):
            super().__init__(compiler.scope)

            self.compiler = compiler

        def generate_node(self, node: Node) -> Value:
            return self.compiler.visit(node)

        def error(self, msg: str):
            self.compiler.error(msg)

        def current_pos(self) -> Position:
            return self.compiler.current_pos

    scope: ScopeStack
    ctx: CompilationContext

    def __init__(self):
        super().__init__()

        self.scope = ScopeStack()
        self.ctx = Compiler.CompilationContext(self)

        self.scope.push("<builtins>", construct_builtins())
        self.scope.push("<main>")

    def compile(self, node: Node):
        self.visit(node)

    def emit(self, *instructions: InstructionInstance):
        self.ctx.emit(*instructions)

    def error(self, msg: str, pos: Position = None):
        CompilerError.custom(self.current_pos if pos is None else pos, msg)

    def resolve_type(self, value: Node | Value) -> Type:
        type_ = self.visit(value) if isinstance(value, Node) else value
        return type_.type.wrapped_type(self.ctx)

    def resolve_type_opt(self, value: Node | Value | Node) -> Type | None:
        if value is None:
            return None
        return self.resolve_type(value)

    def _var_get(self, name: str) -> Value:
        if (var := self.scope.get(name)) is None:
            self.error(f"Value not found: '{name}'")
        return var

    def _var_declare(self, name: str, type_: Type | Node | None, value: Value, const: bool) -> Value:
        if type_ is None:
            type_ = value.type
        if isinstance(type_, Node):
            type_ = self.resolve_type(type_)
        if (var := self.scope.declare(name, type_, const)) is None:
            self.error(f"Already defined: '{name}'")
        var.assign(self.ctx, value)
        return var

    def _var_declare_special(self, name: str, value: Value) -> Value:
        if not self.scope.declare_special(name, value):
            self.error(f"Already defined: '{name}'")
        return value

    def _declare_target(self, target: AssignmentTarget, value: Value) -> Value:
        if isinstance(target, SingleAssignmentTarget):
            self._var_declare(target.name, target.type, value, target.const)

        elif isinstance(target, UnpackAssignmentTarget):
            values = value.unpack_req(self.ctx)
            if len(values) != len(target.values):
                self.error(
                    f"Value of type '{value.type}' unpacks into {len(values)} values, expected {len(target.values)}")
            for i, (dst, src) in enumerate(zip(target.values, values)):
                type_ = target.types[i] if target.types is not None else None
                self._var_declare(dst, type_, src, target.const)

        else:
            raise TypeError(f"Invalid assignment target: {target}")

        return value

    def visit_block_node(self, node: BlockNode) -> Value:
        # TODO: fix scopes
        last_value = Value.null()
        for n in node.code:
            last_value = self.visit(n)
        if node.returns_last:
            return last_value
        return Value.null()

    def visit_declaration_node(self, node: DeclarationNode) -> Value:
        return self._declare_target(node.target, self.visit(node.value))

    def _build_function(self, func: FunctionDeclaration) -> Value:
        return Value(FunctionType(func.name if func.name is not None else self.ctx.tmp(),
                                  [FunctionType.Param(p.name, p.reference, self.resolve_type_opt(p.type))
                                   for p in func.params],
                                  self.resolve_type_opt(func.result),
                                  func.code), "")

    def visit_function_node(self, node: FunctionNode) -> Value:
        value = self._build_function(node)
        if node.name is not None:
            self._var_declare_special(node.name, value)
        return value

    def visit_lambda_node(self, node: LambdaNode) -> Value:
        captures = []
        for capture in node.captures:
            if capture.reference:
                captures.append(LambdaType.Capture(capture.name, self.visit(
                    capture.value) if capture.value else self._var_get(capture.name), f"&{capture.name}"))
            else:
                val = self.visit(capture.value) if capture.value else self._var_get(capture.name)
                copied = Value(val.type, self.ctx.tmp(), False)
                copied.assign(self.ctx, val)
                captures.append(LambdaType.Capture(capture.name, copied, capture.name))
        value = Value(LambdaType(f"$lambda_{self.ctx.tmp_num()}",
                                 [FunctionType.Param(p.name, p.reference, self.resolve_type_opt(p.type))
                                  for p in node.params],
                                 captures,
                                 self.resolve_type_opt(node.result),
                                 node.code), "")
        return value

    def visit_return_node(self, node: ReturnNode) -> Value:
        if (func := self.scope.get_function()) is None:
            self.error(f"Return statement must be used inside a function")
        if node.value is not None:
            value = self.visit(node.value)
            Value(value.type, ABI.return_value(func), False).assign(self.ctx, value)
        self.emit(Instruction.jump_always(ABI.function_end(func)))
        return Value.null()

    def visit_struct_node(self, node: StructNode) -> Value:
        name = node.name if node.name is not None else self.ctx.tmp()

        parent = self.resolve_type_opt(node.parent)
        if parent is not None:
            # TODO
            self.error(f"Struct inheritance is not yet supported")

        fields = []
        static_fields = {}
        methods = {}
        static_methods = {}

        for field in node.fields:
            assert not field.const
            assert field.type is not None
            fields.append((field.name, self.resolve_type(field.type)))

        for target, value_ in node.static_fields:
            value = self.visit(value_)
            val = Value(value.type, ABI.static_attribute(name, target.name), False, const_on_write=target.const)
            val.assign(self.ctx, value)
            static_fields[target.name] = val

        for const, method in node.methods:
            assert method.name is not None
            methods[method.name] = (
                const,
                StructMethodData(
                    method.name,
                    [FunctionType.Param(p.name, p.reference, self.resolve_type_opt(p.type))
                     for p in method.params],
                    self.resolve_type_opt(method.result),
                    method.code
                )
            )

        for method in node.static_methods:
            assert method.name is not None
            static_methods[method.name] = StructMethodData(method.name,
                                                           [FunctionType.Param(p.name, p.reference,
                                                                               self.resolve_type_opt(p.type))
                                                            for p in method.params],
                                                           self.resolve_type_opt(method.result),
                                                           method.code)

        value = Value(StructBaseType(node.name, fields, static_fields, methods, static_methods), "")
        if node.name is not None:
            self._var_declare_special(node.name, value)
        return value

    def visit_if_node(self, node: IfNode) -> Value:
        cond_value = self.visit(node.cond).to_condition_req(self.ctx)

        if node.const:
            # TODO
            self.error(f"Constant if is not yet supported")

        end_true_branch = self.ctx.tmp()
        end_false_branch = "" if node.code_else is None else self.ctx.tmp()
        self.emit(Instruction.jump(end_true_branch, "equal", cond_value, "0"))
        with self.scope(end_true_branch):
            true_value = self.visit(node.code_if)
            result = Value(true_value.type, self.ctx.tmp(), False)
            result.assign(self.ctx, true_value)
            if end_false_branch:
                self.emit(Instruction.jump_always(end_false_branch))
        self.emit(Instruction.label(end_true_branch))
        if end_false_branch:
            with self.scope(end_false_branch):
                code_else = node.code_else
                assert isinstance(code_else, Node)
                result.assign(self.ctx, self.visit(code_else))
            self.emit(Instruction.label(end_false_branch))

        return result

    def visit_namespace_node(self, node: NamespaceNode) -> Value:
        name = node.name if node.name is not None else self.ctx.tmp()
        with self.scope(f"$namespace_{name}:{self.ctx.tmp_num()}"):
            self.visit(node.code)
            variables = self.scope.scopes[-1].variables.copy()
        value = Value(NamespaceType(node.name, variables), "")
        if node.name is not None:
            self._var_declare_special(node.name, value)
        return value

    def visit_enum_node(self, node: EnumNode) -> Value:
        value = Value(EnumBaseType(node.name if node.name is not None else self.ctx.tmp(),
                                   {opt: i for i, opt in enumerate(node.options)}), "")
        if node.name is not None:
            self._var_declare_special(node.name, value)
        return value

    def visit_while_node(self, node: WhileNode) -> Value:
        name = self.ctx.tmp()

        self.emit(Instruction.label(name + "_continue"))

        cond_value = self.visit(node.cond).to_condition_req(self.ctx)

        self.emit(Instruction.jump(name + "_break", "equal", cond_value, "0"))
        with self.scope.loop(name):
            self.visit(node.code)
        self.emit(
            Instruction.jump_always(name + "_continue"),
            Instruction.label(name + "_break")
        )
        return Value.null()

    def visit_for_node(self, node: ForNode) -> Value:
        pass

    def visit_comprehension_node(self, node: ComprehensionNode) -> Value:
        pass

    def visit_break_node(self, node: BreakNode) -> Value:
        if (name := self.scope.get_loop()) is None:
            self.error(f"Break statement must be in a loop")
        self.emit(Instruction.jump_always(name + "_break"))
        return Value.null()

    def visit_continue_node(self, node: ContinueNode) -> Value:
        if (name := self.scope.get_loop()) is None:
            self.error(f"Continue statement must be in a loop")
        self.emit(Instruction.jump_always(name + "_continue"))
        return Value.null()

    def visit_cast_node(self, node: CastNode) -> Value:
        return self.visit(node.value).into_req(self.ctx, self.resolve_type(node.type))

    def visit_binary_op_node(self, node: BinaryOpNode) -> Value:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if (result := left.binary_op(self.ctx, node.op, right)) is None:
            self.error(
                f"Operator '{node.op}' is not supported between values of types '{left.type}' and '{right.type}'")
        return result

    def visit_unary_op_node(self, node: UnaryOpNode) -> Value:
        value = self.visit(node.value)
        if (result := value.unary_op(self.ctx, node.op)) is None:
            self.error(f"Operator '{node.op}' is not supported for value of type '{value.type}'")
        return result

    def visit_call_node(self, node: CallNode) -> Value:
        func = self.visit(node.value)
        if not func.callable():
            self.error(f"Value of type '{func.type}' is not callable")

        param_types = func.call_with_suggestion()

        unpacked_params = []
        for param, unpack in node.params:
            if unpack:
                unpacked_params += self.visit(param).unpack_req(self.ctx)

            else:
                bottom_scope = None
                if param_types is not None and len(unpacked_params) < len(param_types):
                    type_ = param_types[len(unpacked_params)]
                    if type_ is not None:
                        bottom_scope = type_.bottom_scope()

                with self.scope.bottom("<enum>", bottom_scope):
                    unpacked_params.append(self.visit(param))

        if not func.callable_with([v.type for v in unpacked_params]):
            self.error(f"Function of type '{func.type}' is not callable with parameters of types \
[{', '.join(str(v.type) for v in unpacked_params)}]")

        return func.call(self.ctx, unpacked_params)

    def visit_index_node(self, node: IndexNode) -> Value:
        value = self.visit(node.value)
        if not value.indexable():
            self.error(f"Value of type '{value.type}' is not indexable", node.value.pos)
        if (valid := value.validate_index_count(len(node.indices))) != -1:
            self.error(f"Value of type '{value.type}' requires {valid} indices, provided {len(node.indices)}")
        indices = [self.visit(i) for i in node.indices]
        return value.index(self.ctx, indices)

    def visit_attribute_node(self, node: AttributeNode) -> Value:
        return self.visit(node.value).getattr_req(self.ctx, node.static, node.attr)

    def visit_number_value_node(self, node: NumberValueNode) -> Value:
        return Value.of_number(node.value)

    def visit_string_value_node(self, node: StringValueNode) -> Value:
        return Value.of_string(f"\"{node.value}\"")

    def visit_color_value_node(self, node: ColorValueNode) -> Value:
        return Value.of_number(node.value)

    def visit_variable_value_node(self, node: VariableValueNode) -> Value:
        return self._var_get(node.name)

    def visit_tuple_value_node(self, node: TupleValueNode) -> Value:
        return Value.of_tuple(self.ctx, [self.visit(n) for n in node.values])

    def visit_range_value_node(self, node: RangeValueNode) -> Value:
        return Value.of_range(self.ctx, self.visit(node.start), self.visit(node.end))

    def visit_tuple_type_node(self, node: TupleTypeNode) -> Value:
        return Value.of_type(TupleType([self.resolve_type(t) for t in node.types]))
