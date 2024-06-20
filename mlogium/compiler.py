from __future__ import annotations

from .node import *
from .value import *
from .instruction import Instruction, InstructionInstance
from .compilation_context import CompilationContext, InstructionSection
from .error import CompilerError
from .scope import ScopeStack
from .macro import MacroInvocationContext
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

        def error(self, msg: str):
            self.compiler._error(msg)

    ctx: CompilationContext
    scope: ScopeStack
    functions_with_pointers: set[str]

    def __init__(self):
        super().__init__()

        self.scope = ScopeStack()
        self.ctx = Compiler.CompilationContext(self, self.scope)
        self.functions_with_pointers = set()

        self.scope.scopes.append(ScopeStack.Scope("<builtins>"))
        for name, value in builtins.BUILTINS.items():
            self.scope.scopes[-1].variables[name] = value
        self.scope.scopes.append(ScopeStack.Scope("<main>"))

    def compile(self, node: Node):
        TypeImplRegistry.reset_basic_type_impls()
        self.visit(node)

    def emit(self, *instructions: InstructionInstance):
        self.ctx.emit(*instructions)

    def _error(self, msg: str, pos: Position = None):
        CompilerError.custom(self.current_pos if pos is None else pos, msg)

    def _var_get(self, name: str) -> Value:
        if (var := self.scope.get(name)) is None:
            self._error(f"Variable not found: '{name}'")
        return var

    def _var_declare(self, name: str, type_: Type | None, value: Value, const: bool) -> Value:
        if type_ is None:
            type_ = value.type
        if type_.is_opaque():
            self._error(f"Cannot declare variable of opaque type")
        if (var := self.scope.declare(name, type_, const)) is None:
            self._error(f"Variable already exists: '{name}'")
        var.assign(self.ctx, value)
        return var

    def _var_declare_special(self, name: str, value: Value) -> Value:
        if not self.scope.declare_special(name, value):
            self._error(f"Variable already exists: '{name}'")
        return value

    def visit_block_node(self, node: BlockNode) -> Value:
        last_value = Value.null()
        for n in node.code:
            last_value = self.visit(n)
        if node.returns_last:
            return last_value
        return Value.null()

    def visit_macro_invocation_node(self, node: MacroInvocationNode) -> Value:
        return node.registry.get(node.name).invoke(MacroInvocationContext(
            self.ctx, node.pos, node.registry), self, node.params)

    def _declare_target(self, target: AssignmentTarget, value: Value, const: bool) -> Value:
        if isinstance(target, SingleAssignmentTarget):
            self._var_declare(target.name, target.type, value, const)

        elif isinstance(target, UnpackAssignmentTarget):
            values = value.unpack(self.ctx)
            if values is None:
                self._error(f"Value of type {value.type} is not unpackable")
            if len(values) != len(target.values):
                self._error(
                    f"Value of type {value.type} unpacks into {len(values)} values, {len(target.values)} expected")
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
        return self._declare_target(node.target, self.visit(node.value), node.const)

    def _make_function_value(self, name: str, type_: NamedParamFunctionType | LambdaType,
                             code: Node, is_lambda: bool) -> Value:
        if name in self.functions_with_pointers:
            self._error(f"Function already defined: '{name}'")
        self.functions_with_pointers.add(name)

        if is_lambda:
            captures = {}
            for name, ref, value in type_.captures:
                if name in captures:
                    self._error(f"Duplicate capture in lambda: '{name}'", self.current_pos)
                if ref:
                    captures[name] = self.visit(value)
                else:
                    val = self.visit(value)
                    copied = Value.variable(self.ctx.tmp(), val.type)
                    copied.assign(self.ctx, val)
                    captures[name] = copied
            new_type = LambdaType(type_.named_params, type_.ret, type_.captures, {"code": code, "captures": captures})
        else:
            new_type = ConcreteFunctionType(name, type_.named_params, type_.ret, {"code": code})

        return Value(new_type, name, True)

    def _register_function_value(self, name: str, val: Value):
        if not val.needs_function_ref():
            return

        with self.ctx.in_section(InstructionSection.FUNCTIONS):
            self.emit(Instruction.label(ABI.function_label(name)))
            ret_addr = Value.variable(self.ctx.tmp(), Type.NUM)
            ret_addr.assign(self.ctx, Value.variable(ABI.function_return_address(), Type.NUM))
            params = [Value.variable(ABI.function_parameter(i),
                                     type_ if type_ is not None else Type.ANY) for i, type_ in enumerate(val.params())]
            ret = val.call(self.ctx, params)
            Value.variable(ABI.function_return_value(), ret.type).assign(self.ctx, ret)
            self.emit(Instruction.jump_addr(ret_addr.value))

    def register_function(self, name: str, type_: NamedParamFunctionType | LambdaType,
                          code: Node, is_lambda: bool = False) -> Value:
        val = self._make_function_value(name, type_, code, is_lambda)
        self._register_function_value(name, val)
        return val

    def visit_function_node(self, node: FunctionNode) -> Value:
        return self._var_declare_special(node.name, self.register_function(node.name, node.type, node.code))

    def visit_lambda_node(self, node: LambdaNode) -> Value:
        return self.register_function(f"__lambda_{self.ctx.tmp_num()}", node.type, node.code, True)

    def visit_return_node(self, node: ReturnNode) -> Value:
        if (func := self.scope.get_function()) is None:
            self._error(f"Return must be in a function")

        val = self.visit(node.value)
        Value.variable(ABI.return_value(func), val.type).assign(self.ctx, val)
        self.emit(Instruction.jump_always(ABI.function_end(func)))

        return Value.null()

    @staticmethod
    def _resolve_struct(struct: StructBaseTypeImpl, names: set[str], base: StructBaseTypeImpl):
        for field in base.fields:
            if field[0] not in names:
                struct.fields.insert(0, field)

        for name, method in base.methods.items():
            if name not in names:
                struct.methods[name] = method

        for name, value in base.static_values:
            if name not in base.static_values:
                struct.static_values[name] = value

        struct.rebuild()

    def _visit_wrapper_struct(self, node: StructNode) -> Value:
        wrapped = node.wrapped
        assert wrapped is not None

        methods = {}
        names = set()
        static_values = {}
        functions_to_register = []
        static_variables = []

        for const, name, type_, code in node.methods:
            if name in names:
                self._error(f"Struct already has field '{name}'")
            names.add(name)
            name_ = ABI.attribute(node.name, name)
            type_modified = NamedParamFunctionType(
                [("self", None, True)] + type_.named_params, type_.ret)
            val = self._make_function_value(name_, type_modified, code, False)
            methods[name] = (const, val)

        for name, type_, code in node.static_methods:
            if name in static_values:
                self._error(f"Struct already has attribute '{name}'")
            name_ = ABI.static_attribute(node.name, name)
            val = self._make_function_value(name_, type_, code, False)
            static_values[name] = val
            functions_to_register.append((name_, val))

        for const, target, value in node.static_fields:
            if target.name in static_values:
                self._error(f"Struct already has attribute '{target.name}'")
            names.add(target.name)
            static_variables.append((const, target, value))

        impl = WrapperStructBaseTypeImpl(node.name, wrapped, methods, static_values)
        struct = Value(BasicType(f"$WrapperStructBase_{node.name}"), node.name, impl=impl)
        self._var_declare_special(node.name, struct)
        impl.register_type()

        for const, target, value in static_variables:
            val = self.visit(value)
            var = Value.variable(self.ctx.tmp(), val.type, const_on_write=const)
            var.assign(self.ctx, val)
            static_values[target.name] = var

        for name, val in functions_to_register:
            self._register_function_value(name, val)

        return struct

    def visit_struct_node(self, node: StructNode) -> Value:
        if node.wrapped is not None:
            return self._visit_wrapper_struct(node)

        if node.parent is not None:
            if node.parent == node.name:
                self._error(f"Struct cannot inherit from itself", node.pos)

            base_impl = TypeImplRegistry.get_impl(BasicType(f"$StructBase_{node.parent}"))
            if not isinstance(base_impl, StructBaseTypeImpl):
                self._error(f"Struct cannot inherit from type '{node.parent}'", node.pos)

        else:
            base_impl = None

        fields = []
        methods = {}
        names = set()
        static_values = {}
        functions_to_register = []
        static_variables = []

        for const, name, type_, code in node.methods:
            if name in names:
                self._error(f"Struct already has field '{name}'")
            names.add(name)
            name_ = ABI.attribute(node.name, name)
            type_modified = NamedParamFunctionType(
                [("self", None, True)] + type_.named_params, type_.ret)
            val = self._make_function_value(name_, type_modified, code, False)
            methods[name] = (const, val)

        for target in node.fields:
            if target.name in names:
                self._error(f"Struct already has field '{target.name}'")
            names.add(target.name)
            fields.append((target.name, target.type))

        for name, type_, code in node.static_methods:
            if name in static_values:
                self._error(f"Struct already has attribute '{name}'")
            name_ = ABI.static_attribute(node.name, name)
            val = self._make_function_value(name_, type_, code, False)
            static_values[name] = val
            functions_to_register.append((name_, val))

        for const, target, value in node.static_fields:
            if target.name in static_values:
                self._error(f"Struct already has attribute '{target.name}'")
            names.add(target.name)
            static_variables.append((const, target, value))

        impl = StructBaseTypeImpl(node.name, fields, methods, static_values, base_impl)
        struct = Value(BasicType(f"$StructBase_{node.name}"), node.name, impl=impl)
        self._var_declare_special(node.name, struct)
        impl.register_type()

        for const, target, value in static_variables:
            val = self.visit(value)
            var = Value.variable(self.ctx.tmp(), val.type, const_on_write=const)
            var.assign(self.ctx, val)
            static_values[target.name] = var

        for name, val in functions_to_register:
            self._register_function_value(name, val)

        if base_impl is not None:
            self._resolve_struct(impl, names, base_impl)

        return struct

    def visit_if_node(self, node: IfNode) -> Value:
        cond_value = self.visit(node.cond)
        if (condition := cond_value.to_condition(self.ctx)) is None:
            self._error(f"Value of type '{cond_value.type}' is not usable as a condition", node.cond.pos)

        if node.const:
            try:
                num = int(condition.value)
            except ValueError:
                self._error(f"Value '{node.cond}' is not usable as a constant condition")
            else:
                if num != 0:
                    return self.visit(node.code_if)
                else:
                    if node.code_else is not None:
                        return self.visit(node.code_else)
                    else:
                        return Value.null()

        end_true_branch = self.ctx.tmp()
        end_false_branch = "" if node.code_else is None else self.ctx.tmp()
        self.emit(Instruction.jump(end_true_branch, "equal", condition.value, "0"))
        with self.scope(end_true_branch):
            true_value = self.visit(node.code_if)
            result = Value.variable(self.ctx.tmp(), true_value.type)
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

    def visit_scope_node(self, node: ScopeNode) -> Value:
        with self.scope(self.ctx.tmp()):
            return self.visit(node.code)

    def visit_enum_node(self, node: EnumNode) -> Value:
        impl = CustomEnumBaseTypeImpl(node.name, {name: i for i, name in enumerate(node.options)})
        val = Value(BasicType("$Enum_" + node.name), node.name, impl=impl)
        self._var_declare_special(node.name, val)
        impl.register_type()
        return val

    def visit_while_node(self, node: WhileNode) -> Value:
        name = self.ctx.tmp()
        self.emit(Instruction.label(name + "_continue"))

        cond_value = self.visit(node.cond)
        if (cond := cond_value.to_condition(self.ctx)) is None:
            self._error(f"Value of type '{cond_value.type}' is not usable as a condition", node.cond.pos)

        self.emit(Instruction.jump(name + "_break", "equal", cond.value, "0"))
        with self.scope.loop(name):
            self.visit(node.code)
        self.emit(
            Instruction.jump_always(name + "_continue"),
            Instruction.label(name + "_break")
        )
        return Value.null()

    def visit_for_node(self, node: ForNode) -> Value:
        name = self.ctx.tmp()
        iterable = self.visit(node.iterable)
        iterator = iterable.iterate(self.ctx)
        if iterator is None:
            self._error(f"Value of type '{iterable.type}' is not iterable", node.iterable.pos)
        self.emit(Instruction.label(name + "_start"))
        has_value = iterator.has_value(self.ctx).value
        self.emit(Instruction.jump(name + "_break", "equal", has_value, "0"))
        with self.scope.loop(name):
            self._declare_target(node.target, iterator.get_current(self.ctx), iterable.const)
            self.visit(node.code)
        self.emit(Instruction.label(name + "_continue"))
        iterator.to_next(self.ctx)
        self.emit(
            Instruction.jump_always(name + "_start"),
            Instruction.label(name + "_break")
        )
        return Value.null()

    def visit_comprehension_node(self, node: ComprehensionNode) -> Value:
        iterable = self.visit(node.iterable)
        if (values := iterable.unpack(self.ctx)) is None:
            self._error(f"Value of type '{iterable.type}' is not iterable", node.iterable.pos)
        results = []
        for val in values:
            with self.scope(self.ctx.tmp()):
                self._declare_target(node.target, val, True)
                results.append(self.visit(node.expr))
        return Value.tuple(self.ctx, results)

    def visit_break_node(self, node: BreakNode) -> Value:
        if (name := self.scope.get_loop()) is None:
            self._error(f"Break statement must be in a loop")
        self.emit(Instruction.jump_always(name + "_break"))
        return Value.null()

    def visit_continue_node(self, node: ContinueNode) -> Value:
        if (name := self.scope.get_loop()) is None:
            self._error(f"Continue statement must be in a loop")
        self.emit(Instruction.jump_always(name + "_continue"))
        return Value.null()

    def visit_cast_node(self, node: CastNode) -> Value:
        value = self.visit(node.value)
        return value.into_req(self.ctx, node.type)

    def visit_binary_op_node(self, node: BinaryOpNode) -> Value:
        left = self.visit(node.left)
        right = self.visit(node.right)
        result = left.binary_op(self.ctx, node.op, right)
        if result is None:
            self._error(f"Unsupported operation: {left.type} {node.op} {right.type}")
        return result

    def visit_unary_op_node(self, node: UnaryOpNode) -> Value:
        value = self.visit(node.value)
        result = value.unary_op(self.ctx, node.op)
        if result is None:
            self._error(f"Unsupported operation: {node.op} {value.type}")
        return result

    def visit_call_node(self, node: CallNode) -> Value:
        func = self.visit(node.value)
        if not func.callable():
            self._error(f"Value of type '{func.type}' is not callable", node.value.pos)

        param_types = func.params()

        unpacked_params = []
        for param, unpack in node.params:
            if unpack:
                value = self.visit(param)
                if (unpacked := value.unpack(self.ctx)) is None:
                    self._error(f"Value '{param}' of type {value.type} is not unpackable", param.pos)
                unpacked_params += unpacked
            else:
                bottom_scope = {}
                if len(unpacked_params) < len(param_types):
                    type_ = param_types[len(unpacked_params)]
                    if isinstance(type_, BasicType) and (val := builtins.BUILTIN_ENUMS.get(type_.name.lstrip("$"))):
                        impl = val.impl
                        assert isinstance(impl, EnumBaseTypeImpl)
                        bottom_scope = impl.prefixed_values
                    else:
                        impl = TypeImplRegistry.get_impl(type_)
                        if isinstance(impl, CustomEnumInstanceTypeImpl):
                            bottom_scope = impl.base.prefixed_values
                with self.scope.bottom("<enum>", bottom_scope):
                    unpacked_params.append(self.visit(param))

        if len(param_types) != len(unpacked_params):
            self._error(
                f"Invalid number of parameters to function ({len(node.params)} provided, {len(param_types)} expected)")
        params = []
        for type_, param in zip(param_types, unpacked_params):
            params.append(param.into_req(self.ctx, type_))

        return func.call(self.ctx, params)

    def visit_index_node(self, node: IndexNode) -> Value:
        value = self.visit(node.value)
        index = self.visit(node.index)
        if (result := value.index_at(self.ctx, index)) is None:
            self._error(f"Not indexable: '{value}'", node.value.pos)
        return result

    def visit_attribute_node(self, node: AttributeNode) -> Value:
        value = self.visit(node.value)
        if (val := value.getattr(self.ctx, node.attr, node.static)) is None:
            self._error(
                f"Value of type '{value.type}' has no{' static' if node.static else ''} attribute '{node.attr}'")
        return val

    def visit_number_value_node(self, node: NumberValueNode) -> Value:
        return Value.number(node.value)

    def visit_string_value_node(self, node: StringValueNode) -> Value:
        return Value.string(f"\"{node.value}\"")

    def visit_color_value_node(self, node: ColorValueNode) -> Value:
        return Value(Type.NUM, node.value, True)

    def visit_variable_value_node(self, node: VariableValueNode) -> Value:
        return self._var_get(node.name)

    def visit_tuple_value_node(self, node: TupleValueNode) -> Value:
        return Value.tuple(self.ctx, [self.visit(val) for val in node.values])

    def visit_range_value_node(self, node: RangeValueNode) -> Value:
        value = Value.variable(self.ctx.tmp(), Type.RANGE)
        value.getattr(self.ctx, "start", False).assign(self.ctx, self.visit(node.start))
        value.getattr(self.ctx, "end", False).assign(self.ctx, self.visit(node.end))
        return value
