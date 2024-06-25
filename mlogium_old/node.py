from __future__ import annotations

import copy

from .util import Position
from .value_types import *
from .tokens import Token
from .macro import MacroRegistry


class AstVisitor[T](ABC):
    current_node: Node | None
    current_pos: Position | None

    def __init__(self):
        self.current_node = None
        self.current_pos = None

    def visit(self, node: Node) -> T:
        prev_node = self.current_node
        self.current_node = node
        self.current_pos = node.pos
        value = node.accept(self)
        self.current_node = prev_node
        return value

    @abstractmethod
    def visit_block_node(self, node: BlockNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_macro_invocation_node(self, node: MacroInvocationNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_declaration_node(self, node: DeclarationNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_function_node(self, node: FunctionNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_lambda_node(self, node: LambdaNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_return_node(self, node: ReturnNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_struct_node(self, node: StructNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_if_node(self, node: IfNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_scope_node(self, node: ScopeNode) -> T:
        raise NotImplementedError

    # @abstractmethod
    # def visit_match_node(self, node: MatchNode) -> T:
    #     raise NotImplementedError

    @abstractmethod
    def visit_enum_node(self, node: EnumNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_while_node(self, node: WhileNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_for_node(self, node: ForNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_comprehension_node(self, node: ComprehensionNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_break_node(self, node: BreakNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_continue_node(self, node: ContinueNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_cast_node(self, node: CastNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_binary_op_node(self, node: BinaryOpNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_unary_op_node(self, node: UnaryOpNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_call_node(self, node: CallNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_index_node(self, node: IndexNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_attribute_node(self, node: AttributeNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_number_value_node(self, node: NumberValueNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_string_value_node(self, node: StringValueNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_color_value_node(self, node: ColorValueNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_variable_value_node(self, node: VariableValueNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_tuple_value_node(self, node: TupleValueNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_range_value_node(self, node: RangeValueNode) -> T:
        raise NotImplementedError


class AssignmentTargetVisitor[T](ABC):
    def visit(self, target: AssignmentTarget) -> T:
        return target.accept(self)

    @abstractmethod
    def visit_single_assignment_target(self, target: SingleAssignmentTarget) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_unpack_assignment_target(self, target: UnpackAssignmentTarget) -> T:
        raise NotImplementedError


class Node(ABC):
    pos: Position

    def __init__(self, pos: Position):
        self.pos = pos

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def accept[T](self, visitor: AstVisitor[T]) -> T:
        raise NotImplementedError

    def __bool__(self):
        return True

    def copy(self) -> Node:
        return copy.deepcopy(self)


class BlockNode(Node):
    code: list[Node]
    returns_last: bool

    def __init__(self, pos: Position, code: list[Node], returns_last: bool):
        super().__init__(pos)

        self.code = code
        self.returns_last = returns_last

    def __str__(self):
        code = '\n'.join(map(str, self.code))
        return f"{'[returns_last]' if self.returns_last else ''}{{\n{code}\n}}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_block_node(self)

    @staticmethod
    def empty(pos: Position):
        return BlockNode(pos, [], False)


class MacroInvocationNode(Node):
    registry: MacroRegistry
    name: str
    params: list[Type | Token | Node]

    def __init__(self, pos: Position, registry: MacroRegistry, name: str, params: list[Type | Token | Node]):
        super().__init__(pos)

        self.registry = registry
        self.name = name
        self.params = params

    def __str__(self):
        return f"#{self.name}({','.join(p.value if isinstance(p, Token) else str(p) for p in self.params)})"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_macro_invocation_node(self)


class AssignmentTarget(ABC):
    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def accept[T](self, visitor: AssignmentTargetVisitor[T]) -> T:
        raise NotImplementedError


class SingleAssignmentTarget(AssignmentTarget):
    name: str
    type: Type | None

    def __init__(self, name: str, type_: Type | None):
        self.name = name
        self.type = type_

    def __str__(self):
        if self.type is None:
            return self.name
        return f"{self.name}: {self.type}"

    def accept[T](self, visitor: AssignmentTargetVisitor[T]) -> T:
        return visitor.visit_single_assignment_target(self)


class UnpackAssignmentTarget(AssignmentTarget):
    values: list[str | tuple[str, Type]]

    def __init__(self, values: list[tuple[str, Type]]):
        self.values = values

    def __str__(self):
        if len(self.values) > 0 and isinstance(self.values[0], str):
            return f"({', '.join(self.values)})"
        return f"({', '.join(p[0] for p in self.values)}): ({', '.join(str(p[1]) for p in self.values)})"

    def accept[T](self, visitor: AssignmentTargetVisitor[T]) -> T:
        return visitor.visit_unpack_assignment_target(self)


class DeclarationNode(Node):
    const: bool
    target: AssignmentTarget
    value: Node

    def __init__(self, pos: Position, const: bool, target: AssignmentTarget, value: Node):
        super().__init__(pos)

        self.const = const
        self.target = target
        self.value = value

    def __str__(self):
        return f"let {self.target} = {self.value}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_declaration_node(self)


class FunctionNode(Node):
    name: str
    type: NamedParamFunctionType
    code: Node

    def __init__(self, pos: Position, name: str, type_: NamedParamFunctionType, code: Node):
        super().__init__(pos)

        self.name = name
        self.type = type_
        self.code = code

    def __str__(self):
        return f"fn {self.name}{self.type} {self.code}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_function_node(self)


class LambdaNode(Node):
    type: LambdaType
    code: Node

    def __init__(self, pos: Position, type_: LambdaType, code: Node):
        super().__init__(pos)

        self.type = type_
        self.code = code

    def __str__(self):
        return f"fn{self.type} {self.code}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_lambda_node(self)


class ReturnNode(Node):
    value: Node | None

    def __init__(self, pos: Position, value: Node | None):
        super().__init__(pos)

        self.value = value

    def __str__(self):
        return f"return {self.value if self.value is not None else ''}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_return_node(self)


class StructNode(Node):
    name: str
    parent: str | None
    wrapped: Type | None
    fields: list[SingleAssignmentTarget]
    static_fields: list[tuple[bool, SingleAssignmentTarget, Node]]
    methods: list[tuple[bool, str, NamedParamFunctionType, Node]]
    static_methods: list[tuple[str, NamedParamFunctionType, Node]]

    def __init__(self, pos: Position, name: str, parent: str | None, wrapped: Type | None,
                 fields: list[SingleAssignmentTarget],
                 static_fields: list[tuple[bool, SingleAssignmentTarget, Node]],
                 methods: list[tuple[bool, str, NamedParamFunctionType, Node]],
                 static_methods: list[tuple[str, NamedParamFunctionType, Node]]):
        super().__init__(pos)

        self.name = name
        self.parent = parent
        self.wrapped = wrapped
        self.fields = fields
        self.static_fields = static_fields
        self.methods = methods
        self.static_methods = static_methods

    def __str__(self):
        return f"struct {self.name}{' : ' + self.parent if self.parent is not None else ''}{' of ' + str(self.wrapped) if self.wrapped is not None else ''} {{...}}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_struct_node(self)


class IfNode(Node):
    const: bool
    cond: Node
    code_if: Node
    code_else: Node | None

    def __init__(self, pos: Position, const: bool, cond: Node, code_if: Node, code_else: Node | None):
        super().__init__(pos)

        self.const = const
        self.cond = cond
        self.code_if = code_if
        self.code_else = code_else

    def __str__(self):
        return f"if {'const ' if self.const else ''}{self.cond} {self.code_if}{(' else ' + str(self.code_else)) if self.code_else is not None else ''}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_if_node(self)


class ScopeNode(Node):
    code: Node

    def __init__(self, pos: Position, code: Node):
        super().__init__(pos)

        self.code = code

    def __str__(self):
        return f"scope {{ {self.code} }}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_scope_node(self)


# class MatchNode(Node):
#     value: Node
#     patterns: list[tuple[list[Pattern], Node]]
#
#     def __init__(self, pos: Position, value: Node, patterns: list[tuple[list[Pattern], Node]]):
#         super().__init__(pos)
#
#         self.value = value
#         self.patterns = patterns
#
#     def __str__(self):
#         return f"match {self.value} {{{'\n'.join(f'{' | '.join(map(str, p[0]))} -> {p[1]}' for p in self.patterns)}}}"
#
#     def accept[T](self, visitor: AstVisitor[T]) -> T:
#         return visitor.visit_match_node(self)


class EnumNode(Node):
    name: str
    options: list[str]

    def __init__(self, pos: Position, name: str, options: list[str]):
        super().__init__(pos)

        self.name = name
        self.options = options

    def __str__(self):
        return f"enum {self.name} {{ {', '.join(self.options)} }}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_enum_node(self)


class WhileNode(Node):
    cond: Node
    code: Node

    def __init__(self, pos: Position, cond: Node, code: Node):
        super().__init__(pos)

        self.cond = cond
        self.code = code

    def __str__(self):
        return f"while {self.cond} {self.code}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_while_node(self)


class ForNode(Node):
    target: AssignmentTarget
    iterable: Node
    code: Node

    def __init__(self, pos: Position, target: AssignmentTarget, iterable: Node, code: Node):
        super().__init__(pos)

        self.target = target
        self.iterable = iterable
        self.code = code

    def __str__(self):
        return f"for {self.target} in {self.iterable} {self.code}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_for_node(self)


class ComprehensionNode(Node):
    expr: Node
    target: AssignmentTarget
    iterable: Node

    def __init__(self, pos: Position, expr: Node, target: AssignmentTarget, iterable: Node):
        super().__init__(pos)

        self.expr = expr
        self.target = target
        self.iterable = iterable

    def __str__(self):
        return f"({self.expr} for {self.target} in {self.iterable})"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_comprehension_node(self)


class BreakNode(Node):
    def __str__(self):
        return "break"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_break_node(self)


class ContinueNode(Node):
    def __str__(self):
        return "continue"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_continue_node(self)


class CastNode(Node):
    value: Node
    type: Type

    def __init__(self, pos: Position, value: Node, type_: Type):
        super().__init__(pos)

        self.value = value
        self.type = type_

    def __str__(self):
        return f"{self.value} as {self.type}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_cast_node(self)


class BinaryOpNode(Node):
    left: Node
    op: str
    right: Node

    def __init__(self, pos: Position, left: Node, op: str, right: Node):
        super().__init__(pos)

        self.left = left
        self.op = op
        self.right = right

    def __str__(self):
        return f"{self.left} {self.op} {self.right}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_binary_op_node(self)


class UnaryOpNode(Node):
    op: str
    value: Node

    def __init__(self, pos: Position, op: str, value: Node):
        super().__init__(pos)

        self.op = op
        self.value = value

    def __str__(self):
        return f"{self.op}{self.value}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_unary_op_node(self)


class CallNode(Node):
    value: Node
    params: list[tuple[Node, bool]]

    def __init__(self, pos: Position, value: Node, params: list[tuple[Node, bool]]):
        super().__init__(pos)

        self.value = value
        self.params = params

    def __str__(self):
        return f"{self.value}({', '.join(str(p) + ('...' if u else '') for p, u in self.params)})"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_call_node(self)


class IndexNode(Node):
    value: Node
    index: Node

    def __init__(self, pos: Position, value: Node, index: Node):
        super().__init__(pos)

        self.value = value
        self.index = index

    def __str__(self):
        return f"{self.value}[{self.index}]"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_index_node(self)


class AttributeNode(Node):
    value: Node
    attr: str
    static: bool

    def __init__(self, pos: Position, value: Node, attr: str, static: bool):
        super().__init__(pos)

        self.value = value
        self.attr = attr
        self.static = static

    def __str__(self):
        return f"{self.value}{'::' if self.static else '.'}{self.attr}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_attribute_node(self)


class NumberValueNode(Node):
    value: int | float

    def __init__(self, pos: Position, value: int | float):
        super().__init__(pos)

        self.value = value

    def __str__(self):
        return str(self.value)

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_number_value_node(self)


class StringValueNode(Node):
    value: str

    def __init__(self, pos: Position, value: str):
        super().__init__(pos)

        self.value = value

    def __str__(self):
        return f"\"{self.value}\""

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_string_value_node(self)


class ColorValueNode(Node):
    value: str

    def __init__(self, pos: Position, value: str):
        super().__init__(pos)

        self.value = value

    def __str__(self):
        return self.value

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_color_value_node(self)


class VariableValueNode(Node):
    name: str

    def __init__(self, pos: Position, value: str):
        super().__init__(pos)

        self.name = value

    def __str__(self):
        return self.name

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_variable_value_node(self)


class TupleValueNode(Node):
    values: list[Node]

    def __init__(self, pos: Position, values: list[Node]):
        super().__init__(pos)

        self.values = values

    def __str__(self):
        return f"({', '.join(map(str, self.values))})"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_tuple_value_node(self)


class RangeValueNode(Node):
    start: Node
    end: Node

    def __init__(self, pos: Position, start: Node, end: Node):
        super().__init__(pos)

        self.start = start
        self.end = end

    def __str__(self):
        return f"{self.start}..{self.end}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_range_value_node(self)
