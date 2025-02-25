from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .util import Position


class AssignmentTargetVisitor[T](ABC):
    def visit(self, target: AssignmentTarget) -> T:
        return target.accept(self)

    @abstractmethod
    def visit_single_assignment_target(self, target: SingleAssignmentTarget) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_unpack_assignment_target(self, target: UnpackAssignmentTarget) -> T:
        raise NotImplementedError


@dataclass
class AssignmentTarget(ABC):
    const: bool

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def accept[T](self, visitor: AssignmentTargetVisitor[T]) -> T:
        raise NotImplementedError


@dataclass
class SingleAssignmentTarget(AssignmentTarget):
    name: str
    type: Node | None

    def __str__(self):
        if self.type is None:
            return self.name
        return f"{self.name}: {self.type}"

    def accept[T](self, visitor: AssignmentTargetVisitor[T]) -> T:
        return visitor.visit_single_assignment_target(self)


@dataclass
class UnpackAssignmentTarget(AssignmentTarget):
    values: list[str]
    types: list[Node] | None

    def __str__(self):
        if self.types is None:
            return f"({', '.join(self.values)})"
        else:
            return f"({', '.join(self.values)}): ({', '.join(map(str, self.types))})"

    def accept[T](self, visitor: AssignmentTargetVisitor[T]) -> T:
        return visitor.visit_unpack_assignment_target(self)


class AstVisitor[T](ABC):
    current_node: Node | None
    current_pos: Position | None

    def __init__(self):
        self.current_node = None
        self.current_pos = None

    def visit(self, node: Node) -> T:
        prev_node = self.current_node
        prev_pos = self.current_pos
        self.current_node = node
        self.current_pos = node.pos
        value = node.accept(self)
        self.current_node = prev_node
        self.current_pos = prev_pos
        return value

    @abstractmethod
    def visit_block_node(self, node: BlockNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_declaration_node(self, node: DeclarationNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_comptime_node(self, node: ComptimeNode) -> T:
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
    def visit_namespace_node(self, node: NamespaceNode) -> T:
        raise NotImplementedError

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

    @abstractmethod
    def visit_tuple_type_node(self, node: TupleTypeNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_function_type_node(self, node: FunctionTypeNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_null_value_node(self, node: NullValueNode) -> T:
        raise NotImplementedError


@dataclass
class Node(ABC):
    pos: Position

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def accept[T](self, visitor: AstVisitor[T]) -> T:
        raise NotImplementedError

    def __bool__(self):
        return True


@dataclass
class BlockNode(Node):
    code: list[Node]
    returns_last: bool

    def __str__(self):
        lines = []
        for i, node in enumerate(self.code):
            lines.append(str(node))
            if not self.returns_last or i != len(self.code) - 1:
                lines[-1] += ";"
        return f"{{\n{'\n'.join(lines)}\n}}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_block_node(self)


@dataclass
class DeclarationNode(Node):
    target: AssignmentTarget
    value: Node

    def __str__(self):
        return f"{'const' if self.target.const else 'let'} {self.target} = {self.value}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_declaration_node(self)


@dataclass
class ComptimeNode(Node):
    value: Node

    def __str__(self):
        return f"comptime {self.value}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_comptime_node(self)


@dataclass
class FunctionParam:
    name: str
    reference: bool
    type: Node | None
    variadic: bool

    debug_variadic_pos: Position | None

    def __str__(self):
        return f"{'&' if self.reference else ''}{self.name}{'...' if self.variadic else ''}\
{': ' + str(self.type) if self.type is not None else ''}"


@dataclass
class FunctionDeclaration:
    name: str | None
    params: list[FunctionParam]
    result: Node | None
    code: Node
    attributes: set[str]

    def __str__(self):
        return f"fn {self.name if self.name is not None else ''}({', '.join(map(str, self.params))}){' -> ' + str(self.result) if self.result is not None else ''} {self.code}"


@dataclass
class FunctionNode(FunctionDeclaration, Node):
    def __str__(self):
        return super().__str__()

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_function_node(self)


@dataclass
class LambdaCapture:
    name: str
    reference: bool
    value: Node | None

    def __str__(self):
        return f"{'&' if self.reference else ''}{self.name}{' = ' + str(self.value) if self.value is not None else ''}"


@dataclass
class LambdaNode(Node):
    params: list[FunctionParam]
    captures: list[LambdaCapture]
    result: Node | None
    code: Node

    def __str__(self):
        captures = f"[{', '.join(map(str, self.captures))}]" if len(self.captures) > 0 else ""
        return f"|{', '.join(map(str, self.params))}|{captures}{' -> ' + str(self.result) if self.result is not None else ''} {self.code}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_lambda_node(self)


@dataclass
class ReturnNode(Node):
    value: Node | None

    def __str__(self):
        return f"return {self.value if self.value is not None else ''}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_return_node(self)


@dataclass
class StructNode(Node):
    name: str | None
    parent: Node | None
    fields: list[SingleAssignmentTarget]
    static_fields: list[tuple[SingleAssignmentTarget, Node]]
    methods: list[tuple[bool, FunctionDeclaration]]
    static_methods: list[FunctionDeclaration]

    def __str__(self):
        return f"struct {self.name}{' : ' + str(self.parent) if self.parent is not None else ''} {{ ... }}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_struct_node(self)


@dataclass
class IfNode(Node):
    const: bool
    cond: Node
    code_if: Node
    code_else: Node | None

    def __str__(self):
        return f"if {'const ' if self.const else ''}{self.cond} {self.code_if}{(' else ' + str(self.code_else)) if self.code_else is not None else ''}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_if_node(self)


@dataclass
class NamespaceNode(Node):
    name: str | None
    code: Node

    def __str__(self):
        return f"scope {self.name + ' ' if self.name is not None else ''}{self.code}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_namespace_node(self)


@dataclass
class EnumNode(Node):
    name: str | None
    options: list[str]

    def __str__(self):
        return f"enum {self.name + ' ' if self.name is not None else ''}{{ {', '.join(self.options)} }}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_enum_node(self)


@dataclass
class WhileNode(Node):
    cond: Node
    code: Node

    def __str__(self):
        return f"while {self.cond} {self.code}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_while_node(self)


@dataclass
class ForNode(Node):
    target: AssignmentTarget
    iterable: Node
    code: Node

    def __str__(self):
        return f"for {self.target} in {self.iterable} {self.code}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_for_node(self)


@dataclass
class ComprehensionNode(Node):
    expr: Node
    target: AssignmentTarget
    iterable: Node

    def __str__(self):
        return f"({self.expr} for {self.target} in {self.iterable})"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_comprehension_node(self)


@dataclass
class BreakNode(Node):
    def __str__(self):
        return "break"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_break_node(self)


@dataclass
class ContinueNode(Node):
    def __str__(self):
        return "continue"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_continue_node(self)


@dataclass
class CastNode(Node):
    value: Node
    type: Node

    def __str__(self):
        return f"{self.value} as {self.type}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_cast_node(self)


@dataclass
class BinaryOpNode(Node):
    left: Node
    op: str
    right: Node

    def __str__(self):
        return f"{self.left} {self.op} {self.right}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_binary_op_node(self)


@dataclass
class UnaryOpNode(Node):
    op: str
    value: Node

    def __str__(self):
        return f"{self.op}{self.value}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_unary_op_node(self)


@dataclass
class CallNode(Node):
    value: Node
    params: list[tuple[Node, bool]]

    def __str__(self):
        return f"{self.value}({', '.join(str(p) + ('...' if u else '') for p, u in self.params)})"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_call_node(self)


@dataclass
class IndexNode(Node):
    value: Node
    indices: list[Node]

    def __str__(self):
        return f"{self.value}[{', '.join(map(str, self.indices))}]"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_index_node(self)


@dataclass
class AttributeNode(Node):
    value: Node
    attr: str
    static: bool

    def __str__(self):
        return f"{self.value}{'::' if self.static else '.'}{self.attr}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_attribute_node(self)


@dataclass
class NumberValueNode(Node):
    value: int | float

    def __str__(self):
        return str(self.value)

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_number_value_node(self)


@dataclass
class StringValueNode(Node):
    value: str

    def __str__(self):
        return f"\"{self.value}\""

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_string_value_node(self)


@dataclass
class ColorValueNode(Node):
    value: str

    def __str__(self):
        return self.value

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_color_value_node(self)


@dataclass
class VariableValueNode(Node):
    name: str

    def __str__(self):
        return self.name

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_variable_value_node(self)


@dataclass
class TupleValueNode(Node):
    values: list[tuple[Node, bool]]

    def __str__(self):
        return f"({', '.join(str(v) + ('...' if u else '') for v, u in self.values)})"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_tuple_value_node(self)


@dataclass
class RangeValueNode(Node):
    start: Node
    end: Node
    step: Node | None

    def __str__(self):
        if self.step is None:
            return f"{self.start}..{self.end}"
        else:
            return f"{self.start}..{self.end}..{self.step}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_range_value_node(self)


@dataclass
class TupleTypeNode(Node):
    types: list[Node]

    def __str__(self):
        return f"({', '.join(map(str, self.types))})"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_tuple_type_node(self)


@dataclass
class FunctionTypeNode(Node):
    params: list[Node | None]
    result: Node | None

    def __str__(self):
        return f"fn({', '.join(str(p) if p is not None else '?' for p in self.params)}) -> {self.result if self.result is not None else '?'}"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_function_type_node(self)


@dataclass
class NullValueNode(Node):
    def __str__(self):
        return "null"

    def accept[T](self, visitor: AstVisitor[T]) -> T:
        return visitor.visit_null_value_node(self)
