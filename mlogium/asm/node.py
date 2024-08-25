from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..util import Position


class AsmAstVisitor[T](ABC):
    current_node: AsmNode | None
    current_pos: Position | None

    def __init__(self):
        self.current_node = None
        self.current_pos = None

    def visit(self, node: AsmNode) -> T:
        prev_node = self.current_node
        self.current_node = node
        self.current_pos = node.pos
        value = node.accept(self)
        self.current_node = prev_node
        return value

    @abstractmethod
    def visit_root_node(self, node: RootAsmNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_jump_node(self, node: JumpAsmNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_label_node(self, node: LabelAsmNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_assignment_node(self, node: AssignmentAsmNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_modify_node(self, node: ModifyAsmNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_unary_op_node(self, node: UnaryOpAsmNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_binary_op_node(self, node: BinaryOpAsmNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_property_read_node(self, node: PropertyReadAsmNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_property_write_node(self, node: PropertyWriteAsmNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_index_read_node(self, node: IndexReadAsmNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_index_write_node(self, node: IndexWriteAsmNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_call_node(self, node: CallAsmNode) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_unpack_call_node(self, node: UnpackCallAsmNode) -> T:
        raise NotImplementedError


@dataclass
class AsmNode(ABC):
    pos: Position

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def accept[T](self, visitor: AsmAstVisitor[T]) -> T:
        raise NotImplementedError

    def __bool__(self):
        return True


@dataclass
class RootAsmNode(AsmNode):
    nodes: list[AsmNode]

    def __str__(self):
        return "\n".join(map(str, self.nodes))

    def accept[T](self, visitor: AsmAstVisitor[T]) -> T:
        return visitor.visit_root_node(self)


@dataclass
class JumpAsmNode(AsmNode):
    @dataclass
    class Condition:
        left: str
        op: str
        right: str

        def __str__(self):
            return f"({self.left} {self.op} {self.right})"

    label_name: str
    condition: JumpAsmNode.Condition | None

    def __str__(self):
        return f"-> {self.label_name}{str(self.condition) if self.condition is not None else ''}"

    def accept[T](self, visitor: AsmAstVisitor[T]) -> T:
        return visitor.visit_jump_node(self)


@dataclass
class LabelAsmNode(AsmNode):
    label_name: str

    def __str__(self):
        return f"{self.label_name}:"

    def accept[T](self, visitor: AsmAstVisitor[T]) -> T:
        return visitor.visit_label_node(self)


@dataclass
class AssignmentAsmNode(AsmNode):
    target: str
    value: str

    def __str__(self):
        return f"{self.target} = {self.value}"

    def accept[T](self, visitor: AsmAstVisitor[T]) -> T:
        return visitor.visit_assignment_node(self)


@dataclass
class ModifyAsmNode(AsmNode):
    target: str
    op: str
    value: str

    def __str__(self):
        return f"{self.target} {self.op} {self.value}"

    def accept[T](self, visitor: AsmAstVisitor[T]) -> T:
        return visitor.visit_modify_node(self)


@dataclass
class UnaryOpAsmNode(AsmNode):
    result: str
    op: str
    value: str

    def __str__(self):
        return f"{self.result} = {self.op}{self.value}"

    def accept[T](self, visitor: AsmAstVisitor[T]) -> T:
        return visitor.visit_unary_op_node(self)


@dataclass
class BinaryOpAsmNode(AsmNode):
    result: str
    left: str
    op: str
    right: str

    def __str__(self):
        return f"{self.result} = {self.left} {self.op} {self.right}"

    def accept[T](self, visitor: AsmAstVisitor[T]) -> T:
        return visitor.visit_binary_op_node(self)


@dataclass
class PropertyReadAsmNode(AsmNode):
    result: str
    value: str
    property: str

    def __str__(self):
        return f"{self.result} = {self.value}.{self.property}"

    def accept[T](self, visitor: AsmAstVisitor[T]) -> T:
        return visitor.visit_property_read_node(self)


@dataclass
class PropertyWriteAsmNode(AsmNode):
    target: str
    property: str
    value: str

    def __str__(self):
        return f"{self.target}.{self.property} = {self.value}"

    def accept[T](self, visitor: AsmAstVisitor[T]) -> T:
        return visitor.visit_property_write_node(self)


@dataclass
class IndexReadAsmNode(AsmNode):
    result: str
    value: str
    index: str

    def __str__(self):
        return f"{self.result} = {self.value}[{self.index}]"

    def accept[T](self, visitor: AsmAstVisitor[T]) -> T:
        return visitor.visit_index_read_node(self)


@dataclass
class IndexWriteAsmNode(AsmNode):
    target: str
    index: str
    value: str

    def __str__(self):
        return f"{self.target}[{self.index}] = {self.value}"

    def accept[T](self, visitor: AsmAstVisitor[T]) -> T:
        return visitor.visit_index_write_node(self)


@dataclass
class CallAsmNode(AsmNode):
    result: str | None
    function: str
    params: list[str]

    def __str__(self):
        if self.result is not None:
            return f"{self.result} = {self.function}({', '.join(self.params)})"
        else:
            return f"{self.function}({', '.join(self.params)})"

    def accept[T](self, visitor: AsmAstVisitor[T]) -> T:
        return visitor.visit_call_node(self)


@dataclass
class UnpackCallAsmNode(AsmNode):
    results: list[str]
    function: str
    params: list[str]

    def __str__(self):
        return f"[{', '.join(self.results)}] = {self.function}({', '.join(self.params)})"

    def accept[T](self, visitor: AsmAstVisitor[T]) -> T:
        return visitor.visit_unpack_call_node(self)
