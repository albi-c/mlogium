from .node import *
from ..error import CompilerError
from ..util import Position
from ..instruction import Instruction, InstructionInstance
from typing import Callable


class RawInstruction(InstructionInstance):
    def __init__(self, name: str, *params: str):
        super().__init__(Instruction.noop.base_class, [], True, {},
                         name, *params, internal=True)


class AsmCompiler(AsmAstVisitor[None]):
    JUMP_OPS: dict[str, str] = {
        "<": "lessThan",
        "<=": "lessThanEq",
        ">": "greaterThan",
        ">=": "greaterThanEq",
        "==": "equal",
        "!=": "notEqual",
        "===": "strictEqual"
    }

    UNARY_OPERATORS: dict[str, Callable[[str, str], tuple[str, str, str, str]]] = {
        "-": lambda result, value: ("sub", result, "0", value),
        "~": lambda result, value: ("not", result, value, "_"),
        "!": lambda result, value: ("notEqual", result, value, "0")
    }

    BINARY_OPERATORS: dict[str, str] = {
        "+": "add",
        "-": "sub",
        "*": "mul",
        "/": "div",
        "//": "idiv",
        "%": "mod",
        "**": "pow",
        "&&": "land",
        "||": "or",
        "<": "lessThan",
        "<=": "lessThanEq",
        ">": "greaterThan",
        ">=": "greaterThanEq",
        "==": "equal",
        "!=": "notEqual",
        "===": "strictEqual",
        "<<": "shl",
        ">>": "shr",
        "|": "or",
        "&": "and",
        "^": "xor"
    }

    instructions: list[InstructionInstance]

    def __init__(self):
        super().__init__()
        self.instructions = []

    def get_instructions(self) -> list[InstructionInstance]:
        return self.instructions

    def emit(self, *instructions: InstructionInstance):
        self.instructions += instructions

    def compile(self, ast: AsmNode):
        self.visit(ast)

    def visit_root_node(self, node: RootAsmNode):
        for n in node.nodes:
            self.visit(n)

    def visit_jump_node(self, node: JumpAsmNode):
        if node.condition is not None:
            self.emit(Instruction.jump(node.label_name, AsmCompiler.JUMP_OPS[node.condition.op],
                                       node.condition.left, node.condition.right))
        else:
            self.emit(Instruction.jump_always(node.label_name))

    def visit_label_node(self, node: LabelAsmNode):
        self.emit(Instruction.label(node.label_name))

    def visit_assignment_node(self, node: AssignmentAsmNode):
        self.emit(Instruction.set(node.target, node.value))

    def visit_modify_node(self, node: ModifyAsmNode):
        self.emit(Instruction.op(AsmCompiler.BINARY_OPERATORS[node.op[:-1]],
                                 node.target, node.target, node.value))

    def visit_unary_op_node(self, node: UnaryOpAsmNode):
        self.emit(Instruction.op(*AsmCompiler.UNARY_OPERATORS[node.op](node.result, node.value)))

    def visit_binary_op_node(self, node: BinaryOpAsmNode):
        self.emit(Instruction.op(AsmCompiler.BINARY_OPERATORS[node.op],
                                 node.result, node.left, node.right))

    def visit_property_read_node(self, node: PropertyReadAsmNode):
        self.emit(RawInstruction("sensor", node.result, node.value, "@" + node.property))

    def visit_property_write_node(self, node: PropertyWriteAsmNode):
        # TODO: multiple values
        self.emit(RawInstruction("control", node.property, node.target, node.value))

    def visit_index_read_node(self, node: IndexReadAsmNode):
        self.emit(Instruction.read(node.result, node.value, node.index))

    def visit_index_write_node(self, node: IndexWriteAsmNode):
        self.emit(Instruction.write(node.value, node.target, node.index))

    def visit_call_node(self, node: CallAsmNode):
        # TODO: use result
        self.emit(RawInstruction(node.function, *node.params))

    def visit_unpack_call_node(self, node: UnpackCallAsmNode):
        # TODO: use result
        self.emit(RawInstruction(node.function, *node.params))
