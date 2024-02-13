from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict

from .instruction import *
from .node import Node


class InstructionSection:
    index: int
    prefix: list[Instruction]
    suffix: list[Instruction]

    FUNCTIONS: InstructionSection
    DEFAULT: InstructionSection

    def __init__(self, value: int, prefix: list[Instruction] = None, suffix: list[Instruction] = None):
        self.index = value
        self.prefix = prefix if prefix is not None else []
        self.suffix = suffix if suffix is not None else []

    def __eq__(self, other):
        return isinstance(other, InstructionSection) and self.index == other.index

    def __hash__(self):
        return self.index

    def __repr__(self):
        return f"InstructionSection({self.index})"

    def emit(self, instructions: list[Instruction]):
        self.instructions += instructions


InstructionSection.FUNCTIONS = InstructionSection(
    0, [Instruction.jump_always("__func_end")],
    [Instruction.label("__func_end")]
)
InstructionSection.DEFAULT = InstructionSection(1)


class CompilationContext:
    _instruction_sections: defaultdict[InstructionSection, list[Instruction]]
    _tmp_index: int

    def __init__(self, scope):
        self.scope = scope

        self._instruction_sections = defaultdict(list)
        self._tmp_index = 0

    def emit(self, *instructions: Instruction, section: InstructionSection = InstructionSection.DEFAULT):
        self._instruction_sections[section] += instructions

    def tmp(self) -> str:
        self._tmp_index += 1
        return f"__tmp{self._tmp_index}"

    def tmp_num(self) -> int:
        self._tmp_index += 1
        return self._tmp_index

    def get_instructions(self) -> list[Instruction]:
        instructions = []
        for section in sorted(self._instruction_sections.keys(), key=lambda x: x.index):
            instructions += section.prefix
            instructions += self._instruction_sections[section]
            instructions += section.suffix
        return instructions

    @abstractmethod
    def generate_node(self, node: Node):
        raise NotImplementedError
