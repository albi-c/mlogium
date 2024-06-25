from __future__ import annotations

import contextlib
from abc import abstractmethod
from collections import defaultdict

from .instruction import InstructionInstance, Instruction


class InstructionSection:
    index: int
    prefix: list[InstructionInstance]
    suffix: list[InstructionInstance]

    FUNCTIONS: InstructionSection
    DEFAULT: InstructionSection

    def __init__(self, value: int, prefix: list[InstructionInstance] = None, suffix: list[InstructionInstance] = None):
        self.index = value
        self.prefix = prefix if prefix is not None else []
        self.suffix = suffix if suffix is not None else []

    def __eq__(self, other):
        return isinstance(other, InstructionSection) and self.index == other.index

    def __hash__(self):
        return self.index

    def __repr__(self):
        return f"InstructionSection({self.index}, {self.prefix}, {self.suffix})"


InstructionSection.FUNCTIONS = InstructionSection(
    0,
    [Instruction.jump_always("__functions_end")],
    [Instruction.label("__functions_end")]
)
InstructionSection.DEFAULT = InstructionSection(1)


class CompilationContext:
    _instruction_sections: defaultdict[InstructionSection, list[InstructionInstance]]
    _tmp_index: int
    _section: InstructionSection

    def __init__(self, scope):
        self.scope = scope

        self._instruction_sections = defaultdict(list)
        self._tmp_index = 0
        self._section = InstructionSection.DEFAULT

    def emit(self, *instructions: InstructionInstance):
        self._instruction_sections[self._section] += instructions

    @contextlib.contextmanager
    def in_section(self, section: InstructionSection):
        sect = self._section
        self._section = section
        try:
            yield
        finally:
            self._section = sect

    def tmp(self) -> str:
        self._tmp_index += 1
        return f"__tmp{self._tmp_index}"

    def tmp_num(self) -> int:
        self._tmp_index += 1
        return self._tmp_index

    def get_instructions(self) -> list[InstructionInstance]:
        instructions = []
        for section in sorted(self._instruction_sections.keys(), key=lambda x: x.index):
            instructions += section.prefix
            instructions += self._instruction_sections[section]
            instructions += section.suffix
        return instructions

    @abstractmethod
    def generate_node(self, node):
        raise NotImplementedError

    @abstractmethod
    def error(self, msg: str):
        raise NotImplementedError
