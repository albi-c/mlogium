from __future__ import annotations

import copy
from abc import ABC, abstractmethod

from .instruction import InstructionInstance, Instruction


class CompilationContext(ABC):
    _instructions: list[InstructionInstance]
    _tmp_index: int

    def __init__(self):
        self._instructions = []
        self._tmp_index = 0

    def emit(self, *instructions: InstructionInstance):
        self._instructions += instructions

    def tmp(self) -> str:
        self._tmp_index += 1
        return f"__tmp{self._tmp_index}"

    def tmp_num(self) -> int:
        self._tmp_index += 1
        return self._tmp_index

    def get_instructions(self) -> list[InstructionInstance]:
        return copy.deepcopy(self._instructions)

    @abstractmethod
    def generate_node(self, node):
        raise NotImplementedError

    @abstractmethod
    def error(self, msg: str):
        raise NotImplementedError
