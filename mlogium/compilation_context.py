from __future__ import annotations

import copy
from abc import ABC, abstractmethod

from .instruction import InstructionInstance
from .util import Position


class CompilationContext(ABC):
    scope: 'ScopeStack'

    _instructions: list[InstructionInstance]
    _tmp_index: int

    def __init__(self, scope: 'ScopeStack'):
        self.scope = scope

        self._instructions = []
        self._tmp_index = 0

    def emit(self, *instructions: InstructionInstance):
        self._instructions += instructions

    def tmp(self) -> str:
        return f"__tmp{self.tmp_num()}"

    def tmp_num(self) -> int:
        self._tmp_index += 1
        return self._tmp_index

    def get_instructions(self) -> list[InstructionInstance]:
        return copy.deepcopy(self._instructions)

    @abstractmethod
    def generate_node(self, node: 'Node') -> 'Value':
        raise NotImplementedError

    @abstractmethod
    def error(self, msg: str):
        raise NotImplementedError

    @abstractmethod
    def current_pos(self) -> Position:
        raise NotImplementedError
