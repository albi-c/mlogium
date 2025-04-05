from __future__ import annotations

import copy
from abc import ABC, abstractmethod
import contextlib

from .instruction import InstructionInstance
from .util import Position


class ErrorContext(ABC):
    notes: list[tuple[str, Position | None]]

    def __init__(self):
        self.notes = []

    @abstractmethod
    def error(self, msg: str, pos: Position | None = None):
        raise NotImplementedError

    def note(self, msg: str, pos: Position | None = None):
        self.notes.append((msg, pos))


class CompilationContext(ErrorContext, ABC):
    scope: 'ScopeStack'

    _instructions: list[InstructionInstance]
    _tmp_index: int
    _modules: list[tuple[str, list[InstructionInstance]]]

    def __init__(self, scope: 'ScopeStack'):
        super().__init__()

        self.scope = scope

        self._instructions = []
        self._tmp_index = 0
        self._modules = []

    def note(self, msg: str, pos: Position | None = None):
        ErrorContext.note(self, msg, pos if pos is not None else self.current_pos())

    def emit(self, *instructions: InstructionInstance):
        self._instructions += instructions

    def tmp(self) -> str:
        return f"__tmp{self.tmp_num()}"

    def tmp_num(self) -> int:
        self._tmp_index += 1
        return self._tmp_index

    def get_instructions(self) -> list[InstructionInstance]:
        return copy.deepcopy(self._instructions)

    def get_modules(self) -> list[tuple[str, list[InstructionInstance]]]:
        return copy.deepcopy(self._modules)

    @contextlib.contextmanager
    def module(self, label: str):
        instructions = self._instructions
        scope = self.scope
        self._instructions = []
        self.set_scope(scope.module())
        try:
            yield
        finally:
            self._modules.append((label, self._instructions))
            self._instructions = instructions
            self.set_scope(scope)

    @abstractmethod
    def generate_node(self, node: 'Node') -> 'Value':
        raise NotImplementedError

    @abstractmethod
    def current_pos(self) -> Position:
        raise NotImplementedError

    def set_scope(self, scope: 'ScopeStack'):
        self.scope = scope
