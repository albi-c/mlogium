from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any
from abc import abstractmethod, ABC

from .lexer import *
from .compilation_context import CompilationContext
from .value import Value


class MacroInput(enum.Enum):
    TYPE = enum.auto()
    TOKEN = enum.auto()
    VALUE_NODE = enum.auto()
    BLOCK_NODE = enum.auto()


@dataclass
class CustomMacroInput:
    func: Callable[[Any], Any]


@dataclass
class RepeatMacroInput:
    inp: MacroInput | CustomMacroInput


@dataclass
class MacroInvocationContext:
    ctx: CompilationContext
    pos: Position
    registry: MacroRegistry


class BaseMacro(ABC):
    Input = MacroInput | CustomMacroInput | RepeatMacroInput

    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def inputs(self) -> tuple[Input, ...]:
        raise NotImplementedError

    def top_level_only(self) -> bool:
        return False

    @abstractmethod
    def invoke(self, ctx: MacroInvocationContext, compiler, params: list) -> Value:
        raise NotImplementedError


class MacroRegistry:
    _macros: dict[str, BaseMacro]

    def __init__(self):
        self._macros = {}

    def add(self, name: str, macro: BaseMacro):
        self._macros[name] = macro

    def get(self, name: str) -> BaseMacro | None:
        return self._macros.get(name)
