from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any
from abc import abstractmethod, ABC

from .lexer import *
from .compilation_context import CompilationContext
from .value import Value, Type


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


class Macro(ABC):
    Input = MacroInput | CustomMacroInput | RepeatMacroInput

    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def inputs(self) -> tuple[Input, ...]:
        raise NotImplementedError

    def is_type(self) -> bool:
        return False

    def top_level_only(self) -> bool:
        return False

    def invoke(self, ctx: MacroInvocationContext, compiler, params: list) -> Value:
        raise NotImplementedError

    def type_invoke(self, ctx: MacroInvocationContext, compiler, params: list) -> Type:
        raise NotImplementedError


class MacroRegistry:
    _macros: dict[str, Macro]

    def __init__(self):
        self._macros = {}

    def add(self, name: str, macro: Macro):
        self._macros[name] = macro

    def get(self, name: str) -> Macro | None:
        return self._macros.get(name)
