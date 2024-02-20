from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

from .node import *
from .lexer import *


class MacroInput(enum.Enum):
    TYPE = enum.auto()
    TOKEN = enum.auto()
    VALUE_NODE = enum.auto()
    BLOCK_NODE = enum.auto()


@dataclass
class CustomMacroInput:
    func: Callable[[Any], Any]


@dataclass
class MacroInvocationContext:
    pos: Position
    registry: MacroRegistry


class BaseMacro(ABC):
    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def inputs(self) -> tuple[MacroInput, ...]:
        raise NotImplementedError

    def top_level_only(self) -> bool:
        return False

    @abstractmethod
    def invoke_to_str(self, ctx: MacroInvocationContext, params: list) -> str:
        raise NotImplementedError

    @abstractmethod
    def invoke_to_tokens(self, ctx: MacroInvocationContext, params: list) -> list[Token]:
        raise NotImplementedError

    @abstractmethod
    def invoke(self, ctx: MacroInvocationContext, params: list) -> Node:
        raise NotImplementedError


class MacroRegistry:
    _macros: dict[str, BaseMacro]

    def __init__(self):
        self._macros = {}

    def add(self, name: str, macro: BaseMacro):
        self._macros[name] = macro

    def get(self, name: str) -> BaseMacro | None:
        return self._macros.get(name)
