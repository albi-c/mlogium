from __future__ import annotations

import contextlib

from .abi import ABI
from .value import Value


class ScopeStack:
    class Scope:
        name: str
        variables: dict[str, Value]

        def __init__(self, name: str):
            self.name = name
            self.variables = {}

    scopes: list[Scope]
    functions: list[str]
    loops: list[str]

    def __init__(self):
        self.scopes = []
        self.functions = []
        self.loops = []

    def get_function(self) -> str | None:
        return self.functions[-1] if len(self.functions) > 0 else None

    def get_loop(self) -> str | None:
        return self.loops[-1] if len(self.loops) > 0 else None

    @contextlib.contextmanager
    def __call__(self, name: str):
        self.scopes.append(ScopeStack.Scope(name))
        try:
            yield
        finally:
            self.scopes.pop(-1)

    @contextlib.contextmanager
    def function_call(self, name: str):
        self.scopes.append(ScopeStack.Scope(name))
        self.functions.append(name)
        try:
            yield
        finally:
            self.scopes.pop(-1)
            self.functions.pop(-1)

    @contextlib.contextmanager
    def loop(self, name: str):
        self.scopes.append(ScopeStack.Scope(name))
        self.loops.append(name)
        try:
            yield
        finally:
            self.scopes.pop(-1)
            self.loops.pop(-1)

    def get(self, name: str) -> Value | None:
        for scope in reversed(self.scopes):
            if name in scope.variables:
                return scope.variables[name]

        return None

    def declare(self, name: str, type_: Type, const_on_write: bool) -> Value | None:
        if name in self.scopes[-1].variables:
            return None

        value = Value.variable(ABI.namespaced_name(self.scopes[-1].name, name), type_, const_on_write=const_on_write)
        self.scopes[-1].variables[name] = value
        return value

    def declare_special(self, name: str, value: Value) -> bool:
        if name in self.scopes[-1].variables:
            return False

        self.scopes[-1].variables[name] = value
        return True
