from __future__ import annotations

import contextlib

from .comptime_value import CValue


class ComptimeScopeStack:
    class Scope:
        name: str
        variables: dict[str, CValue]

        def __init__(self, name: str, variables: dict[str, CValue] = None):
            self.name = name
            self.variables = variables if variables is not None else {}

    scopes: list[Scope]

    def __init__(self):
        self.scopes = []

    def push(self, name: str, variables: dict[str, CValue] = None):
        self.scopes.append(ComptimeScopeStack.Scope(name, variables))

    def pop(self):
        self.scopes.pop(-1)

    @contextlib.contextmanager
    def __call__(self, name: str, variables: dict[str, CValue] = None):
        self.scopes.append(ComptimeScopeStack.Scope(name, variables))
        try:
            yield
        finally:
            self.scopes.pop(-1)

    @contextlib.contextmanager
    def bottom(self, name: str, variables: dict[str, CValue] = None):
        if variables is not None:
            self.scopes.insert(0, ComptimeScopeStack.Scope(name, variables))
        try:
            yield
        finally:
            if variables is not None:
                self.scopes.pop(0)

    def get(self, name: str) -> CValue | None:
        for scope in reversed(self.scopes):
            if name in scope.variables:
                return scope.variables[name]

        return None

    def declare(self, name: str, value: CValue) -> bool:
        if name in self.scopes[-1].variables:
            return False

        self.scopes[-1].variables[name] = value
        return True

    def assign(self, name: str, value: CValue) -> bool:
        for scope in reversed(self.scopes):
            if name in scope.variables:
                scope.variables[name] = value
                return True

        return False
