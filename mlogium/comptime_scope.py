from __future__ import annotations

import contextlib

from .comptime_value import VariableCLValue, CValue


class ComptimeScopeStack:
    class Scope:
        name: str
        variables: dict[str, VariableCLValue]

        def __init__(self, name: str, variables: dict[str, VariableCLValue] = None):
            self.name = name
            self.variables = variables if variables is not None else {}

    scopes: list[Scope]

    def __init__(self):
        self.scopes = []

    def push(self, name: str, variables: dict[str, VariableCLValue] = None):
        self.scopes.append(ComptimeScopeStack.Scope(name, variables))

    def pop(self):
        self.scopes.pop(-1)

    @contextlib.contextmanager
    def __call__(self, name: str, variables: dict[str, VariableCLValue] = None):
        self.scopes.append(ComptimeScopeStack.Scope(name, variables))
        try:
            yield
        finally:
            self.scopes.pop(-1)

    @contextlib.contextmanager
    def bottom(self, name: str, variables: dict[str, VariableCLValue] = None):
        if variables is not None:
            self.scopes.insert(0, ComptimeScopeStack.Scope(name, variables))
        try:
            yield
        finally:
            if variables is not None:
                self.scopes.pop(0)

    def _find(self, name: str) -> VariableCLValue | None:
        for scope in reversed(self.scopes):
            if (var := scope.variables.get(name)) is not None:
                return var

        return None

    def get(self, name: str) -> CValue | None:
        if (var := self._find(name)) is not None:
            return var.value

        return None

    def declare(self, name: str, value: CValue, constant: bool) -> bool:
        if name in self.scopes[-1].variables:
            return False

        self.scopes[-1].variables[name] = VariableCLValue(value, constant)
        return True

    def assign(self, name: str, value: CValue) -> bool:
        if (var := self._find(name)) is not None:
            var.set(value)
            return True

        return False

    def capture(self, name: str, ref: bool) -> VariableCLValue | None:
        if (var := self._find(name)) is not None:
            if ref:
                return var

            else:
                return VariableCLValue(var.value, False)

        return None
