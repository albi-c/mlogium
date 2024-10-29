from __future__ import annotations

import contextlib

from .abi import ABI
from .value import Value, Type


class ScopeStack:
    class Scope:
        name: str
        variables: dict[str, Value]

        def __init__(self, name: str, variables: dict[str, Value] = None):
            self.name = name
            self.variables = variables if variables is not None else {}

    scopes: list[Scope]
    functions: list[tuple[str, Type | None]]
    loops: list[str]
    global_closures: list[dict[str, Value]]

    def __init__(self):
        self.scopes = []
        self.functions = []
        self.loops = []
        self.global_closures = []

    def get_function(self) -> tuple[str, Type | None] | None:
        return self.functions[-1] if len(self.functions) > 0 else None

    def get_loop(self) -> str | None:
        return self.loops[-1] if len(self.loops) > 0 else None

    def push(self, name: str, variables: dict[str, Value] = None):
        self.scopes.append(ScopeStack.Scope(name, variables))

    def pop(self):
        self.scopes.pop(-1)

    @contextlib.contextmanager
    def __call__(self, name: str, variables: dict[str, Value] = None):
        self.scopes.append(ScopeStack.Scope(name, variables))
        try:
            yield
        finally:
            self.scopes.pop(-1)

    @contextlib.contextmanager
    def global_closure(self, variables: dict[str, Value]):
        self.global_closures.append(variables)
        try:
            yield
        finally:
            self.global_closures.pop(-1)

    @contextlib.contextmanager
    def bottom(self, name: str, variables: dict[str, Value] = None):
        if variables is not None:
            self.scopes.insert(0, ScopeStack.Scope(name, variables))
        try:
            yield
        finally:
            if variables is not None:
                self.scopes.pop(0)

    @contextlib.contextmanager
    def function_call(self, ctx, name: str, return_type: Type | None, variables: dict[str, Value] = None):
        self.scopes.append(ScopeStack.Scope(name, variables))
        for n, _ in self.functions:
            if n == name:
                ctx.error(f"Recursion is not allowed: '{name}'")
        self.functions.append((name, return_type))
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

    def get_global_closures(self) -> list[dict[str, Value]]:
        return self.global_closures.copy()

    @staticmethod
    def combine_global_closures(closures: list[dict[str, Value]]) -> dict[str, Value]:
        values = {}
        for closure in closures:
            values |= closure
        return values

    def get_closure_variables(self) -> list[dict[str, Value]]:
        values = {}
        for scope in self.scopes:
            values |= scope.variables
        return self.get_global_closures() + [values]

    def get(self, name: str) -> Value | None:
        for scope in reversed(self.scopes):
            if name in scope.variables:
                return scope.variables[name]

        return None

    def declare(self, name: str, type_: Type, const_on_write: bool) -> Value | None:
        if name in self.scopes[-1].variables:
            return None

        value = Value(type_, ABI.namespaced_name(self.scopes[-1].name, name), False, const_on_write=const_on_write)
        self.scopes[-1].variables[name] = value
        return value

    def declare_special(self, name: str, value: Value) -> bool:
        if name in self.scopes[-1].variables:
            return False

        self.scopes[-1].variables[name] = value
        return True
