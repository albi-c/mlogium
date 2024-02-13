from __future__ import annotations

from abc import ABC, abstractmethod


class Type(ABC):
    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    def contains(self, other: Type) -> bool:
        return other == self


class AnyType(Type):
    def __str__(self):
        return "any"

    def __eq__(self, other):
        return isinstance(other, AnyType)

    def contains(self, other: Type) -> bool:
        return True


class BasicType(Type):
    name: str

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, BasicType) and other.name == self.name


class OpaqueType(BasicType):
    def __eq__(self, other):
        return False

    def contains(self, other: Type) -> bool:
        return False


class UnionType(Type):
    types: list[Type]

    def __init__(self, types: list[Type]):
        self.types = types

    def __str__(self):
        return " | ".join(map(str, sorted(self.types, key=hash)))

    def __eq__(self, other):
        return isinstance(other, UnionType) and other.types == self.types

    def contains(self, other: Type) -> bool:
        if isinstance(other, UnionType):
            return all(t in self.types for t in other.types)

        return other in self.types


class FunctionType(Type):
    params: list[Type]
    ret: Type

    @staticmethod
    def format_type(params: list[Type], ret: Type) -> str:
        return f"({', '.join(str(p) for p in params)}) -> {str(ret)}"

    @staticmethod
    def format_type_named(params: list[tuple[str, Type]], ret: Type) -> str:
        return f"({', '.join(p[0] + ': ' + str(p[1]) for p in params)}) -> {str(ret)}"

    def __init__(self, params: list[Type], ret: Type):
        self.params = params
        self.ret = ret

    def __str__(self):
        return FunctionType.format_type(self.params, self.ret)

    def __eq__(self, other):
        return isinstance(other, FunctionType) and other.params == self.params and other.ret == self.ret


class NamedParamFunctionType(FunctionType):
    named_params: list[tuple[str, Type]]

    def __init__(self, named_params: list[tuple[str, Type]], ret: Type):
        super().__init__([p[1] for p in named_params], ret)

        self.named_params = named_params

    def __str__(self):
        return FunctionType.format_type_named(self.named_params, self.ret)

    def __eq__(self, other):
        return isinstance(other, NamedParamFunctionType) and other.named_params == self.named_params and other.ret == self.ret


class ConcreteFunctionType(Type):
    name: str
    params: list[Type]
    named_params: list[tuple[str, Type]]
    ret: Type

    attributes: dict

    def __init__(self, name: str, named_params: list[tuple[str, Type]], ret: Type, attributes: dict):
        self.name = name
        self.params = [p[1] for p in named_params]
        self.named_params = named_params
        self.ret = ret
        self.attributes = attributes

    def __str__(self):
        return f"fn {self.name}{FunctionType.format_type_named(self.named_params, ret)}"

    def __eq__(self, other):
        return isinstance(other, FunctionType) and other.params == self.params and other.ret == self.ret


class IntrinsicFunctionType(Type):
    name: str
    param_types: list[Type]
    params: list[tuple[Type, bool]]
    ret_type: Type
    inputs: list[int]
    outputs: list[int]
    instruction_func: Callable[[str, tuple[Value, ...]], Instruction]

    def __init__(self, name: name, params: list[tuple[Type, bool]], instruction_func: Callable[[str, tuple[Value, ...]], Instruction]):
        self.name = name
        self.params = params
        self.param_types = [t for t, _ in params]
        self.num_inputs = 0
        self.num_outputs = 0
        for i, (_, o) in enumerate(params):
            if o:
                self.outputs.append(i)
            else:
                self.inputs.append(i)
        self.instruction_func = instruction_func
        if len(self.outputs) == 0:
            self.ret_type = NullType()
        elif len(self.outputs) == 1:
            self.ret_type = self.param_types[self.outputs[0]]
        else:
            self.ret_type = TupleType([self.param_types[i] for i in self.outputs])

    def __str__(self):
        return f"fn {self.name}{FunctionType.format_type(self.param_types, self.ret_type)}"

    def __eq__(self, other):
        return isinstance(other, IntrinsicFunctionType) and self.name == other.name and self.params == other.param_types and self.instruction_func == other.instruction_func


class TupleType(Type):
    types: list[Type]

    def __init__(self, types: list[Type]):
        self.types = types

    def __str__(self):
        return f"({', '.join(map(str, self.types))})"

    def __eq__(self, other):
        return isinstance(other, TupleType) and other.types == self.types


class NullType(TupleType):
    def __init__(self):
        super().__init__([])

    def __str__(self):
        return "()"
