from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable


class Type(ABC):
    NUM: BasicType
    STR: BasicType
    NULL: NullType
    BLOCK: BasicType
    UNIT: BasicType
    TEAM: BasicType
    UNIT_TYPE: BasicType
    ITEM_TYPE: BasicType
    BLOCK_TYPE: BasicType
    LIQUID_TYPE: BasicType
    CONTROLLER: BasicType
    COLOR: BasicType
    CONTENT: UnionType
    RANGE: BasicType
    ANY: AnyType

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        return str(self)

    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    def contains(self, other: Type) -> bool:
        return other == self

    def is_opaque(self) -> bool:
        return False


class AnyType(Type):
    def __str__(self):
        return "?"

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

    def is_opaque(self) -> bool:
        return self.name.startswith("$")


class UnionType(Type):
    types: list[Type]

    def __init__(self, types: list[Type]):
        self.types = types

    def __str__(self):
        return " | ".join(map(str, self.types))

    def __eq__(self, other):
        return isinstance(other, UnionType) and other.types == self.types

    def contains(self, other: Type) -> bool:
        if isinstance(other, UnionType):
            return all(any(t_.contains(t) for t_ in self.types) for t in other.types)

        return any(t.contains(other) for t in self.types)

    def is_opaque(self) -> bool:
        return any(type_.is_opaque() for type_ in self.types)


class FunctionType(Type):
    params: list[Type]
    ret: Type | None

    @staticmethod
    def format_type(params: list[Type], ret: Type | None) -> str:
        return f"({', '.join(str(p) for p in params)}) -> {str(ret) if ret is not None else '?'}"

    @staticmethod
    def format_type_optional(params: list[Type | None], ret: Type | None):
        return f"({', '.join(str(p) if p is not None else '?' for p in params)}) -> {str(ret) if ret is not None else '?'}"

    @staticmethod
    def format_type_named(params: list[tuple[str, Type]], ret: Type | None) -> str:
        return f"({', '.join(p[0] + ': ' + str(p[1]) for p in params)}) -> {str(ret) if ret is not None else '?'}"

    @staticmethod
    def format_type_named_ref(params: list[tuple[str, Type, bool]], ret: Type | None) -> str:
        return f"({', '.join(('&' if p[2] else '') + p[0] + ': ' + str(p[1]) for p in params)}) -> {str(ret) if ret is not None else '?'}"

    @staticmethod
    def format_type_named_optional_ref(params: list[tuple[str, Type | None, bool]], ret: Type | None) -> str:
        return f"({', '.join(('&' if p[2] else '') + p[0] + ': ' + (str(p[1]) if p[1] is not None else '?') for p in params)}) -> {str(ret) if ret is not None else '?'}"

    def __init__(self, params: list[Type], ret: Type | None):
        self.params = params
        self.ret = ret

    def __str__(self):
        return FunctionType.format_type(self.params, self.ret)

    def __eq__(self, other):
        return isinstance(other, FunctionType) and other.params == self.params and other.ret == self.ret


class NamedParamFunctionType(FunctionType):
    named_params: list[tuple[str, Type, bool]]

    def __init__(self, named_params: list[tuple[str, Type, bool]], ret: Type | None):
        super().__init__([p[1] for p in named_params], ret)

        self.named_params = named_params

    def __str__(self):
        return FunctionType.format_type_named_ref(self.named_params, self.ret)

    def __eq__(self, other):
        return isinstance(other, NamedParamFunctionType) and other.named_params == self.named_params and other.ret == self.ret


class LambdaType(Type):
    params: list[Type | None]
    ret: Type | None
    captures: list[tuple[str, bool, 'Node']]
    named_params: list[tuple[str, Type | None, bool]]

    attributes: dict

    def __init__(self, named_params: list[tuple[str, Type | None, bool]], ret: Type | None, captures: list[tuple[str, bool, 'Node']], attributes: dict):
        self.params = [p[1] for p in named_params]
        self.ret = ret
        self.captures = captures
        self.named_params = named_params
        self.attributes = attributes

    def __str__(self):
        return FunctionType.format_type_named_optional_ref(self.named_params, self.ret)

    def __eq__(self, other):
        return isinstance(other, LambdaType) and other.named_params == self.named_params and other.ret == self.ret


class WrapperStructInstanceType(Type):
    name: str
    wrapped: Type

    def __init__(self, name: str, wrapped: Type):
        self.name = name
        self.wrapped = wrapped

    def __str__(self):
        return f"{self.name}[{self.wrapped}]"

    def __eq__(self, other):
        return isinstance(other, WrapperStructInstanceType) and self.name == other.name and self.wrapped == other.wrapped


class ConcreteFunctionType(Type):
    name: str
    params: list[Type | None]
    named_params: list[tuple[str, Type | None, bool]]
    ret: Type | None

    attributes: dict

    def __init__(self, name: str, named_params: list[tuple[str, Type | None, bool]], ret: Type | None, attributes: dict):
        self.name = name
        self.params = [p[1] for p in named_params]
        self.named_params = named_params
        self.ret = ret
        self.attributes = attributes

    def __str__(self):
        return f"fn {self.name}{FunctionType.format_type_named_optional_ref(self.named_params, self.ret)}"

    def __eq__(self, other):
        return isinstance(other, ConcreteFunctionType) and other.name == self.name and other.named_params == self.named_params and other.ret == self.ret


class SpecialFunctionType(Type):
    params: list[Type | None]
    func: Callable[['CompilationContext', list['Value']], 'Value']

    def __init__(self, params: list[Type | None], func: Callable[['CompilationContext', list['Value']], 'Value']):
        self.params = params
        self.func = func

    def __str__(self):
        return f"fn_special {FunctionType.format_type_optional(self.params, None)}"

    def __eq__(self, other):
        return isinstance(other, SpecialFunctionType) and self.params == other.params and self.func == other.func


class IntrinsicFunctionType(Type):
    name: str
    param_types: list[Type]
    input_types: list[Type]
    params: list[tuple[Type, bool]]
    ret_type: Type
    inputs: list[int]
    outputs: list[int]
    instruction_func: Callable
    subcommand: str | None

    def __init__(self, name: str, params: list[tuple[Type, bool]], instruction_func: Callable, *, subcommand: str = None):
        self.name = name
        self.params = params
        self.param_types = [t for t, _ in params]
        self.inputs = []
        self.outputs = []
        self.input_types = []
        for i, (t, o) in enumerate(params):
            if o:
                self.outputs.append(i)
            else:
                self.inputs.append(i)
                self.input_types.append(t)
        self.instruction_func = instruction_func
        if len(self.outputs) == 0:
            self.ret_type = NullType()
        elif len(self.outputs) == 1:
            self.ret_type = self.param_types[self.outputs[0]]
        else:
            self.ret_type = TupleType([self.param_types[i] for i in self.outputs])
        self.subcommand = subcommand

    def __str__(self):
        return f"fn_intrinsic {self.name}{FunctionType.format_type(self.param_types, self.ret_type)}"

    def __eq__(self, other):
        return isinstance(other, IntrinsicFunctionType) and self.name == other.name and self.params == other.params


class IntrinsicSubcommandFunctionType(Type):
    name: str
    subcommands: dict[str, IntrinsicFunctionType]

    def __init__(self, name: str, subcommands: dict[str, IntrinsicFunctionType]):
        self.name = name
        self.subcommands = subcommands

    def __str__(self):
        return f"fn_intrinsic {self.name}.({', '.join(self.subcommands.keys())})"

    def __eq__(self, other):
        return isinstance(other, IntrinsicSubcommandFunctionType) and self.name == other.name and self.subcommands == other.subcommands


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


Type.NUM = BasicType("num")
Type.STR = BasicType("str")
Type.NULL = NullType()

Type.BLOCK = BasicType("Block")
Type.UNIT = BasicType("Unit")
Type.TEAM = BasicType("Team")

Type.UNIT_TYPE = BasicType("UnitType")
Type.ITEM_TYPE = BasicType("ItemType")
Type.BLOCK_TYPE = BasicType("BlockType")
Type.LIQUID_TYPE = BasicType("LiquidType")

Type.CONTROLLER = BasicType("Controller")

Type.COLOR = BasicType("Color")

Type.CONTENT = UnionType([Type.UNIT_TYPE, Type.ITEM_TYPE, Type.BLOCK_TYPE, Type.LIQUID_TYPE])

Type.RANGE = BasicType("Range")

Type.ANY = AnyType()
