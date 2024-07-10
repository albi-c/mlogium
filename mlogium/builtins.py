import os

from .value import *
from .lexer import Lexer
from .parser import Parser
from .node import NamespaceNode
from .enums import ALL_ENUMS
from .instruction import ALL_INSTRUCTIONS_BASES
from .value_types import TypeRef


def _parse_code(code: str, filename: str) -> Node:
    tokens = Lexer().lex(code, filename)
    ast = Parser(tokens).parse()
    return ast


def _construct_builtin_types(builtins: dict[str, Value]):
    builtins |= {
        "num": Value.of_type(NumberType()),
        "str": Value.of_type(StringType()),
        "Range": Value.of_type(RangeType()),
        "Block": Value.of_type(BlockType()),
        "Unit": Value.of_type(UnitType()),
        "Controller": Value.of_type(ControllerType()),
        "Tuple": Value(TupleTypeSourceType(), "")
    }


def _construct_builtin_enums(builtins: dict[str, Value]):
    for name, (values, content, non_copyable) in ALL_ENUMS.items():
        builtins[name] = Value(BuiltinEnumBaseType(name, values, content, not non_copyable), "")

    builtins |= {
        "Content": Value.of_type(UnionType([
            builtins["UnitType"].type.wrapped_type(None),
            builtins["ItemType"].type.wrapped_type(None),
            builtins["BlockType"].type.wrapped_type(None),
            builtins["LiquidType"].type.wrapped_type(None)
        ]))
    }


def _construct_special_builtin_variables(builtins: dict[str, Value]):
    builtins |= {
        "ExternBlock": Value(BlockSourceType(), "")
    }


def _construct_builtin_variables(builtins: dict[str, Value]):
    builtins |= {
        "@this": Value(BlockType(), "@this", True),
        "@thisx": Value(NumberType(), "@thisx", True),
        "@thisy": Value(NumberType(), "@thisy", True),
        "@ipt": Value(NumberType(), "@ipt", True),
        "@timescale": Value(NumberType(), "@timescale", True),
        "@counter": Value(NumberType(), "@counter", True),
        "@links": Value(NumberType(), "@links", True),
        "@unit": Value(NumberType(), "@unit", False),
        "@time": Value(NumberType(), "@time", True),
        "@tick": Value(NumberType(), "@tick", True),
        "@second": Value(NumberType(), "@second", True),
        "@minute": Value(NumberType(), "@minute", True),
        "@waveNumber": Value(NumberType(), "@waveNumber", True),
        "@waveTime": Value(NumberType(), "@waveTime", True),
        "@mapw": Value(NumberType(), "@mapw", True),
        "@maph": Value(NumberType(), "@maph", True),

        "@ctrlProcessor": Value(ControllerType(), "@ctrlProcessor", True),
        "@ctrlPlayer": Value(ControllerType(), "@ctrlPlayer", True),
        "@ctrlCommand": Value(ControllerType(), "@ctrlCommand", True),

        "@solid": Value(builtins["BlockType"].type, "@solid", True),

        "_": Value(UnderscoreType(), "_", False),

        "true": Value(NumberType(), "1", True),
        "false": Value(NumberType(), "0", True),
        "null": Value(NullType(), "null", True)
    }


def _const_eval_int(value: Value) -> int | None:
    if value.value == "true":
        return 1

    elif value.value in ("false", "null"):
        return 0

    else:
        try:
            return int(value.value)
        except ValueError:
            pass

    return None


def _construct_special_builtin_functions(builtins: dict[str, Value]):
    _import_impl_imported: set[str] = set()

    def _import_impl(ctx: CompilationContext, params: list[Value]) -> Value:
        path = params[0].value
        if not (path.startswith("\"") and path.endswith("\"")):
            ctx.error(f"Import path has to be an immediate string value")
        path = path[1:-1]

        if path.startswith("std:"):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stdlib", f"{path[4:]}.mu")
            display_path = params[0].value
        else:
            if ctx.current_pos().file not in ("<main>", "<clip>"):
                search_dir = os.path.dirname(os.path.abspath(ctx.current_pos().file))
                path = os.path.join(search_dir, path)
            else:
                path = os.path.abspath(path)
            display_path = path

        if not os.path.isfile(path):
            ctx.error(f"Can't import file '{display_path}'")

        if path in _import_impl_imported:
            ctx.error(f"Circular imports are not allowed: '{display_path}'")
        _import_impl_imported.add(path)

        code = open(path).read()
        ast = NamespaceNode(ctx.current_pos(), None, _parse_code(code, path))

        result = ctx.generate_node(ast)

        _import_impl_imported.remove(path)

        return result

    def _use_impl(ctx: CompilationContext, params: list[Value]) -> Value:
        type_ = params[0].type
        if not isinstance(type_, NamespaceType):
            ctx.error(f"Namespace is required as a parameter to the 'use' function")

        ctx.scope.scopes[-1].variables = type_.variables | ctx.scope.scopes[-1].variables

        return Value.null()

    def _static_assert_impl(ctx: CompilationContext, params: list[Value]) -> Value:
        if (cond := _const_eval_int(params[0])) is None:
            ctx.error(f"Condition has to be a constant expression")

        msg = params[1].value
        if not (msg.startswith("\"") and msg.endswith("\"")):
            ctx.error(f"Message has to be an immediate string value")
        msg = msg[1:-1]

        if not cond:
            ctx.error(f"Static assertion failed: '{msg}'")

        return Value.null()

    builtins |= {
        "typeof": Value(SpecialFunctionType("typeof", [AnyType()], GenericTypeType(),
                                            lambda _, params: Value.of_type(params[0].type)), ""),
        "#import": Value(SpecialFunctionType("import", [StringType()], AnyType(), _import_impl), ""),
        "#use": Value(SpecialFunctionType("use", [AnyType()], NullType(), _use_impl), ""),
        "#static_assert": Value(SpecialFunctionType("static_assert", [NumberType(), StringType()], NullType(),
                                                    _static_assert_impl), "")
    }


_BASIC_TYPE_TRANSLATIONS = {
    "NUM": "num",
    "STR": "str",
    "BLOCK": "Block",
    "CONTENT": "Content",
    "UNIT": "Unit",
    "ITEM_TYPE": "ItemType",
    "BLOCK_TYPE": "BlockType",
    "UNIT_TYPE": "UnitType",
    "LIQUID_TYPE": "LiquidType",
    "TEAM": "Team",
    "CONTROLLER": "Controller"
}


def _resolve_type_ref(builtins: dict[str, Value], ref: TypeRef) -> Type:
    if ref.type == "basic":
        data = ref.data[1:] if ref.data.startswith("$") else ref.data
        if data == "ANY":
            return AnyType()
        return builtins[_BASIC_TYPE_TRANSLATIONS.get(data, data)].type.wrapped_type(None)
    elif ref.type == "union":
        return UnionType([_resolve_type_ref(builtins, r) for r in ref.data])
    else:
        raise ValueError(f"Invalid type reference: {ref}")


_MATH_FUNCTIONS = {
    "max": 2, "min": 2, "angle": 2, "len": 2, "noise": 2, "abs": 1, "log": 1, "log10": 1, "floor": 1, "ceil": 1,
    "sqrt": 1, "rand": 1, "sin": 1, "cos": 1, "tan": 1, "asin": 1, "acos": 1, "atan": 1
}


def _construct_builtin_functions(builtins: dict[str, Value]):
    def _print_impl(ctx: CompilationContext, params_: list[Value]) -> Value:
        ctx.emit(*(
            Instruction.print(string)
            for string in params_[0].to_strings(ctx)
        ))
        return Value.null()

    builtins |= {
        "print": Value(SpecialFunctionType("print", [AnyType()], NullType(), _print_impl), "")
    }

    for base in ALL_INSTRUCTIONS_BASES:
        if base.base_params.get("internal", False):
            continue

        if base.has_subcommands():
            subcommands = {}
            for name, (params, outputs, side_effects, _) in base.subcommands().items():
                subcommands[name] = Value(IntrinsicFunctionType(
                    f"{base.name}.{name}",
                    [_resolve_type_ref(builtins, p) for p in params],
                    outputs,
                    lambda ctx, params_, base_=base, name_=name: ctx.emit(
                        base_.make_subcommand_with_constants(name_, *params_))
                ), "")
            builtins[base.name] = Value(IntrinsicSubcommandFunctionType(base.name, subcommands), "")

        else:
            builtins[base.name] = Value(IntrinsicFunctionType(
                base.name,
                [_resolve_type_ref(builtins, p) for p in base.params],
                base.outputs,
                lambda ctx, params_, base_=base: ctx.emit(base_.make_with_constants(*params_))
            ), "")

    for name, n_params in _MATH_FUNCTIONS.items():
        builtins[name] = Value(IntrinsicFunctionType(
            name,
            [NumberType()] * (1 + n_params),
            [0],
            lambda ctx, params_, name_=name: ctx.emit(Instruction.op(name_, *params_))
        ), "")


def construct_builtins() -> dict[str, Value]:
    builtins = {}
    _construct_builtin_types(builtins)
    _construct_builtin_enums(builtins)
    _construct_special_builtin_variables(builtins)
    _construct_builtin_variables(builtins)
    _construct_special_builtin_functions(builtins)
    _construct_builtin_functions(builtins)
    return builtins
