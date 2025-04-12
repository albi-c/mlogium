import math
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
        "RangeWithStep": Value.of_type(RangeWithStepType()),
        "Block": Value(BlockBaseType(), ""),
        "Unit": Value.of_type(UnitType()),
        "Controller": Value.of_type(ControllerType()),
        "Tuple": Value(TupleTypeSourceType(), ""),
        "Sound": Value(SoundBaseType(), "")
    }


def _construct_builtin_enums(builtins: dict[str, Value]):
    content_values = set()
    for name, (values, content, non_copyable) in ALL_ENUMS.items():
        enum = BuiltinEnumBaseType(name, values, content, not non_copyable)
        builtins[name] = Value(enum, "")

        if content and name != "Property":
            assert len(values & content_values) == 0, values & content_values
            content_values |= values
            builtins |= {f"@{k.replace('-', '_')}": v for k, v in enum.values.items()}

    builtins |= {
        "Content": Value.of_type(UnionType([
            builtins["UnitType"].type.wrapped_type(None),
            builtins["ItemType"].type.wrapped_type(None),
            builtins["BlockType"].type.wrapped_type(None),
            builtins["LiquidType"].type.wrapped_type(None)
        ]))
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
        "@unit": Value(UnitType(), "@unit", False),
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

        "_": Value(UnderscoreType(), "_", False),

        "true": Value.of_number(1),
        "false": Value.of_number(0),
        "null": Value.null(),

        "@pi": Value.of_number(math.pi),
        "@tau": Value.of_number(math.tau),
        "@e": Value.of_number(math.e),
        "@degToRad": Value.of_number(math.pi / 180.0),
        "@radToDeg": Value.of_number(180.0 / math.pi),

        "@blockCount": Value(NumberType(), "@blockCount", True),
        "@unitCount": Value(NumberType(), "@unitCount", True),
        "@itemCount": Value(NumberType(), "@itemCount", True),
        "@liquidCount": Value(NumberType(), "@liquidCount", True)
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

    def _has_attr_impl(ctx: CompilationContext, params: list[Value]) -> Value:
        name = params[1].value
        if not (name.startswith("\"") and name.endswith("\"")):
            ctx.error(f"Name has to be an immediate string value")
        name = name[1:-1]

        return Value.of_boolean(params[0].getattr(ctx, False, name) is not None)

    def _has_static_attr_impl(ctx: CompilationContext, params: list[Value]) -> Value:
        name = params[1].value
        if not (name.startswith("\"") and name.endswith("\"")):
            ctx.error(f"Name has to be an immediate string value")
        name = name[1:-1]

        return Value.of_boolean(params[0].getattr(ctx, True, name) is not None)

    builtins |= {
        "typeof": Value(SpecialFunctionType("typeof", [AnyType()], GenericTypeType(),
                                            lambda _, params: Value.of_type(params[0].type)), ""),
        "#import": Value(SpecialFunctionType("import", [StringType()], AnyType(), _import_impl), ""),
        "#use": Value(SpecialFunctionType("use", [AnyType()], NullType(), _use_impl), ""),
        "#static_assert": Value(SpecialFunctionType("static_assert", [NumberType(), StringType()], NullType(),
                                                    _static_assert_impl), ""),
        "#has_attr": Value(SpecialFunctionType("has_attr", [AnyType(), StringType()], NumberType(), _has_attr_impl), ""),
        "#has_static_attr": Value(SpecialFunctionType("has_static_attr", [AnyType(), StringType()], NumberType(),
                                                      _has_static_attr_impl), ""),
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
    "CONTROLLER": "Controller",
    "ALIGN": "Align",
    "SOUND": "Sound"
}


def _resolve_type_ref(builtins: dict[str, Value], ref: TypeRef) -> Type:
    if ref.type == "basic":
        data = ref.data[1:] if ref.data.startswith("$") else ref.data
        if data == "ANY":
            return AnyType()
        elif data == "ANY_TRIVIAL":
            return AnyTrivialType()
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

    def _proc_read_impl(ctx: CompilationContext, params_: list[Value]) -> Value:
        if not AnyTrivialType().contains(params_[2].type.wrapped_type(ctx)):
            ctx.error("Output type must be trivial")
        res = ctx.tmp()
        ctx.emit(
            Instruction.p_read(res, params_[0].value, params_[1].value)
        )
        return Value(params_[2].type.wrapped_type(ctx), res)

    def _proc_write_impl(ctx: CompilationContext, params_: list[Value]) -> Value:
        ctx.emit(
            Instruction.p_write(params_[2].value, params_[0].value, params_[1].value)
        )
        return Value.null()

    def _rad_to_deg(ctx: CompilationContext, params_: list[Value]) -> Value:
        res = ctx.tmp()
        ctx.emit(
            Instruction.op("mul", res, params_[0].value, 180.0 / math.pi)
        )
        return Value(NumberType(), res)

    def _deg_to_rad(ctx: CompilationContext, params_: list[Value]) -> Value:
        res = ctx.tmp()
        ctx.emit(
            Instruction.op("mul", res, params_[0].value, math.pi / 180.0)
        )
        return Value(NumberType(), res)

    def _load_impl(ctx: CompilationContext, params_: list[Value]) -> Value:
        var_name = params_[0].value
        if not (var_name.startswith("\"") and var_name.endswith("\"")):
            ctx.error(f"Name has to be an immediate string value")
        var_name = var_name[1:-1]
        if any(ch.isspace() for ch in var_name):
            ctx.error("Variable names cannot contain whitespace")

        type_ = params_[1].type.wrapped_type(ctx)
        if not AnyTrivialType().contains(type_):
            ctx.error("Type must be trivial")

        result = Value(type_, ctx.tmp(), False)
        ctx.emit(Instruction.Load(var_name, result.value))
        return result

    def _store_impl(ctx: CompilationContext, params_: list[Value]) -> Value:
        var_name = params_[0].value
        if not (var_name.startswith("\"") and var_name.endswith("\"")):
            ctx.error(f"Name has to be an immediate string value")
        var_name = var_name[1:-1]
        if any(ch.isspace() for ch in var_name):
            ctx.error("Variable names cannot contain whitespace")

        value = params_[1]
        if not AnyTrivialType().contains(value.type):
            ctx.error("Type must be trivial")

        ctx.emit(Instruction.Store(value.value, var_name))

        return Value.null()

    def _lookup_table_gen(start: int, integers: list[int], min_: int) -> tuple[str, int]:
        string = "".join(chr(start + (n - min_)) for n in integers)
        offset = start - min_
        return string, offset

    def _lookup_table_impl(ctx: CompilationContext, params_: list[Value]) -> Value:
        values = params_[1:]
        if len(values) == 0:
            ctx.error(f"lookup_table requires at least one integer")
        integers = []
        for i, val in enumerate(values):
            if not NumberType().contains(val.type):
                ctx.error(f"Value at position {i} has type '{val.type}', expected number")
            try:
                integers.append(int(val.value))
            except ValueError:
                ctx.error(f"Value at position {i} is not a compile time known integer")

        min_ = min(integers)
        max_ = max(integers)
        diff = max_ - min_

        ascii_start = ord("#")
        ascii_end = ord("~")
        max_diff_ascii = ascii_end - ascii_start

        unicode_start = ord("\u0100")
        unicode_end = ord("\uffff")
        max_diff_unicode = unicode_end - unicode_start

        if diff > max_diff_ascii:
            if diff > max_diff_unicode:
                ctx.error(f"Difference between minimum and maximum value is too large ({diff} > {max_diff_unicode})")

            string, offset = _lookup_table_gen(unicode_start, integers, min_)

        else:
            string, offset = _lookup_table_gen(ascii_start, integers, min_)

        return Value.of_lookup_table(ctx, string, offset)

        # result = Value.of_string(f"\"{string}\"").index(ctx, [params_[0]])
        # if diff != 0:
        #     result = result.binary_op_req(ctx, "-", Value.of_number(diff))
        # return result

    builtins |= {
        "print": Value(SpecialFunctionType("print", [AnyType()], NullType(), _print_impl), ""),
        "proc_read": Value(SpecialFunctionType("proc_read", [BlockType(), StringType(), GenericTypeType()],
                                               AnyType(), _proc_read_impl), ""),
        "proc_write": Value(SpecialFunctionType("proc_write", [BlockType(), StringType(), AnyTrivialType()],
                                                NullType(), _proc_write_impl), ""),
        "@deg": Value(SpecialFunctionType("deg", [NumberType()], NumberType(), _rad_to_deg), ""),
        "@rad": Value(SpecialFunctionType("rad", [NumberType()], NumberType(), _deg_to_rad), ""),
        "@load": Value(SpecialFunctionType("load", [StringType(), GenericTypeType()], AnyType(),
                                           _load_impl), ""),
        "@store": Value(SpecialFunctionType("store", [StringType(), AnyType()], NullType(),
                                            _store_impl), ""),
        "@lookup_table": Value(VariadicSpecialFunctionType("lookup_table", [NumberType()], NullType(),
                                                           _lookup_table_impl), ""),

        "status": Value(IntrinsicSubcommandFunctionType("status", {
            "apply": Value(IntrinsicFunctionType(
                "status.apply",
                [builtins["Status"].type.wrapped_type(None), UnitType(), NumberType()],
                [],
                lambda ctx, params_: ctx.emit(Instruction.status("false", params_[0], params_[1], params_[2]))
            ), ""),
            "clear": Value(IntrinsicFunctionType(
                "status.clear",
                [builtins["Status"].type.wrapped_type(None), UnitType()],
                [],
                lambda ctx, params_: ctx.emit(Instruction.status("true", params_[0], params_[1], "0"))
            ), "")
        }), ""),

        "playsound": Value(IntrinsicSubcommandFunctionType("playsound", {
            "global": Value(IntrinsicFunctionType(
                "playsound.global",
                [builtins["Sound"].type.wrapped_type(None), NumberType(), NumberType(), NumberType(), NumberType()],
                [],
                lambda ctx, params_: ctx.emit(Instruction.play_sound("false", params_[0], params_[1], params_[2],
                                                                     params_[3], "0", "0", params_[4]))
            ), ""),
            "positional": Value(IntrinsicFunctionType(
                "playsound.positional",
                [builtins["Sound"].type.wrapped_type(None)] + [NumberType()] * 5,
                [],
                lambda ctx, params_: ctx.emit(Instruction.play_sound("true", params_[0], params_[1], params_[2],
                                                                     "0", params_[3], params_[4], "0", params_[5]))
            ), "")
        }), "")
    }

    for base in ALL_INSTRUCTIONS_BASES:
        if base.base_params.get("internal", False):
            continue

        if base.has_subcommands():
            subcommands = {}
            for name, (params, outputs, side_effects, _) in base.subcommands().items():
                subcommands[name] = Value(IntrinsicFunctionType(
                    f"{base.func}.{name}",
                    [_resolve_type_ref(builtins, p) for p in params],
                    outputs,
                    lambda ctx, params_, base_=base, name_=name: ctx.emit(
                        base_.make_subcommand_with_constants(name_, *params_))
                ), "")
            builtins[base.func] = Value(IntrinsicSubcommandFunctionType(base.func, subcommands), "")

        else:
            builtins[base.func] = Value(IntrinsicFunctionType(
                base.func,
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
    _construct_builtin_variables(builtins)
    _construct_special_builtin_functions(builtins)
    _construct_builtin_functions(builtins)
    return builtins
