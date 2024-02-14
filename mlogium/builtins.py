from typing import Callable

from .value import *
from .instruction import ALL_INSTRUCTIONS_BASES


BUILTIN_VARS = {
    "@this": Value.variable("@this", Type.BLOCK, True),
    "@thisx": Value.variable("@thisx", Type.NUM, True),
    "@thisy": Value.variable("@thisy", Type.NUM, True),
    "@ipt": Value.variable("@ipt", Type.NUM, True),
    "@timescale": Value.variable("@ipt", Type.NUM, True),
    "@counter": Value.variable("@counter", Type.NUM, True),
    "@links": Value.variable("@links", Type.NUM, True),
    "@unit": Value.variable("@unit", Type.UNIT, False),
    "@time": Value.variable("@time", Type.NUM, True),
    "@tick": Value.variable("@tick", Type.NUM, True),
    "@second": Value.variable("@second", Type.NUM, True),
    "@minute": Value.variable("@minute", Type.NUM, True),
    "@waveNumber": Value.variable("@waveNumber", Type.NUM, True),
    "@waveTime": Value.variable("@waveTime", Type.NUM, True),
    "@mapw": Value.variable("@mapw", Type.NUM, True),
    "@maph": Value.variable("@maph", Type.NUM, True),

    "@ctrlProcessor": Value.variable("@ctrlProcessor", Type.CONTROLLER, True),
    "@ctrlPlayer": Value.variable("@ctrlPlayer", Type.CONTROLLER, True),
    "@ctrlCommand": Value.variable("@ctrlCommand", Type.CONTROLLER, True),

    "@solid": Value.variable("@solid", Type.BLOCK_TYPE, True),

    "_": Value.variable("_", Type.ANY, False),

    "true": Value.variable("true", Type.NUM, True),
    "false": Value.variable("false", Type.NUM, True),
    "null": Value.variable("null", Type.NULL, True)
}


def builtin_intrinsic(func: Callable, params: list[Type], outputs: set[int] = None, name: str = None):
    outputs = outputs if outputs is not None else set()
    if name is None:
        ins = func(["_"] * len(params))
        name = ins.name
    return Value(
        IntrinsicFunctionType(
            name,
            [(type_, i in outputs) for i, type_ in enumerate(params)],
            lambda ctx, *par: ctx.emit(func(*par))
        ),
        name,
        True
    )


BUILTIN_FUNCTIONS = {}


for base in ALL_INSTRUCTIONS_BASES:
    if base.base_params.get("internal", False):
        continue

    if base.has_subcommands():
        subcommands = {}
        for name, (params, outputs, side_effects, _) in base.subcommands().items():
            subcommands[name] = IntrinsicFunctionType(
                f"{base.name}.{name}",
                [(type_, i in outputs) for i, type_ in enumerate(params)],
                lambda ctx, *values, base_=base, name_=name: ctx.emit(base_.make_subcommand_with_constants(name_, *values)),
                subcommand=name
            )
        BUILTIN_FUNCTIONS[base.name] = Value(IntrinsicSubcommandFunctionType(base.name, subcommands), base.name)

    else:
        BUILTIN_FUNCTIONS[base.name] = Value(IntrinsicFunctionType(
            base.name,
            [(type_, i in base.outputs) for i, type_ in enumerate(base.params)],
            lambda ctx, *values: ctx.emit(base.make_with_constants(*values))
        ), base.name)


BUILTINS = BUILTIN_VARS | BUILTIN_FUNCTIONS


BUILTINS["add"] = Value(IntrinsicFunctionType(
    "add",
    [
        (BasicType("num"), True),
        (BasicType("num"), False),
        (BasicType("num"), False)
    ],
    lambda ctx, *values: ctx.emit(Instruction.op("add", *values))
), "add")
