from .value import *
from .instruction import ALL_INSTRUCTIONS_BASES
from . import enums


BUILTIN_VARS = {
    "@this": Value.variable("@this", Type.BLOCK, True),
    "@thisx": Value.variable("@thisx", Type.NUM, True),
    "@thisy": Value.variable("@thisy", Type.NUM, True),
    "@ipt": Value.variable("@ipt", Type.NUM, True),
    "@timescale": Value.variable("@timescale", Type.NUM, True),
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


def builtin_intrinsic(func: Callable, params_: list[Type], outputs_: set[int] = None, name_: str = None):
    outputs_ = outputs_ if outputs_ is not None else set()
    if name_ is None:
        ins = func(["_"] * len(params_))
        name_ = ins.name
    return Value(
        IntrinsicFunctionType(
            name_,
            [(type_, i in outputs_) for i, type_ in enumerate(params_)],
            lambda ctx, *par: ctx.emit(func(*par))
        ),
        name_,
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
                lambda ctx, *values_, base_=base, name_=name: ctx.emit(
                    base_.make_subcommand_with_constants(name_, *values_)),
                subcommand=name
            )
        BUILTIN_FUNCTIONS[base.name] = Value(IntrinsicSubcommandFunctionType(base.name, subcommands), base.name)

    else:
        BUILTIN_FUNCTIONS[base.name] = Value(IntrinsicFunctionType(
            base.name,
            [(type_, i in base.outputs) for i, type_ in enumerate(base.params)],
            lambda ctx, *values_, base_=base: ctx.emit(base_.make_with_constants(*values_))
        ), base.name)


BUILTIN_ENUMS = {}

for name, (values, prefix, opaque) in enums.ALL_ENUMS.items():
    BUILTIN_ENUMS[name] = Value(BasicType(f"${name}Base"), name, impl=EnumBaseTypeImpl(name, values, prefix, opaque))


BUILTIN_OPERATIONS = {}

for op, args in {
    "max": 2,
    "min": 2,
    "angle": 2,
    "len": 2,
    "noise": 2,
    "abs": 1,
    "log": 1,
    "log10": 1,
    "floor": 1,
    "ceil": 1,
    "sqrt": 1,
    "rand": 1,
    "sin": 1,
    "cos": 1,
    "tan": 1,
    "asin": 1,
    "acos": 1,
    "atan": 1
}.items():
    BUILTIN_OPERATIONS[op] = Value(IntrinsicFunctionType(
        op,
        [(BasicType("num"), True)] + args * [(BasicType("num"), False)],
        lambda ctx, *values_, op_=op: ctx.emit(Instruction.op(op_, *values_))
    ), op)


BUILTIN_SPECIAL = {
    "ExternBlock": Value(BasicType("$ExternBlockSource"), "ExternBlockSource", impl=ExternBlockTypeImpl()),
    "print": Value(IntrinsicFunctionType(
        "print",
        [(Type.ANY, False)],
        lambda ctx, *values_: ctx.emit(*(Instruction.print(val) for val in values_[0].to_strings(ctx)))
    ), "print")
}


BUILTINS = BUILTIN_VARS | BUILTIN_FUNCTIONS | BUILTIN_ENUMS | BUILTIN_OPERATIONS | BUILTIN_SPECIAL
