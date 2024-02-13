from .value import *


BUILTINS = {
    "print": Value(IntrinsicFunctionType(
        "print",
        [
            (AnyType(), False)
        ],
        lambda ctx, val: ctx.emit(*(Instruction.print(s) for s in val.to_strings(ctx)))
    ), "print"),
    "add": Value(IntrinsicFunctionType(
        "add",
        [
            (BasicType("num"), True),
            (BasicType("num"), False),
            (BasicType("num"), False)
        ],
        lambda ctx, *values: ctx.emit(Instruction("op", "add", *values))
    ), "add")
}
