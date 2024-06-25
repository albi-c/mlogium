from .value import *


def _construct_builtin_types(builtins: dict[str, Value]):
    builtins |= {
        "num": Value.of_type(NumberType()),
        "str": Value.of_type(StringType()),
        "Range": Value.of_type(RangeType())
    }


def _construct_builtin_functions(builtins: dict[str, Value]):
    def _print_impl(ctx: CompilationContext, params: list[Value]) -> Value:
        print(params)
        ctx.emit(*(
            Instruction.print(string)
            for string in params[0].to_strings(ctx)
        ))
        return Value.null()

    builtins |= {
        "print": Value(SpecialFunctionType("print", [AnyType()], NullType(), _print_impl), "")
    }


def construct_builtins() -> dict[str, Value]:
    builtins = {}
    _construct_builtin_types(builtins)
    _construct_builtin_functions(builtins)
    return builtins
