import os

from .value import *
from .lexer import Lexer
from .parser import Parser
from .node import NamespaceNode


def _parse_code(code: str, filename: str) -> Node:
    tokens = Lexer().lex(code, filename)
    ast = Parser(tokens).parse()
    return ast


def _construct_builtin_types(builtins: dict[str, Value]):
    builtins |= {
        "num": Value.of_type(NumberType()),
        "str": Value.of_type(StringType()),
        "Range": Value.of_type(RangeType()),
        "Tuple": Value(TupleTypeSourceType(), "")
    }


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

    builtins |= {
        "typeof": Value(SpecialFunctionType("typeof", [AnyType()], GenericTypeType(),
                                            lambda _, params: Value.of_type(params[0].type)), ""),
        "#import": Value(SpecialFunctionType("import", [StringType()], AnyType(), _import_impl), ""),
        "#use": Value(SpecialFunctionType("use", [AnyType()], NullType(), _use_impl), "")
    }


def _construct_builtin_functions(builtins: dict[str, Value]):
    def _print_impl(ctx: CompilationContext, params: list[Value]) -> Value:
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
    _construct_special_builtin_functions(builtins)
    _construct_builtin_functions(builtins)
    return builtins
