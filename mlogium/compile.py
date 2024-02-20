from .macro_impl import MacroRegistry, MACROS
from .lexer import Lexer
from .parser import Parser
from .compiler import Compiler
from .optimizer import Optimizer
from .linker import Linker


def compile_code(code: str, filename: str) -> str:
    macro_registry = MacroRegistry()
    for macro in MACROS:
        macro_registry.add(macro.name, macro)
    tokens = Lexer().lex(code, filename)
    ast = Parser(tokens, macro_registry).parse()
    compiler = Compiler()
    compiler.compile(ast)
    instructions = Optimizer.optimize(compiler.ctx.get_instructions().copy())
    return Linker.link(instructions)
