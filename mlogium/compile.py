from .lexer import Lexer
from .parser import Parser
from .compiler import Compiler
from .optimizer import Optimizer
from .linker import Linker


def compile_code(code: str, filename: str) -> str:
    tokens = Lexer().lex(code, filename)
    ast = Parser(tokens).parse()
    compiler = Compiler()
    compiler.compile(ast)
    print("\n".join(map(str, compiler.ctx.get_instructions())))
    instructions = Optimizer.optimize(compiler.ctx.get_instructions())
    result = Linker.link(instructions)
    return result
