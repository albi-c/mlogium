from .lexer import Lexer
from .parser import Parser
from .compiler import Compiler
from .optimizer import Optimizer
from .linker import Linker

from .asm.parser import AsmParser
from .asm.compiler import AsmCompiler


def compile_code(code: str, filename: str) -> str:
    tokens = Lexer().lex(code, filename)
    ast = Parser(tokens).parse()
    compiler = Compiler()
    compiler.compile(ast)
    instructions = Optimizer.optimize(compiler.ctx.get_instructions())
    result = Linker.link(instructions)
    return result


def compile_asm_code(code: str, filename: str) -> str:
    tokens = Lexer().lex(code, filename)
    ast = AsmParser(tokens).parse()
    compiler = AsmCompiler()
    compiler.compile(ast)
    result = Linker.link(compiler.get_instructions())
    return result
