from .lexer import Lexer
from .parser import Parser
from .compiler import Compiler

CODE = """\
let x = (10, 20)
let (y, z) = (x, 5)"""

tokens = Lexer().lex(CODE, "<main>")
ast = Parser(tokens).parse()
compiler = Compiler()
compiler.visit(ast)
print(list(map(str, compiler.ctx.get_instructions())))
