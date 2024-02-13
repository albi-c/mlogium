from .lexer import Lexer
from .parser import Parser
from .compiler import Compiler

CODE = """\
fn do_add(a: num, b: num) -> num {
    return add(a, b)
}

fn apply_func(func: fn(num, num) -> num) -> num {
    return func(3, 5)
}

let func: fn(num, num) -> num = do_add

print(do_add(2, apply_func(func)))"""

tokens = Lexer().lex(CODE, "<main>")
ast = Parser(tokens).parse()
compiler = Compiler()
compiler.visit(ast)
print(list(map(str, compiler.ctx.get_instructions())))
