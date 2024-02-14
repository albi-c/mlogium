from .lexer import Lexer
from .parser import Parser
from .compiler import Compiler
from .linker import Linker

CODE = """\
fn do_add(a: num, b: num) -> num {
    add(a, b)
}

fn apply_func(func: fn(num, num) -> num) -> num {
    func(3, 5)
}

let func: fn(num, num) -> num = do_add;
apply_func(|a: num, b: num| -> num { return add(a, b) });

print(do_add(2, apply_func(func)));
print(do_add(1, 2));

draw.clear(1, 2, 3);"""

tokens = Lexer().lex(CODE, "<main>")
ast = Parser(tokens).parse()
compiler = Compiler()
try:
    compiler.visit(ast)
except AssertionError as e:
    compiler.current_node.pos.print()
    raise e
result = Linker.link(compiler.ctx.get_instructions())
print(result)
