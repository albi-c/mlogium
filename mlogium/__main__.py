from .lexer import Lexer
from .parser import Parser
from .compiler import Compiler
from .linker import Linker
from .error import PositionedException

CODE = """\
fn do_max(a: num, b: num) -> num {
    max(a, b)
}

fn apply_func(func: fn(num, num) -> num) -> num {
    func(3, 5)
}

let func: fn(num, num) -> num = do_max;
apply_func(|a: num, b: num| -> num { return max(a, b) });

print(do_max(2, apply_func(func)));
print(do_max(1, 2));

draw.clear(1, 2, 3);

let block = ExternBlock::conveyor1;
let block2 = getlink(5);

print(sensor.copper(block2));

let unit = radar(RadarFilter::any, RadarFilter::any, RadarFilter::any, RadarSort::distance, block, 1);"""

tokens = Lexer().lex(CODE, "<main>")
ast = Parser(tokens).parse()
compiler = Compiler()
try:
    compiler.compile(ast)
except AssertionError as e:
    compiler.current_node.pos.print()
    raise e
except PositionedException as e:
    print(e.msg)
    e.pos.print()
    exit(1)
result = Linker.link(compiler.ctx.get_instructions())
print(result)
