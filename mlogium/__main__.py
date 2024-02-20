from .lexer import Lexer
from .parser import Parser
from .compiler import Compiler
from .optimizer import Optimizer
from .linker import Linker
from .error import PositionedException
from .macro_impl import MacroRegistry, MACROS

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

let unit = radar(RadarFilter::any, RadarFilter::any, RadarFilter::any, RadarSort::distance, block, 1);

print(80 + ~12);
print("ab" !== 12);
print(@counter);"""

# CODE = """\
# struct Vec2 {
#     let x: num;
#     let y: num;
#
#     const fn add(other: Vec2) -> Vec2 {
#         Vec2(self.x + other.x, self.y + other.y)
#     }
#
#     fn iadd(other: Vec2) {
#         self = self.add(other);
#     }
# }
#
# let v = Vec2(1, 2);
# let v2 = v.add(Vec2(4, -4));
# print(v2.x);
# print(v2.y);"""

# CODE = """\
# print(if 1 < 2 { 1 } else { 2 });"""

# CODE = """\
# for (i in 0..12) {
#     print(i);
# }"""

# CODE = """\
# let func: fn(num) = |x: num| { print(x); };
# func(2);"""

# CODE = """\
# if (true) {
#     print(1);
# } else {
#     print(0);
# }"""

CODE = """\
struct S {
    let a: num;
    let b: (num, num);
}
let x = #cast(S, ExternBlock::cell1[0]);
print(x);"""

macro_registry = MacroRegistry()
for macro in MACROS:
    macro_registry.add(macro.name, macro)
tokens = Lexer().lex(CODE, "<main>")
ast = Parser(tokens, macro_registry).parse()
compiler = Compiler()
try:
    compiler.compile(ast)
except AssertionError as e:
    compiler.current_node.pos.print()
    raise e
except PositionedException as e:
    print(e.msg)
    e.pos.print()
    raise e
instructions = Optimizer.optimize(compiler.ctx.get_instructions().copy())
result = Linker.link(instructions)
print(result)
