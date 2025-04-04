from .lexer import Lexer
from .parser import Parser
from .compiler import Compiler
from .optimizer import Optimizer
from .linker import Linker
from .instruction import Instruction

from .asm.parser import AsmParser
from .asm.compiler import AsmCompiler


def compile_code(code: str, filename: str, opt_level: int) -> str:
    tokens = Lexer().lex(code, filename)
    ast = Parser(tokens).parse()
    compiler = Compiler()
    compiler.compile(ast)

    instructions = Optimizer.optimize(compiler.ctx.get_instructions(), opt_level + 1)

    jump_added = False
    for label, mod in compiler.ctx.get_modules():
        if not jump_added:
            instructions.append(Instruction.jump_to_start())
            jump_added = True

        instructions.append(Instruction.label(label))

        instructions += Optimizer.optimize(mod, opt_level + 1)

    result = Linker.link(instructions)

    return result


def compile_asm_code(code: str, filename: str) -> str:
    tokens = Lexer().lex(code, filename)
    ast = AsmParser(tokens).parse()
    compiler = AsmCompiler()
    compiler.compile(ast)
    result = Linker.link(compiler.get_instructions())
    return result
