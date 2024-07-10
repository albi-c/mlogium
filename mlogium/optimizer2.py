from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Callable

from .linking_context import LinkingContext
from .instruction import Instruction, InstructionInstance


class Block(list[InstructionInstance]):
    predecessors: set[Block]
    successors: set[Block]
    variables: dict[str, int]
    assignments: set[str]
    is_ssa: bool
    add_phi: list[InstructionInstance]
    referenced: bool
    constants: dict[str, str] | None

    def __init__(self):
        super().__init__()

        self.predecessors = set()
        self.successors = set()
        self.variables = {}
        self.assignments = set()
        self.is_ssa = False
        self.add_phi = []
        self.referenced = False
        self.constants = None

    def __eq__(self, other):
        return isinstance(other, Block) and self is other

    def __hash__(self):
        return id(self)


class Phi(InstructionInstance):
    name = "$phi"

    variable: str
    output: str
    input_blocks: list[Block]

    def __init__(self, variable: str, output: str, input_blocks: set[Block]):
        self.input_blocks = list(input_blocks)
        super().__init__(Instruction.noop, [0], False, {}, Phi.name,
                         output,
                         *(f"{variable}:{block.variables[variable]}" for block in self.input_blocks),
                         internal=True)

        self.variable = variable
        self.output = output

    def __str__(self):
        return f"phi {self.output} = {self.variable} \
[{', '.join(f'{v} - {str(id(b))}' for v, b in zip(self.params[1:], self.input_blocks))}]"

    def translate_in_linker(self, ctx: LinkingContext) -> list[InstructionInstance]:
        raise RuntimeError("Phi instruction must be converted")


type Instructions = list[InstructionInstance]
type Blocks = list[Block]


class Optimizer:
    OP_CONSTANTS: list[tuple[tuple[str, ...], tuple[tuple[tuple[str, ...], tuple[str, ...], str | None], ...]]] = [
        (("add", "or", "xor"), (
            (("0",), ("0",), None),
        )),
        (("sub",), (
            (("0",), ("0",), None),
            (tuple(), tuple(), "0")
        )),
        (("mul",), (
            (("0",), ("0",), "0"),
            (("1",), ("1",), None)
        )),
        (("div", "idiv"), (
            (tuple(), tuple(), "1"),
            (tuple(), ("1",), None),
            (("0",), tuple(), "0")
        )),
        (("shr", "shl"), (
            (tuple(), ("0",), None),
        )),
        (("and", "land", "or"), (
            (("0",), ("0",), "0"),
            (tuple(), tuple(), None)
        ))
    ]

    OP_PRECALC: dict[str, Callable[[int | float, int | float | None], int | float | bool]] = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b,
        "idiv": lambda a, b: a // b,
        "mod": lambda a, b: a % b,
        "pow": lambda a, b: a ** b,
        "not": lambda a, _: not a,
        "land": lambda a, b: a and b,
        "lessThan": lambda a, b: a < b,
        "lessThanEq": lambda a, b: a <= b,
        "greaterThan": lambda a, b: a > b,
        "greaterThanEq": lambda a, b: a >= b,
        "strictEqual": lambda a, b: a == b,
        "equal": lambda a, b: a == b,
        "notEqual": lambda a, b: a != b,
        "shl": lambda a, b: a << b,
        "shr": lambda a, b: a >> b,
        "or": lambda a, b: a | b,
        "and": lambda a, b: a & b,
        "xor": lambda a, b: a ^ b,
        "max": lambda a, b: max(a, b),
        "min": lambda a, b: min(a, b),
        "abs": lambda a, _: abs(a),
        "log": lambda a, _: math.log(a),
        "log10": lambda a, _: math.log10(a),
        "floor": lambda a, _: math.floor(a),
        "ceil": lambda a, _: math.ceil(a),
        "sqrt": lambda a, _: math.sqrt(a),
        "angle": lambda a, b: math.atan2(b, a) * 180 / math.pi,
        "length": lambda a, b: math.sqrt(a * a + b * b),
        "sin": lambda a, _: math.sin(math.radians(a)),
        "cos": lambda a, _: math.cos(math.radians(a)),
        "tan": lambda a, _: math.tan(math.radians(a)),
        "asin": lambda a, _: math.degrees(math.asin(a)),
        "acos": lambda a, _: math.degrees(math.acos(a)),
        "atan": lambda a, _: math.degrees(math.atan(a))
    }

    JUMP_PRECALC: dict[str, Callable[[int | float, int | float], bool]] = {
        "equal": lambda a, b: a == b,
        "notEqual": lambda a, b: a != b,
        "greaterThan": lambda a, b: a > b,
        "lessThan": lambda a, b: a < b,
        "greaterThanEq": lambda a, b: a >= b,
        "lessThanEq": lambda a, b: a <= b
    }

    @classmethod
    def optimize(cls, code: Instructions):
        cls._remove_unused_labels(code)
        blocks = cls._make_blocks(code)
        cls._remove_noops(blocks)
        cls._remove_empty_blocks(blocks)
        cls._eval_block_jumps(blocks)
        cls._find_assignments(blocks)
        cls._make_ssa(blocks)
        cls._remove_noops(blocks)
        for _ in range(max(5, len(code) // 10)):
            while cls._propagate_constants(blocks) or cls._precalculate_ops(blocks) or cls._precalculate_jumps(blocks):
                cls._remove_unused(blocks)
                cls._remove_noops(blocks)
        cls._resolve_ssa(blocks)
        cls._remove_empty_blocks(blocks)
        code = [ins for block in blocks for ins in block if ins.name != Instruction.noop.name]

        return code

    @classmethod
    def _remove_unused_labels(cls, code: Instructions):
        used = set()
        for ins in code:
            if ins.name == Instruction.jump.name:
                used.add(ins.params[0][1:])
        for i, ins in enumerate(code):
            if ins.name == Instruction.label.name:
                if ins.params[0] not in used:
                    code[i] = Instruction.noop()

    @classmethod
    def _make_blocks(cls, code: Instructions) -> Blocks:
        blocks = [Block()]
        for ins in code:
            if ins.name == Instruction.label.name:
                blocks.append(Block())

            blocks[-1].append(ins)

            if ins.name == Instruction.jump.name:
                blocks.append(Block())

        return blocks

    @classmethod
    def _remove_empty_blocks(cls, blocks: Blocks):
        blocks[:] = [block for block in blocks if len(block) > 0]

    @classmethod
    def _eval_block_jumps(cls, blocks: Blocks):
        if len(blocks) == 0:
            return

        labels = {"$" + lab.params[0]: i for i, block in enumerate(blocks)
                  for lab in block if lab.name == Instruction.label.name}

        visited = set()
        cls._eval_block_jumps_internal(blocks, labels, 0, visited)

        blocks[:] = [block for i, block in enumerate(blocks) if i in visited]
        for block in blocks:
            block.predecessors = {b for b in block.predecessors if b in blocks}
            for b in block.predecessors:
                b.successors.add(block)

    @classmethod
    def _eval_block_jumps_internal(cls, blocks: Blocks, labels: dict[str, int], i: int, visited: set[int],
                                   from_: int = None):
        if i >= len(blocks):
            return

        if from_ is not None:
            blocks[i].predecessors.add(blocks[from_])

        if i in visited:
            return
        visited.add(i)

        for j, ins in enumerate(blocks[i]):
            if ins.name == Instruction.jump.name:
                assert j == len(blocks[i]) - 1

                if ins.params[1] == "always":
                    cls._eval_block_jumps_internal(blocks, labels, labels[ins.params[0]], visited, i)
                    return

                else:
                    cls._eval_block_jumps_internal(blocks, labels, labels[ins.params[0]], visited, i)
                    cls._eval_block_jumps_internal(blocks, labels, i + 1, visited, i)
                    return

        cls._eval_block_jumps_internal(blocks, labels, i + 1, visited, i)

    @classmethod
    def _find_assignments(cls, blocks: Blocks):
        for block in blocks:
            for ins in block:
                for i in ins.outputs:
                    block.assignments.add(ins.params[i])

    @classmethod
    def _make_ssa(cls, blocks: Blocks):
        if len(blocks) > 0:
            cls._make_ssa_internal(blocks[0], {}, {})

    @classmethod
    def _make_ssa_internal(cls, block: Block, variables: dict[str, int], variable_numbers: dict[str, int]):
        if block.is_ssa:
            return

        block.variables = variables.copy()

        phi_required = {}
        for a, b in itertools.combinations(block.predecessors, 2):
            for common in a.assignments & b.assignments:
                if common in phi_required:
                    phi_required[common] |= {a, b}
                else:
                    phi_required[common] = {a, b}

        for name, blocks in phi_required.items():
            block.variables[name] = variable_numbers[name] = variable_numbers.get(name, 0) + 1

        for ins in block:
            if ins.name == Phi.name:
                continue

            for i in ins.inputs:
                inp = ins.params[i]
                if inp in block.variables:
                    ins.params[i] += f":{variable_numbers[inp]}"
            for i in ins.outputs:
                out = ins.params[i]
                block.variables[out] = variable_numbers[out] = variable_numbers.get(out, 0) + 1
                ins.params[i] += f":{variable_numbers[out]}"

        block.is_ssa = True

        for b in block.successors:
            cls._make_ssa_internal(b, block.variables, variable_numbers)

        for name, blocks in phi_required.items():
            block.insert(0, Phi(name, f"{name}:{block.variables[name]}", blocks))

    @classmethod
    def _resolve_ssa(cls, blocks: Blocks):
        for block in blocks:
            block.add_phi.clear()

        for block in blocks:
            for i, ins in enumerate(block):
                if ins.name == Phi.name:
                    assert isinstance(ins, Phi)
                    for b, v in zip(ins.input_blocks, ins.params[1:]):
                        b.add_phi.append(Instruction.set(ins.output, v))
                    block[i] = Instruction.noop()

        for block in blocks:
            if len(block) > 0 and block[-1].name == Instruction.jump.name:
                block[:] = block[:-1] + block.add_phi + [block[-1]]

            else:
                block[:] = block + block.add_phi

    @classmethod
    def _remove_noops(cls, blocks: Blocks):
        for block in blocks:
            block[:] = [ins for ins in block if ins.name != Instruction.noop.name]

    @classmethod
    def _propagate_constants(cls, blocks: Blocks) -> bool:
        constants = {}

        for block in blocks:
            for ins in block:
                if ins.name == Instruction.set.name:
                    constants[ins.params[0]] = ins.params[1]

        found = False
        for block in blocks:
            for ins in block:
                for i in ins.inputs:
                    inp = ins.params[i]
                    if inp in constants:
                        ins.params[i] = constants[inp]
                        found = True

        return found

    @classmethod
    def _precalculate_ops(cls, blocks: Blocks) -> bool:
        found = False
        for block in blocks:
            for i, ins in enumerate(block):
                if ins.name == Instruction.op.name:
                    result = None
                    for operations, patterns in Optimizer.OP_CONSTANTS:
                        if ins.params[0] not in operations:
                            continue

                        for a, b, r in patterns:
                            if r is None:
                                if ins.params[2] in a:
                                    result = ins.params[3]
                                    break
                                elif ins.params[3] in b:
                                    result = ins.params[2]
                                    break

                            else:
                                if len(a) == 0 and len(b) == 0:
                                    if ins.params[2] == ins.params[3]:
                                        result = r
                                        break

                                else:
                                    if ins.params[2] in a or ins.params[3] in b:
                                        result = r
                                        break

                        break

                    if result is not None:
                        if isinstance(result, bool) or (isinstance(result, float) and result.is_integer()):
                            result = int(result)
                        block[i] = Instruction.set(ins.params[1], str(result))
                        found = True
                        continue

                    if (func := cls.OP_PRECALC.get(ins.params[0])) is not None:
                        try:
                            a = float(ins.params[2])
                            if a.is_integer():
                                a = int(a)
                            if ins.params[3] == "_":
                                b = None
                            else:
                                b = float(ins.params[3])
                                if b.is_integer():
                                    b = int(b)

                            result = func(a, b)
                            if isinstance(result, bool) or (isinstance(result, float) and result.is_integer()):
                                result = int(result)

                            block[i] = Instruction.set(ins.params[1], str(result))
                            found = True
                            continue

                        except (ArithmeticError, ValueError):
                            pass

        return found

    @classmethod
    def _precalculate_jumps(cls, blocks: Blocks) -> bool:
        found = False
        for block in blocks:
            for i, ins in enumerate(block):
                if ins.name == Instruction.jump.name:
                    if (func := cls.JUMP_PRECALC.get(ins.params[1])) is not None:
                        try:
                            a = float(ins.params[2])
                            if a.is_integer():
                                a = int(a)
                            b = float(ins.params[3])
                            if b.is_integer():
                                b = int(b)

                            if func(a, b):
                                block[i] = Instruction.jump_always(ins.params[0][1:])
                            else:
                                block[i] = Instruction.noop()
                            found = True

                        except (ArithmeticError, ValueError):
                            pass

        return found

    @classmethod
    def _remove_unused(cls, blocks: Blocks):
        used = set()
        uses = defaultdict(int)

        for block in blocks:
            for ins in block:
                for i in ins.inputs:
                    used.add(ins.params[i])
                for p in ins.params:
                    uses[p] += 1

        for block in blocks:
            for i, ins in enumerate(block):
                if not ins.side_effects and all(ins.params[j] not in used for j in ins.outputs):
                    block[i] = Instruction.noop()
