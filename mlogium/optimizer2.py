from __future__ import annotations

import itertools
from collections import defaultdict

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
        for _ in range(5):
            cls._propagate_constants(blocks)
            cls._remove_unused(blocks)
            cls._remove_noops(blocks)
            # TODO: join blocks
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
            cls._make_ssa_internal(blocks[0], {})

    @classmethod
    def _make_ssa_internal(cls, block: Block, variables: dict[str, int]):
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
            block.variables[name] = block.variables.get(name, 0) + 1
            block.insert(0, Phi(name, f"{name}:{block.variables[name]}", blocks))

        for ins in block:
            if ins.name == Phi.name:
                continue

            for i in ins.inputs:
                inp = ins.params[i]
                if inp in block.variables:
                    ins.params[i] += f":{block.variables[inp]}"
            for i in ins.outputs:
                out = ins.params[i]
                block.variables[out] = block.variables.get(out, 0) + 1
                ins.params[i] += f":{block.variables[out]}"

        block.is_ssa = True

        for b in block.successors:
            cls._make_ssa_internal(b, block.variables)

    @classmethod
    def _resolve_ssa(cls, blocks: Blocks):
        for block in blocks:
            block.add_phi.clear()

        for block in blocks:
            for i, ins in enumerate(block):
                if ins.name == Phi.name:
                    assert isinstance(ins, Phi)
                    for b in ins.input_blocks:
                        b.add_phi.append(Instruction.set(ins.output, f"{ins.variable}:{b.variables[ins.variable]}"))
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
    def _propagate_constants(cls, blocks: Blocks):
        if len(blocks) > 0:
            for block in blocks:
                block.constants = None

            cls._propagate_constants_inner(blocks[0])

    @classmethod
    def _propagate_constants_inner(cls, block: Block):
        if block.constants is not None:
            return
        block.constants = {}

        for b in block.predecessors:
            cls._propagate_constants_inner(b)

        constants = {}
        conflicts = set()
        for b in block.predecessors:
            for k, v in b.constants.items():
                if k in constants:
                    if v != constants[k]:
                        conflicts.add(k)
                constants[k] = v
        for v in conflicts:
            del constants[v]

        block.constants = constants

        for ins in block:
            for i in ins.inputs:
                inp = ins.params[i]
                if inp in constants:
                    ins.params[i] = constants[inp]

            if ins.name == Instruction.set.name:
                constants[ins.params[0]] = ins.params[1]

        for b in block.successors:
            cls._propagate_constants_inner(b)

    @classmethod
    def _remove_unused(cls, blocks: Blocks) -> bool:
        used = set()

        uses = defaultdict(int)

        for block in blocks:
            for ins in block:
                print(ins.params, ins.outputs, ins.inputs)
                for i in ins.inputs:
                    used.add(ins.params[i])
                for p in ins.params:
                    uses[p] += 1

        for block in blocks:
            for i, ins in enumerate(block):
                if not ins.side_effects and all(ins.params[j] not in used for j in ins.outputs):
                    print("UU", ins, [(ins.params[j], ins.params[j] not in used, uses[ins.params[j]]) for j in ins.outputs])
                    block[i] = Instruction.noop()
