from __future__ import annotations

from collections import defaultdict
from typing import Callable
import math
import itertools

from .instruction import Instruction, InstructionInstance


class Block(list[InstructionInstance]):
    predecessors: set[Block]
    successors: set[Block]
    variables: dict[str, int]
    assignments: set[str]
    is_ssa: bool
    add_phi: list[InstructionInstance]
    referenced: bool

    def __init__(self):
        super().__init__()

        self.predecessors = set()
        self.successors = set()
        self.variables = {}
        self.assignments = set()
        self.is_ssa = False
        self.add_phi = []
        self.referenced = False

    def __eq__(self, other):
        return isinstance(other, Block) and id(self) == id(other)

    def __hash__(self):
        return id(self)


class Phi(InstructionInstance):
    name = "$phi"

    variable: str
    output: str
    input_blocks: set[Block]

    def __init__(self, variable: str, output: str, input_blocks: set[Block]):
        super().__init__(Instruction.noop, [0], False, {}, Phi.name, internal=True)

        self.variable = variable
        self.output = output
        self.input_blocks = input_blocks

    def __str__(self):
        return f"phi {self.output} = {self.variable} [{", ".join(map(str, self.input_blocks))}]"

    def translate_in_linker(self, _) -> list[InstructionInstance]:
        raise RuntimeError("Phi instruction must be converted")


type Instructions = list[InstructionInstance]
type Blocks = list[Block]


class Optimizer:
    JUMP_TRANSLATION: dict[str, str] = {
        "equal": "notEqual",
        "notEqual": "equal",
        "greaterThan": "lessThanEq",
        "lessThan": "greaterThanEq",
        "greaterThanEq": "lessThan",
        "lessThanEq": "greaterThan"
    }

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

    PRECALC: dict[str, Callable[[int | float, int | float | None], int | float]] = {
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
        "flip": lambda a, _: ~a,
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
    def optimize(cls, code: Instructions) -> Instructions:
        cls._make_function_sets_essential(code)

        cls._optimize_jumps(code)
        cls._remove_noops(code)

        while cls._precalculate_op_jump(code):
            pass
        cls._remove_noops(code)

        blocks = cls._make_blocks(code)
        cls._eval_block_jumps(blocks)
        cls._find_assignments(blocks)
        cls._optimize_block_jumps(blocks)
        cls._make_ssa(blocks)
        while (cls._propagate_constants(blocks) or cls._precalculate_op_jump_blocks(blocks)
               or cls._eliminate_common_subexpressions(blocks)):
            pass
        cls._resolve_ssa(blocks)
        code = [ins for block in blocks for ins in block]

        cls._remove_noops(code)
        cls._optimize_jumps(code)
        cls._remove_noops(code)

        while cls._optimize_set_op(code) or cls._precalculate_op_jump(code):
            pass
        cls._remove_noops(code)

        while cls._join_instructions(code):
            cls._remove_noops(code)

        return code

    @classmethod
    def _make_function_sets_essential(cls, code: Instructions):
        for i, ins in enumerate(code):
            if ins.name == Instruction.set.name and ins.params[0].startswith("%"):
                code[i] = Instruction.essential_set(*ins.params)

    @classmethod
    def _is_label(cls, ins: InstructionInstance):
        return ins.name in (Instruction.label.name, Instruction.prepare_return_address.name)

    @classmethod
    def _is_jump(cls, ins: InstructionInstance):
        return ins.name in (Instruction.jump.name, Instruction.jump_addr.name)

    @classmethod
    def _make_blocks(cls, code: Instructions) -> Blocks:
        blocks: Blocks = [Block()]
        for ins in code:
            if cls._is_label(ins):
                blocks.append(Block())

            blocks[-1].append(ins)

            if cls._is_jump(ins):
                blocks.append(Block())

        return blocks

    @classmethod
    def _eval_block_jumps(cls, blocks: Blocks):
        if len(blocks) == 0:
            return

        labels = {"$" + lab.params[0]: i for i, block in enumerate(blocks)
                  for lab in block if lab.name == Instruction.label.name}

        labels_in_variables = {lab for block in blocks for ins in block for lab in ins.params if lab.startswith("$")}

        used = set()
        cls._eval_block_jumps_internal(blocks, labels, 0, used)
        for label in labels_in_variables:
            cls._eval_block_jumps_internal(blocks, labels, labels[label], used)
            blocks[labels[label]].referenced = True

        blocks[:] = [block for i, block in enumerate(blocks) if i in used]
        for block in blocks:
            block.predecessors = {pre for pre in block.predecessors if pre in blocks}
            for pre in block.predecessors:
                pre.successors.add(block)

    @classmethod
    def _eval_block_jumps_internal(cls, blocks: Blocks, labels: dict[str, int], i: int, used: set[int],
                                   from_: int = None):
        if i >= len(blocks):
            return

        if from_ is not None:
            blocks[i].predecessors.add(blocks[from_])

        if i in used:
            return
        used.add(i)

        for ins in blocks[i]:
            if ins.name == Instruction.jump.name:
                if ins.params[1] == "always":
                    cls._eval_block_jumps_internal(blocks, labels, labels[ins.params[0]], used, i)
                    return

                else:
                    cls._eval_block_jumps_internal(blocks, labels, labels[ins.params[0]], used, i)
                    cls._eval_block_jumps_internal(blocks, labels, i + 1, used, i)
                    return

        return cls._eval_block_jumps_internal(blocks, labels, i + 1, used, i)

    @classmethod
    def _find_assignments(cls, blocks: Blocks):
        for block in blocks:
            for ins in block:
                for i in ins.outputs:
                    block.assignments.add(ins.params[i])

    @classmethod
    def _optimize_block_jumps(cls, blocks: Blocks):
        labels = {"$" + lab.params[0]: i for i, block in enumerate(blocks)
                  for lab in block if lab.name == Instruction.label.name}
        for i, block in enumerate(blocks):
            if len(block) > 0 and block[-1].name == Instruction.jump.name and labels[block[-1].params[0]] == i + 1:
                block.pop(-1)

    @classmethod
    def _make_ssa(cls, blocks: Blocks):
        if len(blocks) < 1:
            return

        variables = {}
        cls._make_ssa_internal(blocks[0], variables)
        for block in blocks:
            if block.referenced:
                cls._make_ssa_internal(block, variables)

    @classmethod
    def _make_ssa_internal(cls, block: Block, variables: dict[str, int]):
        if block.is_ssa:
            return

        block.variables = variables.copy()

        phi_required: dict[str, set[Block]] = {}
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
                    ins.params[i] += ":" + str(block.variables[inp])
            for o in ins.outputs:
                out = ins.params[o]
                block.variables[out] = block.variables.get(out, 0) + 1
                ins.params[o] += ":" + str(block.variables[out])

        block.is_ssa = True

        for suc in block.successors:
            cls._make_ssa_internal(suc, block.variables)

    @classmethod
    def _propagate_constants(cls, blocks: Blocks) -> bool:
        constants: dict[str, str] = {}

        for block in blocks:
            for ins in block:
                if ins.name == Instruction.set.name:
                    constants[ins.params[0]] = ins.params[1]

        found = False
        for block in blocks:
            for ins in block:
                for i in ins.inputs:
                    if ins.params[i] in constants:
                        ins.params[i] = constants[ins.params[i]]
                        found = True

        return found

    @classmethod
    def _precalculate_op_jump_blocks(cls, blocks: Blocks) -> bool:
        found = False
        for block in blocks:
            for i, ins in enumerate(block):
                if ins.name == Instruction.op.name:
                    result = None
                    for instructions, patterns in Optimizer.OP_CONSTANTS:
                        if ins.params[0] not in instructions:
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

                    if result is None and ins.params[0] in Optimizer.PRECALC:
                        try:
                            func = Optimizer.PRECALC[ins.params[0]]

                            a = float(ins.params[2])
                            if a.is_integer():
                                a = int(a)
                            if ins.params[3] == "_":
                                b = None
                            else:
                                b = float(ins.params[3])
                                if b.is_integer():
                                    b = int(b)

                            result = float(func(a, b))
                            if result.is_integer():
                                result = int(result)
                            result = str(result)

                        except (ArithmeticError, ValueError, TypeError):
                            pass

                    if result is not None:
                        block[i] = Instruction.set(ins.params[1], result)
                        found = True

                elif ins.name == Instruction.jump.name:
                    if ins.params[1] in Optimizer.JUMP_PRECALC:
                        try:
                            func = Optimizer.JUMP_PRECALC[ins.params[1]]

                            a = float(ins.params[2])
                            if a.is_integer():
                                a = int(a)
                            if ins.params[3] == "_":
                                b = None
                            else:
                                b = float(ins.params[3])
                                if b.is_integer():
                                    b = int(b)

                            if func(a, b):
                                block[i] = Instruction.jump_always(ins.params[0][1:])
                            else:
                                block[i] = Instruction.noop()
                            found = True

                        except (ArithmeticError, ValueError, TypeError):
                            pass

        return found

    @classmethod
    def _eliminate_common_subexpressions(cls, blocks: Blocks):
        found = False
        for block in blocks:
            operations: dict[tuple[str, str, str], str] = {}
            for i, ins in enumerate(block):
                if ins.name == Instruction.op.name:
                    if ins.params[0] == "rand":
                        continue

                    operands = (ins.params[0], ins.params[2], ins.params[3])
                    if operands in operations:
                        block[i] = Instruction.set(ins.params[1], operations[operands])
                        found = True
                    else:
                        operations[operands] = ins.params[1]

        return found

    @classmethod
    def _resolve_ssa(cls, blocks: Blocks):
        for block in blocks:
            for i, ins in enumerate(block):
                if ins.name == Phi.name:
                    assert isinstance(ins, Phi)
                    for b in ins.input_blocks:
                        b.add_phi.append(Instruction.set(ins.output, f"{ins.variable}:{b.variables[ins.variable]}"))
                    block[i] = Instruction.noop()

        for block in blocks:
            if len(block) > 0 and cls._is_jump(block[-1]):
                block[-1:] = block.add_phi + block[-1:]

            else:
                block += block.add_phi

    @classmethod
    def _remove_noops(cls, code: Instructions):
        code[:] = [ins for ins in code if ins.name != Instruction.noop.name]

    @classmethod
    def _is_impossible_jump(cls, ins: InstructionInstance) -> bool:
        if ins.params[1] == "equal":
            return ((ins.params[2] in ("true", "1") and ins.params[3] in ("false", "0"))
                    or (ins.params[3] in ("true", "1") and ins.params[2] in ("false", "0")))

        elif ins.params[1] == "notEqual":
            return ((ins.params[2] in ("true", "1") and ins.params[3] in ("true", "1"))
                    or (ins.params[3] in ("false", "0") and ins.params[2] in ("false", "0"))
                    or (ins.params[2] == ins.params[3]))

        return False

    @classmethod
    def _does_always_jump(cls, ins: InstructionInstance) -> bool:
        if ins.params[1] == "always":
            return True

        elif ins.params[1] in ("equal", "strictEqual"):
            return ((ins.params[2] in ("true", "1") and ins.params[3] in ("true", "1"))
                    or (ins.params[3] in ("false", "0") and ins.params[2] in ("false", "0"))
                    or (ins.params[2] == ins.params[3]))

        elif ins.params[1] == "notEqual":
            return ((ins.params[2] in ("true", "1") and ins.params[3] in ("false", "0"))
                    or (ins.params[3] in ("true", "1") and ins.params[2] in ("false", "0")))

        return False

    @classmethod
    def _optimize_jumps(cls, code: Instructions):
        used_labels = {label[1:] for ins in code for label in ins.params if label.startswith("$")}
        code[:] = [ins for ins in code if ins.name != Instruction.label.name or ins.params[0] in used_labels]

        labels = {"$" + ins.params[0]: i for i, ins in enumerate(code) if ins.name == Instruction.label.name}
        code[:] = [ins if ins.name != Instruction.jump.name or labels[ins.params[0]] != i else Instruction.noop()
                   for i, ins in enumerate(code)]

        code[:] = [ins if ins.name != Instruction.jump.name or not cls._is_impossible_jump(ins)
                   else Instruction.noop() for i, ins in enumerate(code)]
        code[:] = [ins if ins.name != Instruction.jump.name or not cls._does_always_jump(ins)
                   else Instruction.jump_always(ins.params[0][1:]) for i, ins in enumerate(code)]

    @classmethod
    def _optimize_set_op(cls, code: Instructions) -> bool:
        """
        Remove single use temporary variables that are immediately moved into another one.

        read __tmp0 cell1 0
        set x __tmp0

        read x cell1 0

        Remove unnecessary comparisons when jumping.

        op lessThan __tmp0 x y
        jump label equal __tmp0 0

        jump label greaterThanEq x y

        Propagate simple constants.

        set x 12
        print x
        set y 12

        print 12
        set y 12

        Remove unused variables.

        set x 12
        print 1

        print 1
        """

        uses = defaultdict(int)
        inputs = defaultdict(int)
        outputs = defaultdict(int)
        first_uses = {}

        for i, ins in enumerate(code):
            for j in ins.inputs:
                param = ins.params[j]
                inputs[param] += 1
                uses[param] += 1
                if uses[param] == 1:
                    first_uses[param] = (i, j)

            for j in ins.outputs:
                param = ins.params[j]
                outputs[param] += 1
                uses[param] += 1
                if uses[param] == 1:
                    first_uses[param] = (i, j)

        found = False
        for i, ins in enumerate(code):
            if ins.name == Instruction.set.name:
                if uses[ins.params[0]] == 1 or ins.params[0] == ins.params[1]:
                    code[i] = Instruction.noop()
                    return True

            if ins.name == Instruction.jump.name and ins.params[1] == "equal" and ins.params[3] == "0":
                tmp = ins.params[2]
                first = first_uses[tmp]
                if (inputs[tmp] == 1 and outputs[tmp] == 1 and i != first[0]
                        and code[first[0]].params[0] in Optimizer.JUMP_TRANSLATION):
                    ins.params[2:] = code[first[0]].params[2:]
                    ins.params[1] = Optimizer.JUMP_TRANSLATION[code[first[0]].params[0]]
                    code[first[0]] = Instruction.noop()
                    return True

            if (all(uses[ins.params[j]] == 1 for j in ins.outputs) and not ins.side_effects
                    and ins.name != Instruction.noop.name):
                code[i] = Instruction.noop()
                return True

            if (all(inputs[ins.params[j]] == 0 for j in ins.outputs) and not ins.side_effects
                    and ins.name != Instruction.noop.name):
                code[i] = Instruction.noop()
                return True

            for j in ins.inputs:
                param = ins.params[j]
                first = first_uses[param]
                first_ins = code[first[0]]
                if outputs[param] == 1 and first_ins.name == Instruction.set.name and first_ins.params[0] == param:
                    ins.params[j] = first_ins.params[1]
                    found = True

        return found

    @classmethod
    def _join_instructions_flush(cls, code: Instructions, prints: list[tuple[int, str]]) -> bool:
        if len(prints) > 1:
            buffer = "".join(p for _, p in prints)

            code[prints[0][0]].params[0] = f"\"{buffer}\""
            for i, _ in prints[1:]:
                code[i] = Instruction.noop()

            return True

        prints.clear()

    @classmethod
    def _join_instructions(cls, code: Instructions) -> bool:
        found = False

        prints: list[tuple[int, str]] = []
        for i, ins in enumerate(code):
            if ins.name == Instruction.print.name:
                val = None
                try:
                    _ = float(ins.params[0])
                    val = ins.params[0]
                except ValueError:
                    if ins.params[0].startswith("\"") and ins.params[0].endswith("\""):
                        val = ins.params[0][1:-1]

                if val is None:
                    found = found or cls._join_instructions_flush(code, prints)
                else:
                    prints.append((i, val))

            else:
                found = found or cls._join_instructions_flush(code, prints)

        found = found or cls._join_instructions_flush(code, prints)

        return found

    @classmethod
    def _precalculate_op_jump(cls, code: Instructions) -> bool:
        found = False
        for i, ins in enumerate(code):
            if ins.name == Instruction.op.name:
                result = None
                for instructions, patterns in Optimizer.OP_CONSTANTS:
                    if ins.params[0] not in instructions:
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

                if result is None and ins.params[0] in Optimizer.PRECALC:
                    try:
                        func = Optimizer.PRECALC[ins.params[0]]

                        a = float(ins.params[2])
                        if a.is_integer():
                            a = int(a)
                        if ins.params[3] == "_":
                            b = None
                        else:
                            b = float(ins.params[3])
                            if b.is_integer():
                                b = int(b)

                        result = float(func(a, b))
                        if result.is_integer():
                            result = int(result)
                        result = str(result)

                    except (ArithmeticError, ValueError, TypeError):
                        pass

                if result is not None:

                    code[i] = Instruction.set(ins.params[1], result)
                    found = True

            elif ins.name == Instruction.jump.name:
                if ins.params[1] in Optimizer.JUMP_PRECALC:
                    try:
                        func = Optimizer.JUMP_PRECALC[ins.params[1]]

                        a = float(ins.params[2])
                        if a.is_integer():
                            a = int(a)
                        if ins.params[3] == "_":
                            b = None
                        else:
                            b = float(ins.params[3])
                            if b.is_integer():
                                b = int(b)

                        if func(a, b):
                            code[i] = Instruction.jump_always(ins.params[0][1:])
                        else:
                            code[i] = Instruction.noop()
                        found = True

                    except (ArithmeticError, ValueError, TypeError):
                        pass

        return found
