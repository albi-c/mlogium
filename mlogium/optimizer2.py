from __future__ import annotations

import math
from re import search
from typing import Callable
from .instruction import InstructionInstance, Instruction
from .structure import CounterDict


class Block(list[InstructionInstance]):
    predecessors: set[Block]
    successors: set[Block]
    variables: dict[str, int]
    assignments: set[str]
    used_variables: set[str]
    add_phi: list[InstructionInstance]

    def __init__(self):
        super().__init__()

        self.predecessors = set()
        self.successors = set()
        self.variables = {}
        self.assignments = set()
        self.used_variables = set()
        self.add_phi = []

    def __eq__(self, other):
        return isinstance(other, Block) and id(self) == id(other)

    def __hash__(self):
        return id(self)


class Phi(InstructionInstance):
    name = "$phi"

    variable: str
    output: str
    input_blocks: set[Block]
    input_block_list: list[Block]

    def __init__(self, variable: str, output: str, input_blocks: set[Block]):
        self.input_block_list = list(input_blocks)
        super().__init__(Instruction.noop, [0], False, {}, Phi.name,
                         internal=True)

        self.variable = variable
        self.output = output
        self.input_blocks = input_blocks

    def __str__(self):
        return f"phi {self.output} = {self.variable} [{", ".join(map(str, self.input_blocks))}]"

    def translate_in_linker(self, _) -> list[InstructionInstance]:
        raise RuntimeError("Phi instruction must be converted")

    @classmethod
    def _find_variable_index(cls, block: Block, variable: str) -> int | None:
        if (i := block.variables.get(variable)) is not None:
            return i
        if block.predecessors != 1:
            return None
        return cls._find_variable_index(next(iter(block.predecessors)), variable)

    def finalize(self) -> bool:
        self.params = [self.output]
        for block in self.input_block_list:
            if (index := self._find_variable_index(block, self.variable)) is None:
                continue
            self.params.append(f"{self.variable}:{index}")

        self.inputs = list(range(1, len(self.params)))

        return len(self.params) < 3


type Instructions = list[InstructionInstance]
type Blocks = list[Block]

type ValueCopyGraph = list[ValueCopyGraph | str]


class Optimizer:
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

    OP_CONSTANTS: list[tuple[tuple[str, ...], tuple[tuple[tuple[str, ...], tuple[str, ...], str | None], ...]]] = [
        (("add", "or", "xor"), (
            (("0",), ("0",), None),
        )),
        (("sub",), (
            (tuple(), ("0",), None),
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
            (("0",), tuple(), "0")
        )),
        (("and", "land"), (
            (("0",), ("0",), "0"),
            (tuple(), tuple(), None)
        ))
    ]

    JUMP_TRANSLATION: dict[str, str] = {
        "equal": "notEqual",
        "notEqual": "equal",
        "greaterThan": "lessThanEq",
        "lessThan": "greaterThanEq",
        "greaterThanEq": "lessThan",
        "lessThanEq": "greaterThan"
    }

    @classmethod
    def optimize(cls, code: Instructions) -> Instructions:
        cls._optimize_jumps(code)
        cls._remove_noops(code)

        blocks = cls._make_blocks(code)
        cls._eval_block_jumps(blocks)
        cls._find_assignments(blocks)
        cls._optimize_block_jumps(blocks)
        cls._make_ssa(blocks)
        cls._remove_noops_blocks(blocks)

        while (cls._propagate_constants(blocks)
               or cls._remove_unused(blocks)
               or cls._optimize_op_jump(blocks)
               or cls._eliminate_common_subexpressions(blocks)
               or cls._optimize_jump_with_op(blocks)
               or cls._propagate_constants_rec(blocks)):
            cls._remove_noops_blocks(blocks)

        cls._optimize_unused_instructions(blocks)
        cls._remove_noops_blocks(blocks)

        cls._resolve_ssa(blocks)
        code = [ins for block in blocks for ins in block]

        cls._remove_noops(code)
        while cls._remove_unused_no_blocks(code):
            cls._remove_noops(code)

        return code

    @classmethod
    def _find_all_dependencies(cls, dependencies: set[str], value: str, graph: dict[str, set[str]]):
        if value in dependencies:
            return
        dependencies.add(value)

        for dep in graph.get(value, set()):
            cls._find_all_dependencies(dependencies, dep, graph)

    @classmethod
    def _optimize_unused_instructions(cls, blocks: Blocks):
        graph = {}
        for block in blocks:
            for ins in block:
                inputs = set(ins.params[i] for i in ins.inputs)
                for i in ins.outputs:
                    if ins.params[i] not in graph:
                        graph[ins.params[i]] = set()
                    graph[ins.params[i]] |= inputs

        dependencies = set()

        for block in blocks:
            for ins in block:
                if ins.side_effects:
                    for i in ins.inputs:
                        cls._find_all_dependencies(dependencies, ins.params[i], graph)

        for block in blocks:
            for i, ins in enumerate(block):
                if ins.name == Instruction.set.name or ins.name == Phi.name:
                    if ins.params[0] not in dependencies:
                        block[i] = Instruction.noop()

    @classmethod
    def _optimize_jump_with_op(cls, blocks: Blocks) -> bool:
        found = False
        for block in blocks:
            for i, ins in enumerate(block):
                if i < 1:
                    continue

                prev = block[i - 1]
                if ins.name == Instruction.jump.name and prev.name == Instruction.op.name and \
                        ins.params[1] == "equal" and ins.params[3] == "0" and ins.params[2] == prev.params[1] and \
                        (translated := cls.JUMP_TRANSLATION.get(prev.params[0])) is not None:
                    ins.params[1] = translated
                    ins.params[2], ins.params[3] = prev.params[2], prev.params[3]
                    found = True

        return found

    @classmethod
    def _eliminate_common_subexpressions(cls, blocks: Blocks) -> bool:
        operations: dict[tuple[str, str, str], str] = {}

        found = False
        for block in blocks:
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
    def _optimize_op_jump(cls, blocks: Blocks) -> bool:
        found = False
        for block in blocks:
            for i, ins in enumerate(block):
                if ins.name == Instruction.op.name:
                    result = None

                    if ins.params[0] in cls.PRECALC:
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

                    if result is None:
                        for instructions, patterns in cls.OP_CONSTANTS:
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
    def _remove_unused_no_blocks(cls, code: Instructions) -> bool:
        for i, ins in enumerate(code):
            if ins.name == Instruction.set.name and ins.params[0] == ins.params[1]:
                code[i] = Instruction.noop()

        input_used = set()

        for ins in code:
            for i in ins.inputs:
                input_used.add(ins.params[i])

        found = False
        for i, ins in enumerate(code):
            if ins.name != Instruction.noop.name and not ins.side_effects and \
                    not any(ins.params[i] in input_used for i in ins.outputs):
                code[i] = Instruction.noop()
                found = True

            elif ins.name == Instruction.set and ins.params[0] == ins.params[1]:
                code[i] = Instruction.noop()
                found = True

        return found

    @classmethod
    def _propagate_constants(cls, blocks: Blocks) -> bool:
        constants = {}

        for block in blocks:
            for ins in block:
                if ins.name == Instruction.set.name:
                    if ins.params[1] in constants:
                        ins.params[1] = constants[ins.params[1]]
                    constants[ins.params[0]] = ins.params[1]

        found = False
        for block in blocks:
            for ins in block:
                for i in ins.inputs:
                    inp = ins.params[i]
                    if (constant := constants.get(inp)) is not None:
                        ins.params[i] = constant
                        found = True

        return found

    @staticmethod
    def _list_unique_merge[T](a: list[T], b: list[T]) -> list[T]:
        return a + [x for x in b if x not in a]

    @classmethod
    def _build_value_copy_graph(cls, value: str, assignments: dict[str, InstructionInstance],
                                graphs: dict[str, ValueCopyGraph | None]) -> ValueCopyGraph | None:
        sentinel = object()
        if (graph := graphs.get(value, sentinel)) is not sentinel:
            return graph

        if (ins := assignments.get(value)) is not None:
            if ins.name == Instruction.set.name:
                graph = []
                graphs[value] = graph
                g = cls._build_value_copy_graph(ins.params[1], assignments, graphs)
                if g is None:
                    graphs[value] = None
                    return None
                graph = cls._list_unique_merge(graph, g)
                return graph

            elif ins.name == Phi.name:
                graph = []
                graphs[value] = graph
                for inp in ins.params[1:]:
                    g = cls._build_value_copy_graph(inp, assignments, graphs)
                    graphs[inp] = g
                    if g is None:
                        graphs[value] = None
                        return None
                    graph += g
                return graph

            else:
                return None

        if not value.endswith("\"") and ":" in value:
            return []
        return [value]

    @classmethod
    def _value_copy_graph_find_constants(cls, graph: ValueCopyGraph | None, seen: list[ValueCopyGraph]) -> set[str]:
        if graph is None:
            return set()
        if any(g is graph for g in seen):
            return set()
        seen.append(graph)
        result = set()
        for elem in graph:
            if isinstance(elem, str):
                result.add(elem)
            else:
                result |= cls._value_copy_graph_find_constants(elem, seen)
        return result

    @classmethod
    def _propagate_constants_rec(cls, blocks: Blocks) -> bool:
        assignments = {}
        for block in blocks:
            for ins in block:
                for i in ins.outputs:
                    assignments[ins.params[i]] = ins

        graphs = {}
        for block in blocks:
            for ins in block:
                for i in ins.inputs:
                    cls._build_value_copy_graph(ins.params[i], assignments, graphs)

        found = False
        for block in blocks:
            for ins in block:
                for i in ins.inputs:
                    if (graph := graphs.get(ins.params[i])) is not None:
                        options = cls._value_copy_graph_find_constants(graph, [])
                        if len(options) == 1:
                            ins.params[i] = next(iter(options))

        return found

    @classmethod
    def _remove_unused(cls, blocks: Blocks) -> bool:
        input_used = set()

        for block in blocks:
            for ins in block:
                for i in ins.inputs:
                    input_used.add(ins.params[i])

        found = False
        for block in blocks:
            for i, ins in enumerate(block):
                if ins.name != Instruction.noop.name and not ins.side_effects and \
                        not any(ins.params[i] in input_used for i in ins.outputs):
                    block[i] = Instruction.noop()
                    found = True

        return found

    @classmethod
    def _remove_noops_blocks(cls, blocks: Blocks):
        for block in blocks:
            cls._remove_noops(block)

    @classmethod
    def _is_label(cls, ins: InstructionInstance):
        return ins.name == Instruction.label.name

    @classmethod
    def _is_jump(cls, ins: InstructionInstance):
        return ins.name == Instruction.jump.name

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

        all_variables = set()
        for block in blocks:
            for ins in block:
                for i in ins.outputs:
                    all_variables.add(ins.params[i])

        variable_numbers = CounterDict(int)
        for block in blocks:
            cls._make_ssa_internal(block, variable_numbers, all_variables)
        for block in blocks:
            cls._finalize_ssa_internal(block)

    @classmethod
    def _make_ssa_internal(cls, block: Block, variable_numbers: CounterDict[str, int],
                           all_variables: set[str]):

        if len(block.predecessors) > 1:
            for name in all_variables:
                block.insert(0, Phi(name, f"{name}:{variable_numbers.inc(name)}",
                                    block.predecessors))

        for ins in block:
            if isinstance(ins, Phi):
                continue

            for i in ins.inputs:
                inp = ins.params[i]
                if inp in all_variables:
                    ins.params[i] = f"{inp}:{variable_numbers[inp]}"

            for i in ins.outputs:
                out = ins.params[i]
                ins.params[i] = f"{out}:{variable_numbers.inc(out)}"

        block.variables = variable_numbers.copy()

        for b in block.successors:
            b.variables = block.variables | b.variables

    @classmethod
    def _finalize_ssa_internal(cls, block: Block):
        for i, ins in enumerate(block):
            if isinstance(ins, Phi):
                if ins.finalize():
                    block[i] = Instruction.noop()

    @classmethod
    def _resolve_ssa(cls, blocks: Blocks):
        for block in blocks:
            for i, ins in enumerate(block):
                if isinstance(ins, Phi):
                    for b, v in zip(ins.input_blocks, ins.params[1:]):
                        b.add_phi.append(Instruction.set(ins.output, v))
                    block[i] = Instruction.noop()

        for block in blocks:
            if len(block) > 0 and cls._is_jump(block[-1]):
                block[:] = block[:-1] + block.add_phi + [block[-1]]

            else:
                block[:] = block + block.add_phi

            block.add_phi = []

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

        used = set()
        cls._eval_block_jumps_internal(blocks, labels, 0, used)

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
        all_assignments = set()
        for block in blocks:
            for ins in block:
                for i in ins.outputs:
                    block.assignments.add(ins.params[i])
                    all_assignments.add(ins.params[i])

        for block in blocks:
            for ins in block:
                for i in ins.inputs:
                    if ins.params[i] in all_assignments:
                        block.used_variables.add(ins.params[i])

    @staticmethod
    def _remove_noops(code: Instructions):
        code[:] = [ins for ins in code if ins.name != Instruction.noop.name]

    @staticmethod
    def _is_impossible_jump(ins: InstructionInstance) -> bool:
        if ins.params[1] == "equal":
            return ((ins.params[2] in ("true", "1") and ins.params[3] in ("false", "0"))
                    or (ins.params[3] in ("true", "1") and ins.params[2] in ("false", "0")))

        elif ins.params[1] == "notEqual":
            return ((ins.params[2] in ("true", "1") and ins.params[3] in ("true", "1"))
                    or (ins.params[3] in ("false", "0") and ins.params[2] in ("false", "0"))
                    or (ins.params[2] == ins.params[3]))

        return False

    @staticmethod
    def _does_always_jump(ins: InstructionInstance) -> bool:
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
