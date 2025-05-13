from __future__ import annotations

from collections import defaultdict
from typing import Callable
import math

from .instruction import Instruction, InstructionInstance
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
    # table for merging `op` with `jump` instructions
    JUMP_TRANSLATION: dict[str, str] = {
        "equal": "notEqual",
        "notEqual": "equal",
        "greaterThan": "lessThanEq",
        "lessThan": "greaterThanEq",
        "greaterThanEq": "lessThan",
        "lessThanEq": "greaterThan"
    }

    """
    mathematical transformations
    
    list of tuples with elements:
        1. tuple of possible `op` operations
        2. tuple of transformations
            - values:
                - a: tuple
                - b: tuple
                - result: str|None
            - two inputs (a, b), first match taken:
                - empty tuple - match any value
                - tuple with elements - match concrete value
            - output:
                - string - constant value
                - None - value of the input which was not  
    """
    OP_CONSTANTS: list[tuple[tuple[str, ...], tuple[tuple[tuple[str, ...], tuple[str, ...], str | None], ...]]] = [
        (("add", "or", "xor"), (
            # inputs can be swapped
            # x + 0 = x
            (("0",), ("0",), None),
        )),
        (("sub",), (
            # x - 0 = x
            # x - x = 0
            ((), ("0",), None),
            ((), (), "0"),
        )),
        (("mul",), (
            # inputs can be swapped
            # x * 0 = 0
            # x * 1 = x
            (("0",), ("0",), "0"),
            (("1",), ("1",), None),
        )),
        (("div", "idiv"), (
            # x / x = 1
            # x / 1 = x
            # 0 / x = 0
            ((), (), "1"),
            ((), ("1",), None),
            (("0",), (), "0"),
        )),
        (("shr", "shl"), (
            # x >> 0 = x
            # 0 >> x = 0
            ((), ("0",), None),
            (("0",), (), "0"),
        )),
        (("and", "land"), (
            # inputs can be swapped
            # x & 0 = 0
            # x & x = x
            (("0",), ("0",), "0"),
            ((), (), None),
        )),
        (("max", "min"), (
            # max(x, x) = x
            ((), (), None),
        )),
        (("equal", "strictEqual", "lessThanEq", "greaterThanEq"), (
            # x == x = 1
            ((), (), "1"),
        )),
        (("notEqual", "lessThan", "greaterThan"), (
            # x > x = 0
            ((), (), "0"),
        )),
        (("pow",), (
            # x ^ 0 = 1
            ((), "0", "1"),
            # x ^ 1 = x
            ((), "1", None),
            # 0 ^ x = 0
            ("0", (), "0"),
            # 1 ^ x = 1
            ("1", (), "1")
        )),
        (("mod",), (
            # x % 1 = 0
            ((), "1", "0"),
            # x % x = 0
            ((), (), "0"),
            # 0 % x = 0
            ("0", (), "0"),
        ))
    ]

    # functions for compile time calculation
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
        "atan": lambda a, _: math.degrees(math.atan(a)),
        "sign": lambda a, _: 1 if a > 0 else -1 if a < 0 else 0
    }

    # functions for compile time calculation of jump conditions
    JUMP_PRECALC: dict[str, Callable[[int | float, int | float], bool]] = {
        "equal": lambda a, b: a == b,
        "notEqual": lambda a, b: a != b,
        "greaterThan": lambda a, b: a > b,
        "lessThan": lambda a, b: a < b,
        "greaterThanEq": lambda a, b: a >= b,
        "lessThanEq": lambda a, b: a <= b
    }

    @classmethod
    def optimize(cls, code: Instructions, level: int) -> Instructions:
        """
        Optimizer entry point. Calls other optimization functions.
        """

        assert level >= 0

        for i in range(level):
            cls._optimize_jumps(code)
            cls._remove_noops(code)

            while cls._precalculate_op_jump(code):
                pass
            cls._remove_noops(code)

            blocks = cls._make_blocks(code)
            cls._eval_block_jumps(blocks)
            if i == 0:
                cls._find_assignments(blocks)
            cls._optimize_block_jumps(blocks)
            if i == 0:
                cls._make_ssa(blocks)
                while (cls._propagate_constants(blocks) or cls._precalculate_op_jump_blocks(blocks)
                       or cls._eliminate_common_subexpressions(blocks)):
                    pass
                cls._optimize_unused_instructions(blocks)
                cls._resolve_ssa(blocks)
            code = [ins for block in blocks for ins in block]

            cls._remove_noops(code)
            cls._optimize_jumps(code)
            cls._remove_noops(code)

            while cls._optimize_set_op(code) or cls._precalculate_op_jump(code) or cls._optimize_jump_tables(code):
                pass
            cls._remove_noops(code)
            while cls._merge_op_set(code):
                pass

            while cls._join_instructions(code):
                cls._remove_noops(code)

        return code

    @classmethod
    def _is_label(cls, ins: InstructionInstance):
        return ins.name == Instruction.label.name

    @classmethod
    def _is_jump(cls, ins: InstructionInstance):
        return ins.name == Instruction.jump.name

    @classmethod
    def _make_blocks(cls, code: Instructions) -> Blocks:
        """
        Convert flat list of instructions to blocks
        """

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
        """
        Find predecessors and successors of blocks
        """

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

    @classmethod
    def _optimize_block_jumps(cls, blocks: Blocks):
        """
        Remove unnecessary jumps
        """

        labels = {"$" + lab.params[0]: i for i, block in enumerate(blocks)
                  for lab in block if lab.name == Instruction.label.name}
        for i, block in enumerate(blocks):
            if len(block) > 0 and block[-1].name == Instruction.jump.name and labels[block[-1].params[0]] == i + 1:
                block.pop(-1)

    @classmethod
    def _make_ssa(cls, blocks: Blocks):
        """
        Convert code to SSA form
        """

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
        """
        Convert Phi instructions to assignments
        """

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
    def _propagate_constants(cls, blocks: Blocks) -> bool:
        """
        Propagate constants in assignments and Phi instructions
        """

        constants: dict[str, str] = {}
        found = False

        for block in blocks:
            for i, ins in enumerate(block):
                if ins.name == Instruction.set.name:
                    constants[ins.params[0]] = ins.params[1]

                elif ins.name == Phi.name:
                    assert len(ins.inputs) >= 2
                    if all(inp == ins.params[1] for inp in ins.params[2:]):
                        constants[ins.params[0]] = ins.params[1]
                        block[i] = Instruction.set(ins.params[0], ins.params[1])
                        found = True

        for block in blocks:
            for ins in block:
                for i in ins.inputs:
                    if ins.params[i] in constants:
                        ins.params[i] = constants[ins.params[i]]
                        found = True

        return found

    @staticmethod
    def _list_unique_merge[T](a: list[T], b: list[T]) -> list[T]:
        """
        Perform set intersection on lists of possibly unhashable values
        """

        return a + [x for x in b if x not in a]

    @classmethod
    def _build_value_copy_graph(cls, value: str, assignments: dict[str, InstructionInstance],
                                graphs: dict[str, ValueCopyGraph | None]) -> ValueCopyGraph | None:
        """
        Build graph of possible variable values
        """

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
        """
        Find constants in a value copy graph
        """

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
        """
        Currently unused. Propagates constants using a tree.
        """

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
            for ins_i, ins in enumerate(block):
                for i in ins.inputs:
                    if (graph := graphs.get(ins.params[i])) is not None:
                        options = cls._value_copy_graph_find_constants(graph, [])
                        if len(options) == 1:
                            print(ins.name, ins.params, options)
                            prev = ins.params[i]
                            new = next(iter(options))
                            ins.params[i] = new
                            if prev != new:
                                found = True

        return found

    @classmethod
    def _find_all_dependencies(cls, dependencies: set[str], value: str, graph: dict[str, set[str]]):
        """
        Find dependencies of a value
        """

        if value in dependencies:
            return
        dependencies.add(value)

        for dep in graph.get(value, set()):
            cls._find_all_dependencies(dependencies, dep, graph)

    @classmethod
    def _optimize_unused_instructions(cls, blocks: Blocks):
        """
        Remove instructions without side effects whose outputs are not used
        """

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
    def _precalculate_op_jump_blocks(cls, blocks: Blocks) -> bool:
        """
        Attempt to perform calculations at compile time
        """

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

                elif ins.name == Instruction.pack_color.name:
                    params = []
                    for param in ins.params[1:]:
                        try:
                            p = int(float(param) * 255)
                            if p < 0 or p > 255:
                                break
                            else:
                                params.append(p)
                        except ValueError:
                            break
                    if len(params) == 4:
                        r, g, b, a = params
                        block[i] = Instruction.set(ins.params[0], f"%{r:02x}{g:02x}{b:02x}{a:02x}")
                        found = True

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
    def _remove_noops(cls, code: Instructions):
        code[:] = [ins for ins in code if ins.name != Instruction.noop.name]

    @classmethod
    def _is_impossible_jump(cls, ins: InstructionInstance) -> bool:
        """
        Check if a jump never happens
        """

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
        """
        Check if a jump always happens
        """

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
        """
        Remove unnecessary jumps
        """

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

    @staticmethod
    def _merge_op_set(code: Instructions) -> bool:
        """
        Move assignments up

        set __tmp1 12
        print 16
        set x __tmp1

        set x 12
        print 16
        """

        inputs = defaultdict(int)
        outputs = defaultdict(int)
        first_writes = {}

        for i, ins in enumerate(code):
            for j in ins.inputs:
                inputs[ins.params[j]] += 1

            for j in ins.outputs:
                out = ins.params[j]
                outputs[out] += 1
                first_writes[out] = i

        for i, ins in enumerate(code):
            if i >= 1 and ins.name == Instruction.set.name:
                for j in range(i):
                    prev = code[j]
                    if len(prev.outputs) == 1:
                        out = prev.params[prev.outputs[0]]
                        if ins.params[1] == out and inputs[out] == 1:
                            # check if the value being assigned to in `ins` is used between the two affected instructions
                            if not any(ins.params[0] in code[k].params for k in range(j + 1, i)):
                                prev.params[prev.outputs[0]] = ins.params[0]
                                code.pop(i)
                                return True

        return False

    @classmethod
    def _join_instructions(cls, code: Instructions) -> bool:
        """
        Join print instructions into one
        """

        found = False

        prints: list[tuple[int, str]] = []
        for i, ins in enumerate(code):
            if ins.name == Instruction.print.name:
                if ins.params[0] == "\"\"" or len(ins.params[0]) == 0:
                    code[i] = Instruction.noop()
                    found = True
                    continue

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

        return found or cls._join_instructions_flush(code, prints)

    @classmethod
    def _join_instructions_flush(cls, code: Instructions, prints: list[tuple[int, str]]) -> bool:
        if len(prints) > 1:
            buffer = "".join(p for _, p in prints)

            if len(buffer) > 0:
                code[prints[0][0]].params[0] = f"\"{buffer}\""
            else:
                code[prints[0][0]] = Instruction.noop()

            for i, _ in prints[1:]:
                code[i] = Instruction.noop()

            return True

        prints.clear()

        return False

    @classmethod
    def _precalculate_op_jump(cls, code: Instructions) -> bool:
        """
        Attempt to perform calculations at compile time
        """

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

            elif ins.name == Instruction.pack_color.name:
                params = []
                for param in ins.params[1:]:
                    try:
                        p = int(float(param) * 255)
                        if p < 0 or p > 255:
                            break
                        else:
                            params.append(p)
                    except ValueError:
                        break
                if len(params) == 4:
                    r, g, b, a = params
                    code[i] = Instruction.set(ins.params[0], f"%{r:02x}{g:02x}{b:02x}{a:02x}")
                    found = True

            elif ins.name == Instruction.lookup.name:
                try:
                    idx = int(ins.params[2])
                    if idx < 0:
                        continue
                    content = ins.params[0]
                    if content == "team":
                        if idx < 6:
                            code[i] = Instruction.set(ins.params[1], ["@derelict", "@sharded", "@crux",
                                                                      "@malis", "@green", "@blue"][idx])
                            found = True
                except ValueError:
                    pass

            elif ins.name == Instruction.read.name:
                if ins.params[1].startswith("\"") and ins.params[1].endswith("\""):
                    try:
                        idx = int(ins.params[2])
                    except ValueError:
                        pass
                    else:
                        string = ins.params[1][1:-1]
                        if 0 <= idx < len(string):
                            code[i] = Instruction.set(ins.params[0], ord(string[idx]))
                            found = True

        return found

    @classmethod
    def _optimize_jump_tables(cls, code: Instructions) -> bool:
        """
        Optimize jump tables with compile time known indices
        """

        found = True
        found_any = False
        while found:
            found = False

            for i, ins in enumerate(code):
                if isinstance(ins, Instruction.TableRead):
                    try:
                        index = int(ins.params[-1])
                    except ValueError:
                        pass
                    else:
                        inputs = ins.get_input_values()
                        if index < len(inputs):
                            outputs = ins.get_output_values()
                            code[i:i+1] = [
                                Instruction.set(a, b)
                                for a, b in zip(outputs, inputs[index])
                            ]

                            found = True
                            break

                elif isinstance(ins, Instruction.TableWrite):
                    try:
                        index = int(ins.params[-1])
                    except ValueError:
                        pass
                    else:
                        if index < ins.num_output_values:
                            generated = []
                            for j in range(ins.num_output_values):
                                generated += [
                                    Instruction.set(a, b)
                                    for a, b in zip(ins.get_output_values()[j],
                                                    ins.get_input_values()
                                                    if j == index else ins.get_initial_values()[j])
                                ]
                            code[i:i + 1] = generated

                            found = True
                            break

            found_any = found_any or found

        return found_any
