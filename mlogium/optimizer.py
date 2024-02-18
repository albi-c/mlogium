from collections import defaultdict
from typing import Callable
import math

from .instruction import Instruction, InstructionInstance


class Optimizer:
    type Instructions = list[InstructionInstance]

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
        (("and",), (
            (("0",), ("0",), "0"),
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

    @classmethod
    def optimize(cls, code: Instructions) -> Instructions:
        cls._optimize_jumps(code)
        cls._remove_noops(code)

        while cls._optimize_set_op(code) or cls._precalculate_op(code):
            pass

        cls._remove_noops(code)

        return code

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
                if uses[ins.params[0]] == 1:
                    code[i] = Instruction.noop()
                    return True

                tmp = ins.params[1]
                first = first_uses[tmp]
                if inputs[tmp] == 1 and outputs[tmp] == 1 and i != first[0]:
                    code[first[0]].params[first[1]] = ins.params[0]
                    code[i] = Instruction.noop()
                    return True

            elif ins.name == Instruction.jump.name and ins.params[1] == "equal" and ins.params[3] == "0":
                tmp = ins.params[2]
                first = first_uses[tmp]
                if (inputs[tmp] == 1 and outputs[tmp] == 1 and i != first[0]
                        and code[first[0]].params[0] in Optimizer.JUMP_TRANSLATION):
                    ins.params[2:] = code[first[0]].params[2:]
                    ins.params[1] = Optimizer.JUMP_TRANSLATION[code[first[0]].params[0]]
                    code[first[0]] = Instruction.noop()
                    return True

            elif (all(uses[ins.params[j]] == 1 for j in ins.outputs) and not ins.side_effects
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
    def _precalculate_op(cls, code: Instructions) -> bool:
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

        return found
