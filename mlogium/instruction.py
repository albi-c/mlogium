from __future__ import annotations

from typing import Any
from dataclasses import dataclass

from .value_types import *
from . import enums
from .abi import ABI
from .linking_context import LinkingContext


@dataclass(frozen=True)
class InstructionBase:
    name: str
    params: list[Type]
    side_effects: bool
    outputs: list[int]
    base_class: type[InstructionInstance]
    base_params: dict[str, Any]
    constants: dict[int, str]
    _subcommands: dict[str, tuple[list[Type], list[int], bool, dict[int, str]]] | None

    def __call__(self, *params: Any) -> InstructionInstance:
        params = params + tuple(["_"] * (len(self.params) - len(params)))
        return self.base_class(self, self.outputs, self.side_effects, self.constants, self.name,
                               *params, **self.base_params)

    def make_with_constants(self, *params: Any) -> InstructionInstance:
        assert len(params) == len(self.params) - len(self.constants)

        par = []
        i = 0
        for pi in range(len(self.params)):
            if pi in self.constants:
                par.append(self.constants[pi])
            else:
                par.append(params[i])
                i += 1
        return self(*par)

    def make_subcommand(self, name: str, *params: Any) -> InstructionInstance:
        assert self.has_subcommands()

        types_, outputs, side_effects, constants = self._subcommands[name]
        assert len(params) == len(types_)
        return self.base_class(self, outputs, side_effects, constants, self.name, name, *params, **self.base_params)

    def make_subcommand_with_constants(self, name: str, *params: Any) -> InstructionInstance:
        assert self.has_subcommands()

        types_, outputs, side_effects, constants = self._subcommands[name]
        assert len(params) == len(types_)

        par = []
        i = 0
        for pi in range(len(types_)):
            if pi in constants:
                par.append(constants[pi])
            else:
                par.append(params[i])
                i += 1

        return self.make_subcommand(name, *par)

    def has_subcommands(self) -> bool:
        return self._subcommands is not None

    def subcommands(self) -> dict[str, tuple[list[Type], list[int], bool, dict[int, str]]]:
        return self._subcommands


class DebugInstructionBase(InstructionBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class InstructionInstance:
    name: str
    params: list[str]
    internal: bool
    base: InstructionBase
    inputs: list[int]
    outputs: list[int]
    side_effects: bool
    constants: dict[int, str]

    def __init__(self, base: InstructionBase, outputs: list[int], side_effects: bool, constants: dict[int, str],
                 name: str, *params: Any, internal: bool = False,
                 param_process: Callable[[list[str]], list[str]] = None, **_):
        self.name = name
        self.params = list(map(str, params))
        self.internal = internal
        self.base = base
        self.inputs = [i for i in range(len(self.params)) if i not in outputs]
        self.outputs = outputs
        self.side_effects = side_effects
        self.constants = constants
        if param_process is not None:
            self.params = param_process(self.params)

    def __str__(self):
        return f"{self.name} {" ".join(self.params)}"

    def translate_in_linker(self, ctx: LinkingContext) -> list[InstructionInstance]:
        return [self]


class LinkerInstructionInstance(InstructionInstance):
    translator: Callable[[LinkerInstructionInstance], InstructionInstance]

    def __init__(self, *args, translator: Callable[[LinkerInstructionInstance], InstructionInstance], **kwargs):
        super().__init__(*args, **kwargs)

        self.translator = translator

    def translate_in_linker(self, ctx: LinkingContext) -> list[InstructionInstance]:
        return [self.translator(self)]


ALL_INSTRUCTIONS_BASES: list[InstructionBase] = []


class Instruction:
    def __init__(self, *_, **__):
        raise TypeError("Not instantiable")

    @staticmethod
    def _make(name: str, params: list[Type], side_effects: bool, outputs: list[int] = None,
              base: type[InstructionInstance] = None, constants: dict[int, str] = None, **base_params):
        outputs = outputs if outputs is not None else []
        base = base if base is not None else InstructionInstance
        constants = constants if constants is not None else {}
        ib = DebugInstructionBase(name, params, side_effects, outputs, base, base_params, constants, None)
        ALL_INSTRUCTIONS_BASES.append(ib)
        return ib

    @staticmethod
    def _make_with_subcommands(name: str, default_side_effects: bool, default_outputs: list[int],
                               subcommands: list[tuple[str, list[Type]
                                                            | tuple[list[Type], bool]
                                                            | tuple[list[Type], bool, list[int]]
                                                            | tuple[list[Type], bool, list[int], dict[int, str]]]],
                               base: type[InstructionInstance] = None, constants: dict[int, str] = None, **base_params):
        base = base if base is not None else InstructionInstance
        constants = constants if constants is not None else {}
        subcommands_processed: dict[str, tuple[list[Type], list[int], bool, dict[int, str]]] = {}
        for n, s in subcommands:
            if isinstance(s, list):
                subcommands_processed[n] = (s, default_outputs, default_side_effects, {})
            elif isinstance(s, tuple):
                if len(s) == 2:
                    subcommands_processed[n] = (s[0], s[1], default_side_effects, {})
                elif len(s) == 3:
                    subcommands_processed[n] = (s[0], s[2], s[1], {})
                elif len(s) == 4:
                    subcommands_processed[n] = (s[0], s[2], s[1], s[3])
                else:
                    raise ValueError(s)
            else:
                raise ValueError(s)

        ib = DebugInstructionBase(name, [], False, [], base, base_params, constants, subcommands_processed)
        ALL_INSTRUCTIONS_BASES.append(ib)
        return ib

    read = _make("read", [Type.NUM, Type.BLOCK, Type.NUM], False, [0])
    write = _make("write", [Type.NUM, Type.BLOCK, Type.NUM], True)
    draw = _make_with_subcommands("draw", True, [], [
        ("clear", [Type.NUM] * 3),
        ("color", [Type.NUM] * 4),
        ("col", [Type.NUM]),
        ("stroke", [Type.NUM]),
        ("line", [Type.NUM] * 4),
        ("rect", [Type.NUM] * 4),
        ("lineRect", [Type.NUM] * 4),
        ("poly", [Type.NUM] * 5),
        ("linePoly", [Type.NUM] * 5),
        ("triangle", [Type.NUM] * 6),
        ("image", [Type.NUM, Type.NUM, Type.CONTENT, Type.NUM, Type.BLOCK])
    ])
    print = _make("print", [Type.ANY], True, internal=True)

    draw_flush = _make("drawflush", [Type.BLOCK], True)
    print_flush = _make("printflush", [Type.BLOCK], True)
    get_link = _make("getlink", [Type.BLOCK, Type.NUM], False, [0])
    control = _make_with_subcommands("control", True, [], [
        ("enabled", [Type.BLOCK, Type.NUM]),
        ("shoot", [Type.BLOCK, Type.NUM, Type.NUM, Type.NUM]),
        ("shootp", [Type.BLOCK, Type.UNIT, Type.NUM]),
        ("config", [Type.BLOCK, Type.CONTENT]),
        ("color", [Type.BLOCK, Type.NUM])
    ])
    radar = _make("radar", [BasicType("$RadarFilter")] * 3 + [BasicType("$RadarSort"), Type.BLOCK, Type.NUM, Type.UNIT],
                  False, [6])
    sensor = _make_with_subcommands("sensor", False, [0], [
        (name, ([type_, UnionType([Type.BLOCK, Type.UNIT])], False, [0])) for name, type_ in enums.ENUM_SENSABLE.items()
    ], param_process=lambda params: [params[1], params[2], "@" + params[0]])

    set = _make("set", [Type.ANY, Type.ANY], False, [0], internal=True)
    essential_set = _make("$essential_set", [Type.ANY, Type.ANY], True, [],
                          internal=True, base=LinkerInstructionInstance,
                          translator=lambda ins: Instruction.set(*ins.params))
    op = _make("op", [Type.ANY, Type.NUM, Type.NUM, Type.NUM], False, [1], internal=True)
    lookup = _make_with_subcommands("lookup", False, [0], [
        ("block", [Type.BLOCK_TYPE, Type.NUM]),
        ("unit", [Type.UNIT_TYPE, Type.NUM]),
        ("item", [Type.ITEM_TYPE, Type.NUM]),
        ("liquid", [Type.LIQUID_TYPE, Type.NUM])
    ])
    pack_color = _make("packcolor", [Type.NUM] * 5, False, [0])

    wait = _make("wait", [Type.NUM], True)
    stop = _make("stop", [], True)
    end = _make("end", [], True)

    _jump_base = _make("jump", [Type.ANY] * 4, True, internal=True)

    class _JumpWrapper:
        name = "jump"

        def __call__(self, label: str, cond: str, a: str, b: str):
            return Instruction._jump_base("$" + label, cond, a, b)

    jump = _JumpWrapper()

    class TableRead(InstructionInstance):
        name = "$table_read"

        output_values: list[str]
        input_values: list[list[str]]
        index: str

        def __init__(self, output_values: list[str], input_values: list[list[str]], index: str):
            super().__init__(Instruction.noop, [i for i in range(len(output_values))],
                             False, {}, Instruction.TableRead.name,
                             *output_values, *(val for option in input_values for val in option),
                             internal=True)

            self.output_values = output_values
            self.input_values = input_values
            self.index = index

            length_of_one = len(self.output_values)
            assert all(len(lst) == length_of_one for lst in input_values)

        def __str__(self):
            return f"$table_read {self.output_values} = [{self.index}] : {self.input_values}"

        def translate_in_linker(self, ctx: LinkingContext) -> list[InstructionInstance]:
            if len(self.input_values) == 0 or len(self.output_values) == 0:
                return []
            instructions = []

            name = f"*__$table_read_{ctx.tmp_num()}"

            length_of_one = len(self.output_values)
            if length_of_one > 1:
                index = name + "_index"
                instructions.append(Instruction.op("mul", index, self.index, "2"))
            else:
                index = self.index
            instructions.append(Instruction.op("add", "@counter", "@counter", index))

            end_label = name + "_end"

            for i in range(1, len(self.input_values) + 1):
                for dst, src in zip(self.outputs, range(length_of_one)):
                    instructions.append(Instruction.set(self.params[dst], self.params[i * length_of_one + src]))
                instructions.append(Instruction.jump_always(end_label))

            if len(instructions) > 0 and instructions[-1].name == Instruction.jump.name:
                instructions.pop(-1)

            instructions.append(Instruction.label(end_label))

            return instructions

    class TableWrite(InstructionInstance):
        name = "$table_write"

        output_values: list[list[str]]
        input_values: list[str]
        index: str
        initial_values_index: int

        def __init__(self, output_values: list[list[str]], input_values: list[str], index: str):
            super().__init__(Instruction.noop, [i + len(input_values) for i in range(sum(map(len, output_values)))],
                             False, {}, Instruction.TableWrite.name,
                             *input_values, *(val for option in output_values for val in option),
                             *(val for option in output_values for val in option),
                             internal=True)

            self.output_values = output_values
            self.input_values = input_values
            self.index = index
            self.initial_values_index = max(self.outputs) + 1

            length_of_one = len(self.input_values)
            assert all(len(lst) == length_of_one for lst in output_values)

        def __str__(self):
            return f"$table_write [{self.index}] : {self.output_values} = {self.input_values}"

        def translate_in_linker(self, ctx: LinkingContext) -> list[InstructionInstance]:
            if len(self.input_values) == 0 or len(self.output_values) == 0:
                return []
            instructions = []

            name = f"*__$table_write_{ctx.tmp_num()}"

            length_of_one = len(self.input_values)

            for dst, src in zip(self.params[length_of_one:self.initial_values_index],
                                self.params[self.initial_values_index:]):
                instructions.append(Instruction.set(dst, src))

            if length_of_one > 1:
                index = name + "_index"
                instructions.append(Instruction.op("mul", index, self.index, "2"))
            else:
                index = self.index
            instructions.append(Instruction.op("add", "@counter", "@counter", index))

            end_label = name + "_end"

            for i in range(1, len(self.output_values) + 1):
                for dst, src in zip(range(length_of_one), self.inputs):
                    instructions.append(Instruction.set(self.params[i * length_of_one + dst], self.params[src]))
                instructions.append(Instruction.jump_always(end_label))

            if len(instructions) > 0 and instructions[-1].name == Instruction.jump.name:
                instructions.pop(-1)

            instructions.append(Instruction.label(end_label))

            return instructions

    ubind = _make("ubind", [Type.UNIT_TYPE], True)
    ucontrol = _make_with_subcommands("ucontrol", True, [], [
        ("idle", []),
        ("stop", []),
        ("move", [Type.NUM] * 2),
        ("approach", [Type.NUM] * 3),
        ("pathfind", [Type.NUM] * 2),
        ("autoPathfind", []),
        ("boost", [Type.NUM]),
        ("target", [Type.NUM] * 3),
        ("targetp", [Type.UNIT, Type.NUM]),
        ("itemDrop", [Type.BLOCK, Type.NUM]),
        ("itemTake", [Type.BLOCK, Type.ITEM_TYPE, Type.NUM]),
        ("payDrop", []),
        ("payTake", [Type.NUM]),
        ("payEnter", []),
        ("mine", [Type.NUM] * 2),
        ("flag", [Type.NUM]),
        ("build", [Type.NUM, Type.NUM, Type.BLOCK_TYPE, Type.NUM, UnionType([Type.CONTENT, Type.BLOCK])]),
        ("getBlock", ([Type.NUM, Type.NUM, Type.BLOCK_TYPE, Type.BLOCK, Type.NUM], [2, 3])),
        ("within", ([Type.NUM] * 4, [3])),
        ("unbind", []),
    ])
    uradar = _make("uradar", [BasicType("$RadarFilter")] * 3 + [BasicType("$RadarSort"), Type.ANY, Type.NUM, Type.UNIT],
                   False, [6], constants={5: "0"})
    ulocate = _make_with_subcommands("ulocate", False, [], [
        ("ore", ([Type.ANY, Type.ANY, Type.BLOCK_TYPE, Type.NUM, Type.NUM, Type.NUM, Type.NUM], False, [3, 4, 5],
                 {0: "_", 1: "_"})),
        ("building", (
        [BasicType("$LocateType"), Type.NUM, Type.ANY, Type.NUM, Type.NUM, Type.NUM, Type.BLOCK], False, [3, 4, 5, 6],
        {2: "_"})),
        ("spawn", ([Type.ANY, Type.ANY, Type.ANY, Type.NUM, Type.NUM, Type.NUM, Type.BLOCK], False, [3, 4, 5, 6],
                   {0: "_", 1: "_", 2: "_"})),
        ("damaged", ([Type.ANY, Type.ANY, Type.ANY, Type.NUM, Type.NUM, Type.NUM, Type.BLOCK], False, [3, 4, 5, 6],
                     {0: "_", 1: "_", 2: "_"}))
    ])

    label = _make("$label", [Type.ANY], True, internal=True)
    prepare_return_address = _make("$prepare_return_address", [], True, [],
                                   internal=True, base=LinkerInstructionInstance,
                                   translator=lambda ins: Instruction.op("add", ABI.function_return_address(),
                                                                         "@counter", "1"))

    @classmethod
    def jump_always(cls, label: str) -> InstructionInstance:
        return cls.jump(label, "always", "_", "_")

    jump_addr = _make("$jump_addr", [Type.NUM], True, internal=True, base=LinkerInstructionInstance,
                      translator=lambda ins: Instruction.set("@counter", ins.params[0]))

    noop = _make("noop", [], False, internal=True)

    getblock = _make_with_subcommands("getblock", False, [0], [
        ("floor", [Type.BLOCK, Type.NUM, Type.NUM]),
        ("ore", [Type.BLOCK, Type.NUM, Type.NUM]),
        ("block", [Type.BLOCK, Type.NUM, Type.NUM]),
        ("building", [Type.BLOCK, Type.NUM, Type.NUM])
    ])
    setblock = _make_with_subcommands("setblock", True, [], [
        ("floor", [Type.BLOCK_TYPE, Type.NUM, Type.NUM]),
        ("ore", [Type.BLOCK_TYPE, Type.NUM, Type.NUM]),
        ("block", [Type.BLOCK_TYPE, Type.NUM, Type.NUM, Type.TEAM, Type.NUM])
    ])

    spawn = _make("spawn", [Type.UNIT_TYPE, Type.NUM, Type.NUM, Type.NUM, Type.TEAM, Type.UNIT], True, [5])
    status = _make_with_subcommands("status", True, [], [
        ("apply", ([Type.ANY, BasicType("$Status"), Type.UNIT, Type.NUM], True, [], {0: "false"})),
        ("clear", ([Type.ANY, BasicType("$Status"), Type.UNIT], True, [], {0: "true"}))
    ])

    spawnwave = _make("spawnwave", [Type.NUM, Type.NUM, Type.NUM], True)

    setrule = _make_with_subcommands("setrule", True, [], [
        (rule, [Type.NUM] + ([Type.TEAM] if has_team else []))
        for rule, has_team in enums.ENUM_RULES.items()
    ] + [
                                         ("mapArea", [Type.ANY] + [Type.NUM] * 4)
                                     ])

    message = _make_with_subcommands("message", True, [], [
        ("notify", []),
        ("announce", [Type.NUM]),
        ("toast", [Type.NUM]),
        ("mission", [])
    ])
    cutscene = _make_with_subcommands("cutscene", True, [], [
        ("pan", [Type.NUM, Type.NUM, Type.NUM]),
        ("zoom", [Type.NUM]),
        ("stop", [])
    ])

    effect = _make_with_subcommands("effect", True, [], [
        (effect, params) for effect, params in enums.ENUM_EFFECT.items()
    ])
    explosion = _make("explosion", [Type.TEAM] + 7 * [Type.NUM], True)

    setrate = _make("setrate", [Type.NUM], True)

    fetch = _make_with_subcommands("fetch", False, [0], [
        ("unit", [Type.UNIT, Type.TEAM, Type.NUM]),
        ("player", [Type.UNIT, Type.TEAM, Type.NUM]),
        ("core", [Type.BLOCK, Type.TEAM, Type.NUM]),
        ("build", [Type.UNIT, Type.TEAM, Type.NUM, Type.BLOCK_TYPE]),
        ("unitCount", [Type.NUM, Type.TEAM]),
        ("playerCount", [Type.NUM, Type.TEAM]),
        ("coreCount", [Type.NUM, Type.TEAM]),
        ("buildCount", ([Type.NUM, Type.TEAM, Type.ANY, Type.BLOCK_TYPE], True, [0], {2: "_"}))
    ])

    sync = _make("sync", [Type.ANY], True, internal=True)

    getflag = _make("getflag", [Type.NUM, Type.STR], False, [0])
    setflag = _make("setflag", [Type.STR, Type.NUM], True)

    setprop = _make("setprop", [
        UnionType([Type.ITEM_TYPE, Type.LIQUID_TYPE, BasicType("Property")]),
        UnionType([Type.BLOCK, Type.UNIT]),
        Type.NUM
    ], True)
