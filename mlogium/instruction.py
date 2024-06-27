from __future__ import annotations

from typing import Any, Callable
from dataclasses import dataclass

from .value_types import Types, BasicTypeRef, UnionTypeRef, TypeRef
from . import enums
from .abi import ABI
from .linking_context import LinkingContext


@dataclass(frozen=True)
class InstructionBase:
    name: str
    params: list[TypeRef]
    side_effects: bool
    outputs: list[int]
    base_class: type[InstructionInstance]
    base_params: dict[str, Any]
    constants: dict[int, str]
    _subcommands: dict[str, tuple[list[TypeRef], list[int], bool, dict[int, str]]] | None

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

    def subcommands(self) -> dict[str, tuple[list[TypeRef], list[int], bool, dict[int, str]]]:
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
    def _make(name: str, params: list[TypeRef], side_effects: bool, outputs: list[int] = None,
              base: type[InstructionInstance] = None, constants: dict[int, str] = None, **base_params):
        outputs = outputs if outputs is not None else []
        base = base if base is not None else InstructionInstance
        constants = constants if constants is not None else {}
        ib = DebugInstructionBase(name, params, side_effects, outputs, base, base_params, constants, None)
        ALL_INSTRUCTIONS_BASES.append(ib)
        return ib

    @staticmethod
    def _make_with_subcommands(name: str, default_side_effects: bool, default_outputs: list[int],
                               subcommands: list[tuple[str, list[TypeRef]
                                                            | tuple[list[TypeRef], bool]
                                                            | tuple[list[TypeRef], bool, list[int]]
                                                            | tuple[list[TypeRef], bool, list[int], dict[int, str]]]],
                               base: type[InstructionInstance] = None, constants: dict[int, str] = None, **base_params):
        base = base if base is not None else InstructionInstance
        constants = constants if constants is not None else {}
        subcommands_processed: dict[str, tuple[list[TypeRef], list[int], bool, dict[int, str]]] = {}
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

    read = _make("read", [Types.NUM, Types.BLOCK, Types.NUM], False, [0])
    write = _make("write", [Types.NUM, Types.BLOCK, Types.NUM], True)
    draw = _make_with_subcommands("draw", True, [], [
        ("clear", [Types.NUM] * 3),
        ("color", [Types.NUM] * 4),
        ("col", [Types.NUM]),
        ("stroke", [Types.NUM]),
        ("line", [Types.NUM] * 4),
        ("rect", [Types.NUM] * 4),
        ("lineRect", [Types.NUM] * 4),
        ("poly", [Types.NUM] * 5),
        ("linePoly", [Types.NUM] * 5),
        ("triangle", [Types.NUM] * 6),
        ("image", [Types.NUM, Types.NUM, Types.CONTENT, Types.NUM, Types.BLOCK])
    ])
    print = _make("print", [Types.ANY], True, internal=True)
    format = _make("format", [Types.ANY], True)

    draw_flush = _make("drawflush", [Types.BLOCK], True)
    print_flush = _make("printflush", [Types.BLOCK], True)
    get_link = _make("getlink", [Types.BLOCK, Types.NUM], False, [0])
    control = _make_with_subcommands("control", True, [], [
        ("enabled", [Types.BLOCK, Types.NUM]),
        ("shoot", [Types.BLOCK, Types.NUM, Types.NUM, Types.NUM]),
        ("shootp", [Types.BLOCK, Types.UNIT, Types.NUM]),
        ("config", [Types.BLOCK, Types.CONTENT]),
        ("color", [Types.BLOCK, Types.NUM])
    ])
    radar = _make("radar", [BasicTypeRef("$RadarFilter")] * 3 + [BasicTypeRef("$RadarSort"), Types.BLOCK, Types.NUM, Types.UNIT],
                  False, [6])
    sensor = _make_with_subcommands("sensor", False, [0], [
        (name, ([type_, UnionTypeRef([Types.BLOCK, Types.UNIT])], False, [0])) for name, type_ in enums.ENUM_SENSABLE.items()
    ], param_process=lambda params: [params[1], params[2], "@" + params[0]])

    set = _make("set", [Types.ANY, Types.ANY], False, [0], internal=True)
    op = _make("op", [Types.ANY, Types.NUM, Types.NUM, Types.NUM], False, [1], internal=True)
    lookup = _make_with_subcommands("lookup", False, [0], [
        ("block", [Types.BLOCK_TYPE, Types.NUM]),
        ("unit", [Types.UNIT_TYPE, Types.NUM]),
        ("item", [Types.ITEM_TYPE, Types.NUM]),
        ("liquid", [Types.LIQUID_TYPE, Types.NUM])
    ])
    pack_color = _make("packcolor", [Types.NUM] * 5, False, [0])

    wait = _make("wait", [Types.NUM], True)
    stop = _make("stop", [], True)
    end = _make("end", [], True)

    _jump_base = _make("jump", [Types.ANY] * 4, True, internal=True)

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

    ubind = _make("ubind", [Types.UNIT_TYPE], True)
    ucontrol = _make_with_subcommands("ucontrol", True, [], [
        ("idle", []),
        ("stop", []),
        ("move", [Types.NUM] * 2),
        ("approach", [Types.NUM] * 3),
        ("pathfind", [Types.NUM] * 2),
        ("autoPathfind", []),
        ("boost", [Types.NUM]),
        ("target", [Types.NUM] * 3),
        ("targetp", [Types.UNIT, Types.NUM]),
        ("itemDrop", [Types.BLOCK, Types.NUM]),
        ("itemTake", [Types.BLOCK, Types.ITEM_TYPE, Types.NUM]),
        ("payDrop", []),
        ("payTake", [Types.NUM]),
        ("payEnter", []),
        ("mine", [Types.NUM] * 2),
        ("flag", [Types.NUM]),
        ("build", [Types.NUM, Types.NUM, Types.BLOCK_TYPE, Types.NUM, UnionTypeRef([Types.CONTENT, Types.BLOCK])]),
        ("getBlock", ([Types.NUM, Types.NUM, Types.BLOCK_TYPE, Types.BLOCK, Types.NUM], [2, 3])),
        ("within", ([Types.NUM] * 4, [3])),
        ("unbind", []),
    ])
    uradar = _make("uradar", [BasicTypeRef("$RadarFilter")] * 3 + [BasicTypeRef("$RadarSort"), Types.ANY, Types.NUM, Types.UNIT],
                   False, [6], constants={5: "0"})
    ulocate = _make_with_subcommands("ulocate", False, [], [
        ("ore", ([Types.ANY, Types.ANY, Types.BLOCK_TYPE, Types.NUM, Types.NUM, Types.NUM, Types.NUM], False, [3, 4, 5],
                 {0: "_", 1: "_"})),
        ("building", (
            [BasicTypeRef("$LocateType"), Types.NUM, Types.ANY, Types.NUM, Types.NUM, Types.NUM, Types.BLOCK], False, [3, 4, 5, 6],
            {2: "_"})),
        ("spawn", ([Types.ANY, Types.ANY, Types.ANY, Types.NUM, Types.NUM, Types.NUM, Types.BLOCK], False, [3, 4, 5, 6],
                   {0: "_", 1: "_", 2: "_"})),
        ("damaged", ([Types.ANY, Types.ANY, Types.ANY, Types.NUM, Types.NUM, Types.NUM, Types.BLOCK], False, [3, 4, 5, 6],
                     {0: "_", 1: "_", 2: "_"}))
    ])

    label = _make("$label", [Types.ANY], True, internal=True)

    @classmethod
    def jump_always(cls, label: str) -> InstructionInstance:
        return cls.jump(label, "always", "_", "_")

    noop = _make("noop", [], False, internal=True)

    getblock = _make_with_subcommands("getblock", False, [0], [
        ("floor", [Types.BLOCK, Types.NUM, Types.NUM]),
        ("ore", [Types.BLOCK, Types.NUM, Types.NUM]),
        ("block", [Types.BLOCK, Types.NUM, Types.NUM]),
        ("building", [Types.BLOCK, Types.NUM, Types.NUM])
    ])
    setblock = _make_with_subcommands("setblock", True, [], [
        ("floor", [Types.BLOCK_TYPE, Types.NUM, Types.NUM]),
        ("ore", [Types.BLOCK_TYPE, Types.NUM, Types.NUM]),
        ("block", [Types.BLOCK_TYPE, Types.NUM, Types.NUM, Types.TEAM, Types.NUM])
    ])

    spawn = _make("spawn", [Types.UNIT_TYPE, Types.NUM, Types.NUM, Types.NUM, Types.TEAM, Types.UNIT], True, [5])
    status = _make_with_subcommands("status", True, [], [
        ("apply", ([Types.ANY, BasicTypeRef("$Status"), Types.UNIT, Types.NUM], True, [], {0: "false"})),
        ("clear", ([Types.ANY, BasicTypeRef("$Status"), Types.UNIT], True, [], {0: "true"}))
    ])

    weather_sense = _make("weathersense", [Types.NUM, BasicTypeRef("Weather")], False, [0])
    weather_set = _make("weatherset", [BasicTypeRef("Weather"), Types.NUM], True)

    spawnwave = _make("spawnwave", [Types.NUM, Types.NUM, Types.NUM], True)

    setrule = _make_with_subcommands("setrule", True, [], [
        (rule, [Types.NUM] + ([Types.TEAM] if has_team else []))
        for rule, has_team in enums.ENUM_RULES.items()
    ] + [
                                         ("mapArea", [Types.ANY] + [Types.NUM] * 4)
                                     ])

    message = _make_with_subcommands("message", True, [], [
        ("notify", []),
        ("announce", [Types.NUM]),
        ("toast", [Types.NUM]),
        ("mission", [])
    ])
    cutscene = _make_with_subcommands("cutscene", True, [], [
        ("pan", [Types.NUM, Types.NUM, Types.NUM]),
        ("zoom", [Types.NUM]),
        ("stop", [])
    ])

    effect = _make_with_subcommands("effect", True, [], [
        (effect, params) for effect, params in enums.ENUM_EFFECT.items()
    ])
    explosion = _make("explosion", [Types.TEAM] + 7 * [Types.NUM], True)

    setrate = _make("setrate", [Types.NUM], True)

    fetch = _make_with_subcommands("fetch", False, [0], [
        ("unit", [Types.UNIT, Types.TEAM, Types.NUM]),
        ("player", [Types.UNIT, Types.TEAM, Types.NUM]),
        ("core", [Types.BLOCK, Types.TEAM, Types.NUM]),
        ("build", [Types.UNIT, Types.TEAM, Types.NUM, Types.BLOCK_TYPE]),
        ("unitCount", [Types.NUM, Types.TEAM]),
        ("playerCount", [Types.NUM, Types.TEAM]),
        ("coreCount", [Types.NUM, Types.TEAM]),
        ("buildCount", ([Types.NUM, Types.TEAM, Types.ANY, Types.BLOCK_TYPE], True, [0], {2: "_"}))
    ])

    sync = _make("sync", [Types.ANY], True, internal=True)

    getflag = _make("getflag", [Types.NUM, Types.STR], False, [0])
    setflag = _make("setflag", [Types.STR, Types.NUM], True)

    setprop = _make("setprop", [
        UnionTypeRef([Types.ITEM_TYPE, Types.LIQUID_TYPE, BasicTypeRef("Property")]),
        UnionTypeRef([Types.BLOCK, Types.UNIT]),
        Types.NUM
    ], True)

    make_marker = _make("makemarker", [BasicTypeRef("$MarkerType"), Types.NUM, Types.NUM, Types.NUM, Types.NUM],
                        True, [])
    set_marker = _make_with_subcommands("setmarker", True, [], [
        ("remove", [Types.NUM]),
        ("world", [Types.NUM, Types.NUM]),
        ("minimap", [Types.NUM, Types.NUM]),
        ("autoscale", [Types.NUM, Types.NUM]),
        ("pos", [Types.NUM, Types.NUM, Types.NUM]),
        ("endPos", [Types.NUM, Types.NUM, Types.NUM]),
        ("drawLayer", [Types.NUM, Types.NUM]),
        ("color", [Types.NUM, Types.NUM]),
        ("radius", [Types.NUM, Types.NUM]),
        ("stroke", [Types.NUM, Types.NUM]),
        ("rotation", [Types.NUM, Types.NUM]),
        ("shape", [Types.NUM, Types.NUM, Types.NUM, Types.NUM]),
        ("arc", [Types.NUM, Types.NUM, Types.NUM]),
        ("flushText", [Types.NUM, Types.NUM]),
        ("fontSize", [Types.NUM, Types.NUM]),
        ("textHeight", [Types.NUM, Types.NUM]),
        ("labelFlags", [Types.NUM, Types.NUM, Types.NUM]),
        ("texture", [Types.NUM, Types.NUM, Types.STR]),
        ("textureSize", [Types.NUM, Types.NUM, Types.NUM]),
        ("posi", [Types.NUM, Types.NUM, Types.NUM, Types.NUM]),
        ("uvi", [Types.NUM, Types.NUM, Types.NUM, Types.NUM]),
        ("colori", [Types.NUM, Types.NUM, Types.NUM])
    ])

    locale_print = _make("localeprint", [Types.STR], True, [])
