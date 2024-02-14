from __future__ import annotations

import types
from typing import Callable, Any
from dataclasses import dataclass

from .value_types import *


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
        assert len(params) == len(self.params)
        return self.base_class(self, self.outputs, self.side_effects, self.constants, self.name, *params, **self.base_params)

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
    outputs: list[int]
    side_effects: bool
    constants: dict[int, str]

    def __init__(self, base: InstructionBase, outputs: list[int], side_effects: bool, constants: dict[int, str],
                 name: str, *params: Any, internal: bool = False, **_):
        self.name = name
        self.params = list(map(str, params))
        self.internal = internal
        self.base = base
        self.outputs = outputs
        self.side_effects = side_effects
        self.constants = constants

    def __str__(self):
        return f"{self.name} {" ".join(self.params)}"

    def translate_in_linker(self) -> InstructionInstance:
        return self


class LinkerInstructionInstance(InstructionInstance):
    translator: Callable[[LinkerInstruction], InstructionInstance]

    def __init__(self, *args, translator: Callable[[LinkerInstruction], InstructionInstance], **kwargs):
        super().__init__(*args, **kwargs)

        self.translator = translator

    def translate_in_linker(self) -> InstructionInstance:
        return self.translator(self)


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
            elif len(s) == 2:
                subcommands_processed[n] = (*s, default_side_effects, {})
            elif len(s) == 3:
                subcommands_processed[n] = (*s, {})
            else:
                subcommands_processed[n] = s

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
    print = _make("print", [Type.ANY], True)

    draw_flush = _make("drawflush", [Type.BLOCK], True)
    print_flush = _make("printflush", [Type.BLOCK], True)
    get_link = _make("getlink", [Type.BLOCK, Type.NUM], True, [0])
    control = _make_with_subcommands("control", True, [], [
        ("enabled", [Type.BLOCK, Type.NUM]),
        ("shoot", [Type.BLOCK, Type.NUM, Type.NUM, Type.NUM]),
        ("shootp", [Type.BLOCK, Type.UNIT, Type.NUM]),
        ("config", [Type.BLOCK, Type.CONTENT]),
        ("color", [Type.BLOCK, Type.NUM])
    ])
    # TODO: enum types
    radar = _make("radar", [Type.ANY, Type.ANY, Type.ANY, Type.ANY, Type.BLOCK, Type.NUM, Type.UNIT],
                  False, [6])
    # TODO: enum of sensable values
    sensor = _make("sensor", [Type.ANY, UnionType([Type.BLOCK, Type.UNIT]), Type.ANY], False, [0])
    # sensor = _make_with_subcommands("sensor", False, [0], [])

    set = _make("set", [Type.ANY, Type.ANY], True, [0], internal=True)
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
    jump = _make("jump", [Type.ANY] * 4, True, internal=True)

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
    # TODO: enum types
    uradar = _make("uradar", [Type.ANY, Type.ANY, Type.ANY, Type.ANY, Type.ANY, Type.NUM, Type.UNIT],
                   False, [6], constants={5: "0"})
    ulocate = _make_with_subcommands("ulocate", False, [], [

    ])

    label = _make("$label", [Type.ANY], True, internal=True)
    get_instruction_pointer_offset = _make("$get_instruction_pointer_offset",
                                           [Type.NUM, Type.NUM], False, [0], internal=True,
                                           base=LinkerInstructionInstance,
                                           translator=lambda ins: Instruction.op("add", ins.params[0], "@counter", ins.params[1]))
    jump_always = _make("$jump_always", [Type.ANY], True, internal=True, base=LinkerInstructionInstance,
                        translator=lambda ins: Instruction.jump("$" + ins.params[0], "always", "_", "_"))
    jump_addr = _make("$jump_addr", [Type.NUM], True, internal=True, base=LinkerInstructionInstance,
                      translator=lambda ins: Instruction.set("@counter", ins.params[0]))
