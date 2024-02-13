from __future__ import annotations

from typing import Callable


class Instruction:
    name: str
    params: list[str]

    def __init__(self, name: str, *params: str):
        self.name = name
        self.params = list(map(str, params))

    def __str__(self):
        return f"{self.name} {" ".join(self.params)}"

    def translate_in_linker(self) -> Instruction:
        return self

    @staticmethod
    def set(a: str, b: str) -> Instruction:
        return Instruction("set", a, b)

    @staticmethod
    def print(value: str) -> Instruction:
        return Instruction("print", value)

    @staticmethod
    def jump(target: str, cond: str, a: str, b: str) -> Instruction:
        return Instruction("jump", target, cond, a, b)

    @staticmethod
    def label(name: str) -> Instruction:
        return Instruction("$label", name)

    @staticmethod
    def get_instruction_pointer_offset(result: str, offset: str):
        return Instruction("op", "add", result, "@counter", offset)

    @staticmethod
    def jump_always(label: str) -> Instruction:
        return LinkerInstruction("$jump_always", label,
                                 translator=lambda ins: Instruction.jump("$" + label, "always", "_", "_"))

    @staticmethod
    def jump_addr(addr: str) -> Instruction:
        return LinkerInstruction("$jump_addr", addr,
                                 translator=lambda ins: Instruction.set("@counter", addr))


class LinkerInstruction(Instruction):
    translator: Callable[[LinkerInstruction], Instruction]

    def __init__(self, name: str, *params: str, translator: Callable[[LinkerInstruction], Instruction]):
        super().__init__(name, *params)

        self.translator = translator

    def translate_in_linker(self) -> Instruction:
        return self.translator(self)
