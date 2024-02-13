from __future__ import annotations


class Instruction:
    name: str
    params: list[str]

    def __init__(self, name: str, *params: str):
        self.name = name
        self.params = list(map(str, params))

    def __str__(self):
        return f"{self.name} {" ".join(self.params)}"

    @classmethod
    def set(cls, a: str, b: str) -> Instruction:
        return Instruction("set", a, b)

    @classmethod
    def print(cls, value: str) -> Instruction:
        return Instruction("print", value)

    @classmethod
    def label(cls, name: str) -> Instruction:
        return Instruction("$label", name)

    @classmethod
    def get_instruction_pointer_offset(cls, result: str, offset: str):
        return Instruction("op", "add", result, "@counter", offset)

    @classmethod
    def jump_always(cls, label: str) -> Instruction:
        return Instruction("$jump_always", label)

    @classmethod
    def jump_addr(cls, addr: str) -> Instruction:
        return Instruction("$jump_addr", addr)
