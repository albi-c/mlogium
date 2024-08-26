from .instruction import InstructionInstance
from .linking_context import LinkingContext
from .error import LinkerError


class Linker:
    @staticmethod
    def _find_labels(instructions: list[InstructionInstance]) -> tuple[list[InstructionInstance], dict[str, int]]:
        labels = {}
        i = 0
        for ins in instructions:
            if ins.name == "$label":
                labels[ins.params[0]] = i
            else:
                i += 1

        return [ins for ins in instructions if ins.name != "$label"], labels

    @staticmethod
    def _translate_instructions(ctx: LinkingContext,
                                instructions: list[InstructionInstance]) -> list[InstructionInstance]:
        return [i for ins in instructions for i in ins.translate_in_linker(ctx)]

    @staticmethod
    def _resolve_simple_label(labels: dict[str, int], string: str) -> int:
        if (lab := labels.get(string)) is not None:
            return lab
        try:
            return int(string)
        except ValueError:
            LinkerError.custom(f"Label '{string}' not found")

    @classmethod
    def _resolve_label_with_offset(cls, labels: dict[str, int], string: str, pos: int) -> int:
        if "+" in string:
            a, b = string.split("+")
            if a:
                base = cls._resolve_simple_label(labels, a)
            else:
                base = pos
            offset = cls._resolve_simple_label(labels, b)
            return base + offset
        elif "-" in string:
            a, b = string.split("-")
            if a:
                base = cls._resolve_simple_label(labels, a)
            else:
                base = pos
            offset = cls._resolve_simple_label(labels, b)
            return base - offset
        else:
            return cls._resolve_simple_label(labels, string)

    @classmethod
    def link(cls, instructions: list[InstructionInstance]) -> str:
        ctx = LinkingContext()

        instructions = cls._translate_instructions(ctx, instructions)
        instructions, labels = cls._find_labels(instructions)
        if len(instructions) > 0:
            ins = instructions[-1]
            if ins.name == "jump":
                lab = cls._resolve_label_with_offset(labels, ins.params[0][1:], len(instructions) - 1)
                if lab == 0 or lab >= len(instructions):
                    instructions.pop(-1)
        generated = []
        for i, ins in enumerate(instructions):
            ln = ins.name
            for p in ins.params:
                if p.startswith("$"):
                    lab = cls._resolve_label_with_offset(labels, p[1:], i)
                    if lab >= len(instructions):
                        lab = 0
                    ln += f" {lab}"
                else:
                    ln += f" {p}"
            generated.append(ln)
        return "\n".join(generated)
