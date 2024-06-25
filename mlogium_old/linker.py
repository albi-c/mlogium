from .instruction import InstructionInstance
from .linking_context import LinkingContext


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

    @classmethod
    def link(cls, instructions: list[InstructionInstance]) -> str:
        ctx = LinkingContext()

        instructions = cls._translate_instructions(ctx, instructions)
        instructions, labels = cls._find_labels(instructions)
        generated = []
        for ins in instructions:
            ln = ins.name
            for p in ins.params:
                if p.startswith("$"):
                    lab = labels[p[1:]]
                    if lab == len(instructions):
                        lab = 0
                    ln += f" {lab}"
                else:
                    ln += f" {p}"
            generated.append(ln)
        return "\n".join(generated)
