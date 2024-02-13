from .instruction import Instruction


class Linker:
    @staticmethod
    def _find_labels(instructions: list[Instruction]) -> tuple[list[Instruction], dict[str, int]]:
        labels = {}
        i = 0
        for ins in instructions:
            if ins.name == "$label":
                labels[ins.params[0]] = i
            else:
                i += 1

        return [ins.translate_in_linker() for ins in instructions if ins.name != "$label"], labels

    @classmethod
    def link(cls, instructions: list[Instruction]) -> str:
        instructions, labels = cls._find_labels(instructions)
        generated = []
        for ins in instructions:
            ln = ins.name
            for p in ins.params:
                if p.startswith("$"):
                    lab = labels[p[1:]]
                    ln += f" {lab}"
                else:
                    ln += f" {p}"
            generated.append(ln)
        return "\n".join(generated)
