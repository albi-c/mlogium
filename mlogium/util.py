from dataclasses import dataclass


@dataclass
class Position:
    line: int
    start: int
    end: int
    code: str
    file: str

    def print(self):
        print(f"In file {self.file} on line {self.line + 1}, column {self.start + 1}")
        print("Here:")
        print(self.code)
        print(" " * self.start + "^" * (self.end - self.start + 1))

    def __add__(self, other):
        if not isinstance(other, Position):
            return NotImplemented

        if self.line < other.line:
            return Position(self.line, 0, self.end, self.code, self.file)
        elif self.line > other.line:
            return Position(other.line, 0, other.end, other.code, other.file)
        else:
            return Position(self.line, min(self.start, other.start), max(self.end, other.end), self.code, self.file)

    def __radd__(self, _):
        return NotImplemented

    def __iadd__(self, _):
        return NotImplemented
