import os.path
import unittest
from typing import Callable

from mlogium.error import PositionedException, NonPositionedException
from mlogium.compile import compile_code, compile_asm_code


class CompilationTestCase(unittest.TestCase):
    def _test_directory(self, path: str, extension: str, compile_function: Callable[[str, str], str]):
        directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
        for filename in (os.path.join(directory, fn) for fn in os.listdir(directory) if fn.endswith(extension)):
            with self.subTest(msg=filename):
                with open(filename, "r") as f:
                    code = f.read()

                compile_function(code, filename)

    @staticmethod
    def _reraise(res: PositionedException | NonPositionedException | str) -> str:
        if isinstance(res, PositionedException):
            raise res
        elif isinstance(res, NonPositionedException):
            raise res
        else:
            return res

    def test_examples_compilation(self):
        self._test_directory("examples", ".mu",
                             lambda code, filename: self._reraise(compile_code(code, filename, 2)[0]))

    def test_asm_examples_compilation(self):
        self._test_directory("examples", ".mua",
                             lambda code, filename: self._reraise(compile_asm_code(code, filename)))

    def test_stdlib_compilation(self):
        self._test_directory("mlogium/stdlib", ".mu",
                             lambda code, filename: self._reraise(compile_code(code, filename, 2)[0]))


if __name__ == '__main__':
    unittest.main()
