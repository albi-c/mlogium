import os.path
import unittest
from typing import Callable, Any

from mlogium.compile import compile_code, compile_asm_code


class CompilationTestCase(unittest.TestCase):
    def _test_directory(self, path: str, extension: str, compile_function: Callable[[str, str], Any]):
        directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
        for filename in (os.path.join(directory, fn) for fn in os.listdir(directory) if fn.endswith(extension)):
            with self.subTest(msg=filename):
                with open(filename, "r") as f:
                    code = f.read()

                compile_function(code, filename)

    def test_examples_compilation(self):
        self._test_directory("examples", ".mu", lambda code, filename: compile_code(code, filename, 2))

    def test_asm_examples_compilation(self):
        self._test_directory("examples", ".mua", compile_asm_code)

    def test_stdlib_compilation(self):
        self._test_directory("mlogium/stdlib", ".mu", lambda code, filename: compile_code(code, filename, 2))


if __name__ == '__main__':
    unittest.main()
