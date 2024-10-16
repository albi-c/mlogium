import os.path
import unittest
from typing import Callable, Any

from mlogium.compile import compile_code, compile_asm_code


class ExamplesTestCast(unittest.TestCase):
    def _test_examples(self, extension: str, compile_function: Callable[[str, str], Any]):
        directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")
        for filename in (os.path.join(directory, fn) for fn in os.listdir(directory) if fn.endswith(extension)):
            with self.subTest(msg=filename):
                with open(filename, "r") as f:
                    code = f.read()

                compile_function(code, filename)

    def test_examples_compilation(self):
        self._test_examples(".mu", compile_code)

    def test_asm_examples_compilation(self):
        self._test_examples(".mua", compile_asm_code)


if __name__ == '__main__':
    unittest.main()
