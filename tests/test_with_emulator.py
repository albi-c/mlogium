import os.path
import unittest
from typing import Callable

from mlogium.error import PositionedException, NonPositionedException
from mlogium.compile import compile_code, compile_asm_code
from mlog_emulator import Executor, Device, ExecutionResult


class EmulatedTestCase(unittest.TestCase):
    @staticmethod
    def _run(code: str) -> tuple[str, str]:
        ex = Executor(code)
        ex.end_on_wrap = True
        ex.add_device("message1", Device.Message())

        res = ex.execute()
        assert isinstance(res, ExecutionResult.Success)

        return res.devices["message1"].text, res.print_buffer

    def _run_test(self, path: str, name: str, expected: tuple[str, str], compile_function: Callable[[str, str], str]):
        directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
        filename = os.path.join(directory, name)

        with self.subTest(msg=f"Compile {filename}"):
            with open(filename, "r") as f:
                code = f.read()

            compiled = compile_function(code, filename)

        with self.subTest(msg=f"Run {filename}"):
            result = self._run(compiled)
            assert result == expected, (result, expected)

    @staticmethod
    def _reraise(res: PositionedException | NonPositionedException | str) -> str:
        if isinstance(res, PositionedException):
            raise res
        elif isinstance(res, NonPositionedException):
            raise res
        else:
            return res

    def test_examples(self):
        for name, expected in (
            ("comptime", ("", "120")),
            ("connected_list", ("message1\\n", "")),
            ("hello_world", ("Hello, World!", "")),
            ("lambda", ("", "99818189")),
        ):
            self._run_test("examples", f"{name}.mu", expected,
                           lambda code, filename: self._reraise(compile_code(code, filename, 2)[0]))

    def test_asm_examples(self):
        for name, expected in (
            ("connected_list", ("message1\\n", "")),
        ):
            self._run_test("examples", f"{name}.mua", expected,
                           lambda code, filename: self._reraise(compile_asm_code(code, filename)))


if __name__ == '__main__':
    unittest.main()
