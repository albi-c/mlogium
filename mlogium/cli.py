import argparse
import os

import pyperclip

from . import __version__
from .compile import compile_code, compile_asm_code
from .error import PositionedException, NonPositionedException
from .util import Position


def print_notes(notes: list[tuple[str, Position | None]]):
    if len(notes) == 0:
        return

    print()
    for note in notes:
        print("Note:", note[0])
        if note[1] is not None:
            note[1].print()
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="High level mindustry logic language", prog="mlogium")

    parser.add_argument("file", type=str, help="input file [@clip for clipboard]")

    parser.add_argument("-a", "--assembly", help="compile as mlogium assembly", action="store_true")

    parser.add_argument("-o:f", "--output-file", help="write output to a file")
    parser.add_argument("-o:s", "--output-stdout", help="write output to stdout", action="store_true")
    parser.add_argument("-o:c", "--output-clip", help="write output to clipboard (default)", action="store_true")

    parser.add_argument("-v", "--verbose", help="print additional information", action="store_true")
    parser.add_argument("-l", "--lines", help="print line numbers when output to stdout is selected",
                        action="store_true")

    parser.add_argument("-O", "--opt", "--optimize", type=int, help="optimization level", default=2)

    parser.add_argument("--print-exceptions", help="print all exceptions from the compilation",
                        action="store_true")

    parser.add_argument("-V", "--version", action="version", version=f"mlogium {__version__}")

    args = parser.parse_args()

    output_method = "clip"
    output_file = ""

    verbose = False
    lines = False

    opt_level = args.opt
    if opt_level < 0:
        print("Optimization level must be at least 0")
        exit(1)

    for k, v in vars(args).items():
        if v:
            if k.startswith("output"):
                output_method = k.split("_", 1)[-1]
                if output_method == "file":
                    output_file = v

            elif k == "verbose":
                verbose = v

            elif k == "lines":
                lines = v

    if not os.path.isfile(args.file) and args.file != "@clip":
        print(f"Error: can't open input file '{args.file}'")
        return 1

    if args.file == "@clip":
        code = pyperclip.paste()
        filename = "<clip>"
    else:
        code = open(args.file).read()
        filename = args.file

    try:
        if args.assembly:
            result = compile_asm_code(code, filename).strip()
            notes = []
        else:
            result, notes = compile_code(code, filename, opt_level)
    except PositionedException as e:
        result = e
        notes = []
    except NonPositionedException as e:
        result = e
        notes = []

    if isinstance(result, PositionedException):
        print("Error:", result.msg)
        if result.pos is not None:
            result.pos.print()
        print_notes(notes)
        if args.print_exceptions:
            raise result
        return 1
    elif isinstance(result, NonPositionedException):
        print("Error:", result.msg)
        print_notes(notes)
        if args.print_exceptions:
            raise result
        return 1

    if output_method == "file":
        with open(output_file, "w+") as f:
            f.write(result)

    elif output_method == "clip":
        pyperclip.copy(result)

    elif output_method == "stdout":
        if lines:
            lns = result.splitlines()
            max_line = len(str(len(lns) - 1))
            for i, ln in enumerate(lns):
                print(f"{str(i).zfill(max_line)}: {ln}")

        else:
            print(result)

        if verbose:
            print()

    else:
        assert False, output_method

    if verbose:
        print(f"Output: {len(result)} characters, {len(result.splitlines())} lines")

    return 0
