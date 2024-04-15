import argparse
import os

import pyperclip

from . import __version__
from .compile import compile_code
from .error import PositionedException


def main() -> int:
    parser = argparse.ArgumentParser(description="High level mindustry logic language", prog="mlogium")

    parser.add_argument("file", type=str, help="input file [@clip for clipboard]")

    parser.add_argument("-o:f", "--output-file", help="write output to a file")
    parser.add_argument("-o:s", "--output-stdout", help="write output to stdout", action="store_true")
    parser.add_argument("-o:c", "--output-clip", help="write output to clipboard (default)", action="store_true")

    parser.add_argument("-v", "--verbose", help="print additional information", action="store_true")
    parser.add_argument("-l", "--lines", help="print line numbers when output to stdout is selected",
                        action="store_true")

    parser.add_argument("--print-exceptions", help="print all exceptions from the compilation",
                        action="store_true")

    parser.add_argument("-V", "--version", action="version", version=f"mlogium {__version__}")

    args = parser.parse_args()

    output_method = "clip"
    output_file = ""

    verbose = False
    lines = False

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
        result = compile_code(code, filename).strip()
    except PositionedException as e:
        print(e.msg)
        e.pos.print()
        if args.print_exceptions:
            raise e
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
