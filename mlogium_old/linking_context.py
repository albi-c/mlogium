class LinkingContext:
    _tmp_index: int

    def __init__(self):
        self._tmp_index = 0

    def tmp(self) -> str:
        self._tmp_index += 1
        return f"__tmp{self._tmp_index}"

    def tmp_num(self) -> int:
        self._tmp_index += 1
        return self._tmp_index
