from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .utils import tokenize_legacy_line


@dataclass
class InputCursor:
    path: Path
    lines: list[str]
    index: int = 0

    @classmethod
    def from_path(cls, path: Path) -> "InputCursor":
        return cls(path=path, lines=path.read_text().splitlines())

    def next_line(self, skip_blank: bool = True) -> str:
        while self.index < len(self.lines):
            line = self.lines[self.index]
            self.index += 1
            if skip_blank and not line.strip():
                continue
            return line
        raise EOFError(f"unexpected end of file while reading {self.path}")

    def next_tokens(self) -> list[str]:
        return tokenize_legacy_line(self.next_line())

    def next_ints(self) -> list[int]:
        return [int(token) for token in self.next_tokens()]
