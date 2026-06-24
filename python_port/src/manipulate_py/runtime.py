from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .models import PathContext
from .utils import ensure_parent


@dataclass
class LegacyRuntime:
    paths: PathContext
    job_name: str
    out_lines: list[str] = field(default_factory=list)
    ext_lines: list[str] = field(default_factory=list)
    pun2_lines: list[str] = field(default_factory=list)
    rebin_lines: list[str] = field(default_factory=list)

    def log(self, line: str = "") -> None:
        self.out_lines.append(line)

    def ext(self, line: str = "") -> None:
        self.ext_lines.append(line)

    def pun2(self, line: str = "") -> None:
        self.pun2_lines.append(line)

    def set_rebin(self, lines: list[str]) -> None:
        self.rebin_lines = list(lines)

    def write_text(self, path: Path, lines: list[str]) -> None:
        ensure_parent(path)
        path.write_text("\n".join(lines).rstrip() + ("\n" if lines else ""))

    def finalize(self) -> None:
        out_path = self.paths.output_dir / f"{self.job_name}.out"
        ext_path = self.paths.output_dir / f"{self.job_name}.ext"
        self.write_text(out_path, self.out_lines)
        self.write_text(ext_path, self.ext_lines)
        if self.pun2_lines:
            self.write_text(self.paths.punch_dir / f"{self.job_name}.pun2", self.pun2_lines)
        if self.rebin_lines:
            self.write_text(self.paths.punch_dir / f"{self.job_name}.rebin", self.rebin_lines)
