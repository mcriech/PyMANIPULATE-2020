from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import PathContext


OFFICIAL_GRID_BY_ICON5 = {
    0: "sand641.nrg",
    1: "sand771.nrg",
    2: "nuget90.nrg",
    3: "nuget49.nrg",
    5: "vit176.nrg",
    6: "IAEA725.nrg",
}


@dataclass
class PathResolver:
    repo_root: Path

    def build(self) -> PathContext:
        root = self.repo_root.resolve()
        work_dir = root / "snl-work-python"
        work_dir.mkdir(parents=True, exist_ok=True)
        (root / "output" / "punch").mkdir(parents=True, exist_ok=True)
        return PathContext(
            repo_root=root,
            work_dir=work_dir,
            input_dir=root / "input",
            output_dir=root / "output",
            punch_dir=root / "output" / "punch",
            response_dir=root / "response",
            spectrum_dir=root / "spectrum",
            covar_dir=root / "covar",
            library_dir=root / "library",
            documentation_dir=root / "documentation",
            groupr_support_dir=root / "use_case_support_files" / "NJOY-2016_Groupr",
            correlation_support_dir=root / "use_case_support_files" / "NJOY-2016_correlation",
            self_shield_dir=root / "use_case_support_files" / "SNL-LSL_self_shield",
            lsl_output_dir=root / "use_case_support_files" / "SNL-LSL_output",
        )


def existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(str(candidates[0]))


def resolve_work_relative(paths: PathContext, text: str) -> Path:
    path = Path(text)
    if path.is_absolute() or (len(text) > 1 and text[1] == ":"):
        return path
    return (paths.work_dir / path).resolve()


def resolve_any(paths: PathContext, text: str) -> Path:
    path = Path(text)
    if path.is_absolute() or (len(text) > 1 and text[1] == ":"):
        return path
    candidates = [
        resolve_work_relative(paths, text),
        (paths.repo_root / text).resolve(),
        (paths.response_dir / text).resolve(),
        (paths.spectrum_dir / text).resolve(),
        (paths.covar_dir / text).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_response_like(paths: PathContext, text: str) -> Path:
    candidates = [
        resolve_work_relative(paths, text),
        (paths.response_dir / text).resolve(),
        (paths.repo_root / text).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_spectrum_like(paths: PathContext, text: str) -> Path:
    candidates = [
        resolve_work_relative(paths, text),
        (paths.spectrum_dir / text).resolve(),
        (paths.repo_root / text).resolve(),
        (paths.response_dir / text).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_group_source(paths: PathContext, text: str) -> Path:
    candidates = [
        resolve_work_relative(paths, text),
        (paths.groupr_support_dir / text).resolve(),
        (paths.response_dir / text).resolve(),
        (paths.spectrum_dir / text).resolve(),
        (paths.repo_root / text).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_correlation_stem(paths: PathContext, stem: str, suffix: str) -> Path:
    filename = f"{stem}{suffix}"
    candidates = [
        (paths.correlation_support_dir / filename).resolve(),
        (paths.covar_dir / filename).resolve(),
        (paths.repo_root / filename).resolve(),
        resolve_work_relative(paths, filename),
    ]
    return existing_path(*candidates)


def resolve_self_shield(paths: PathContext, stem: str) -> Path:
    filename = f"{stem}.summary"
    return existing_path((paths.self_shield_dir / filename).resolve(), resolve_work_relative(paths, filename))


def official_grid_path(paths: PathContext, icon5: int) -> Path:
    name = OFFICIAL_GRID_BY_ICON5.get(icon5)
    if name is None:
        raise KeyError(f"unsupported icon(5) grid code: {icon5}")
    return paths.response_dir / name
