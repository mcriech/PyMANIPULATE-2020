from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import CurveTable
from .paths import official_grid_path
from .utils import (
    build_spectrum_columns,
    compute_bounds_from_midpoints,
    ensure_parent,
    fitmd,
    parse_fortran_float,
    read_all_floats,
    read_fixed_width_floats,
)


def read_energy_grid(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"invalid energy grid file: {path}")
    count = int(lines[1].split()[0])
    values: list[float] = []
    for line in lines[2:]:
        for token in line.split():
            values.append(float(token))
    if len(values) < count:
        raise ValueError(f"energy grid file {path} ended before {count} values were read")
    return np.asarray(values[:count], dtype=float)


def read_midpoint_pairs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mids: list[float] = []
    values: list[float] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        mids.append(parse_fortran_float(parts[0]))
        values.append(parse_fortran_float(parts[1]))
    if not mids:
        raise ValueError(f"no midpoint/value pairs found in {path}")
    return np.asarray(mids, dtype=float), np.asarray(values, dtype=float)


def read_grspin_table(path: Path, icon5: int | None = None, reverse_output: bool = False) -> CurveTable:
    mids, values = read_midpoint_pairs(path)
    if mids.size > 1 and mids[0] > mids[-1]:
        mids = mids[::-1]
        values = values[::-1]
    bounds = compute_bounds_from_midpoints(mids)
    if icon5 is not None:
        try:
            grid_path = official_grid_path_from_icon(path, icon5)
        except KeyError:
            grid_path = None
        if grid_path is not None and grid_path.exists():
            official = read_energy_grid(grid_path)
            if official.size == bounds.size:
                bounds = official
    table = CurveTable(bounds, values[:, np.newaxis], title=path.name)
    return table.reversed() if reverse_output else table


def official_grid_path_from_icon(path: Path, icon5: int) -> Path:
    repo_root = path.parents[1] if path.parent.name in {"response", "spectrum"} else path.parent
    while repo_root != repo_root.parent and not (repo_root / "response").exists():
        repo_root = repo_root.parent
    from .paths import PathResolver

    return official_grid_path(PathResolver(repo_root).build(), icon5)


def _parse_groupr_energy_bounds(lines: list[str], nenergy: int) -> np.ndarray:
    bounds_ev: list[float] = []
    line_index = 3
    first = read_fixed_width_floats(lines[line_index], limit=66)
    bounds_ev.extend(first[2:6])
    line_index += 1
    while len(bounds_ev) < nenergy + 1 and line_index < len(lines):
        bounds_ev.extend(read_fixed_width_floats(lines[line_index], limit=66))
        line_index += 1
    if len(bounds_ev) < nenergy + 1:
        raise ValueError("GROUPR energy grid ended early")
    return np.asarray(bounds_ev[: nenergy + 1], dtype=float) * 1.0e-6


def read_groupr_table(path: Path, reaction: int) -> CurveTable:
    lines = path.read_text().splitlines()
    if len(lines) < 4:
        raise ValueError(f"invalid GROUPR-like file: {path}")
    try:
        nenergy = int(lines[2][22:33])
    except ValueError as exc:
        raise ValueError(f"invalid GROUPR header in {path}") from exc
    bounds_mev = _parse_groupr_energy_bounds(lines, nenergy)
    start_index = None
    for idx, line in enumerate(lines):
        if len(line) >= 75:
            try:
                mt = int(line[72:75])
            except ValueError:
                continue
            next_has_index = False
            if idx + 1 < len(lines):
                next_slice = lines[idx + 1][63:66]
                next_has_index = bool(next_slice.strip())
            if mt == reaction and line[63:66].strip() and next_has_index:
                start_index = idx
                break
    if start_index is None:
        raise ValueError(f"reaction {reaction} not found in {path}")
    nend = int(lines[start_index][63:66])
    nstart = int(lines[start_index + 1][63:66])
    values = np.full(nenergy, 1.0e-35, dtype=float)
    line_index = start_index + 1
    target_points = nend - nstart + 1
    found_points = 0
    while line_index + 1 < len(lines) and found_points < target_points:
        header = lines[line_index]
        if not header[63:66].strip():
            line_index += 1
            continue
        try:
            index = int(header[63:66])
            mt = int(header[72:75])
        except ValueError:
            line_index += 1
            continue
        if mt != reaction:
            line_index += 1
            continue
        data_line = lines[line_index + 1]
        data_fields = read_fixed_width_floats(data_line, limit=22)
        if len(data_fields) >= 2:
            values[index - 1] = max(data_fields[1], 1.0e-35)
            found_points += 1
            line_index += 2
        else:
            line_index += 1
    return CurveTable(bounds_mev, values[:, np.newaxis], title=path.name)


def read_histogram_response(path: Path, grid_path: Path, reverse_storage: bool = False) -> CurveTable:
    lines = path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"invalid response histogram file: {path}")
    renorm = parse_fortran_float(lines[1].split()[0])
    nenergy = int(lines[2].split()[0])
    values = np.asarray([parse_fortran_float(tok) for tok in " ".join(lines[3:]).split()], dtype=float)
    if values.size < nenergy:
        raise ValueError(f"response file {path} ended before {nenergy} values were read")
    values = values[:nenergy] * renorm
    if reverse_storage:
        values = values[::-1]
    bounds = read_energy_grid(grid_path)
    if bounds.size != nenergy + 1:
        raise ValueError(f"grid {grid_path} does not match response length in {path}")
    return CurveTable(bounds, values[:, np.newaxis], title=path.name)


def read_list_spectrum(path: Path) -> CurveTable:
    lines = path.read_text().splitlines()
    nenergy = int(lines[0].split()[0])
    floats = [parse_fortran_float(tok) for tok in " ".join(lines[3:]).split()]
    if len(floats) < 2 * (nenergy + 1):
        raise ValueError(f"invalid list spectrum file: {path}")
    energies = np.asarray(floats[0::2][: nenergy + 1], dtype=float)
    values = np.asarray(floats[1::2][: nenergy + 1], dtype=float)
    if np.nanmax(energies) > 1.0e6:
        energies = energies * 1.0e-6
    bounds = energies
    differential = values[:-1]
    widths_ev = np.abs(np.diff(bounds)) * 1.0e6
    number_fraction = np.abs(differential * widths_ev)
    columns = build_spectrum_columns(number_fraction, bounds)
    return CurveTable(bounds, columns, title=path.name)


def read_plain_midpoint_table(path: Path, nenergy_first: bool = False) -> CurveTable:
    lines = path.read_text().splitlines()
    if nenergy_first:
        nenergy = int(lines[0].split()[0])
        floats = [parse_fortran_float(tok) for tok in " ".join(lines[1:]).split()]
        mids = np.asarray(floats[0::2][:nenergy], dtype=float)
        values = np.asarray(floats[1::2][:nenergy], dtype=float)
    else:
        mids, values = read_midpoint_pairs(path)
    bounds = compute_bounds_from_midpoints(mids)
    return CurveTable(bounds, values[:, np.newaxis], title=path.name)


def read_user_flux(path: Path) -> CurveTable:
    lines = path.read_text().splitlines()
    header = lines[0].split()
    nenergy = int(header[1])
    comment_count = int(header[0])
    data_tokens = [parse_fortran_float(tok) for tok in " ".join(lines[1 + comment_count :]).split()]
    mids = np.asarray(data_tokens[0::2][:nenergy], dtype=float)
    values = np.asarray(data_tokens[1::2][:nenergy], dtype=float)
    bounds = compute_bounds_from_midpoints(mids)
    columns = build_spectrum_columns(values, bounds)
    return CurveTable(bounds, columns, title=path.name, comments=lines[1 : 1 + comment_count])


def read_lsl_spectrum(path: Path) -> CurveTable:
    lines = path.read_text().splitlines()
    count = abs(int(lines[2].split()[0]))
    energies_ev = np.asarray([parse_fortran_float(tok) for tok in " ".join(lines[4:6]).split()], dtype=float)
    if energies_ev.size < count:
        energies_ev = np.asarray(read_all_floats(path, fortran=True), dtype=float)
    bounds = energies_ev[:count] * 1.0e-6
    values = np.asarray([parse_fortran_float(tok) for tok in " ".join(lines[6:]).split()], dtype=float)
    number_fraction = values[: count - 1]
    columns = build_spectrum_columns(number_fraction, bounds)
    return CurveTable(bounds, columns, title=path.name)


def read_mf5_spectrum(path: Path) -> CurveTable:
    lines = path.read_text().splitlines()
    header = read_fixed_width_floats(lines[2], limit=33)
    nenergy = int(round(header[2]))
    bounds_ev: list[float] = []
    bounds_ev.extend(read_fixed_width_floats(lines[3], limit=66)[2:])
    line_index = 4
    while len(bounds_ev) < nenergy + 1:
        bounds_ev.extend(read_fixed_width_floats(lines[line_index], limit=66))
        line_index += 1
    while line_index < len(lines) and " 5 18" not in lines[line_index][70:75]:
        line_index += 1
    if line_index >= len(lines) - 1:
        raise ValueError(f"MF=5 MT=18 section not found in {path}")
    values = np.asarray(read_fixed_width_floats(lines[line_index + 1], limit=66), dtype=float)
    if values.size < nenergy:
        raise ValueError(f"MF5 spectrum in {path} ended early")
    number_fraction = values[:nenergy]
    columns = build_spectrum_columns(number_fraction, np.asarray(bounds_ev[: nenergy + 1], dtype=float) * 1.0e-6)
    return CurveTable(np.asarray(bounds_ev[: nenergy + 1], dtype=float) * 1.0e-6, columns, title=path.name)


def write_pair_table(path: Path, mids_mev: np.ndarray, values: np.ndarray, reverse: bool = False, scale: float = 1.0) -> None:
    ensure_parent(path)
    mids = np.asarray(mids_mev, dtype=float)
    vals = np.asarray(values, dtype=float) * scale
    if reverse:
        mids = mids[::-1]
        vals = vals[::-1]
    with path.open("w", encoding="utf-8") as handle:
        for mid, value in zip(mids, vals):
            handle.write(f"{mid:16.7E} {value:16.7E}\n")


def write_step_pairs(path: Path, bounds_mev: np.ndarray, values: np.ndarray) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for low, high, value in zip(bounds_mev[:-1], bounds_mev[1:], values):
            handle.write(f"{low:16.7E} {value:16.7E}\n")
            handle.write(f"{high:16.7E} {value:16.7E}\n")


def make_rebin_lines(table: CurveTable) -> list[str]:
    columns = build_spectrum_columns(table.column(1), table.bounds_mev)
    number_fraction = columns[:, 0]
    bounds = table.bounds_mev
    return [
        "Rebin format interface file ",
        " ".join(f"{value:14.7E}" for value in bounds[::-1]),
        " ".join(f"{value:14.7E}" for value in number_fraction[::-1]),
    ]
