from __future__ import annotations

from pathlib import Path
import math
import re
import shlex

import numpy as np


FORTRAN_FLOAT_RE = re.compile(r"(?<![A-Za-z])([+-]?(?:\d+(?:\.\d*)?|\.\d+))([+-]\d+)$")


def parse_fortran_float(token: str) -> float:
    text = token.strip().replace("D", "E").replace("d", "E")
    if not text:
        raise ValueError("empty float token")
    if "E" not in text.upper():
        match = FORTRAN_FLOAT_RE.match(text)
        if match:
            text = f"{match.group(1)}E{match.group(2)}"
    return float(text)


def parse_int(token: str) -> int:
    return int(token.strip())


def tokenize_legacy_line(line: str) -> list[str]:
    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    lexer.quotes = "'\""
    return list(lexer)


def is_absolute_path_like(text: str) -> bool:
    path = Path(text)
    return path.is_absolute() or (len(text) > 1 and text[1] == ":")


def read_fixed_width_floats(line: str, width: int = 11, limit: int | None = None) -> list[float]:
    values: list[float] = []
    upto = len(line) if limit is None else min(len(line), limit)
    for start in range(0, upto, width):
        chunk = line[start : start + width]
        if not chunk.strip():
            continue
        try:
            values.append(parse_fortran_float(chunk))
        except ValueError:
            continue
    return values


def read_all_floats(path: Path, fortran: bool = False) -> list[float]:
    values: list[float] = []
    for token in path.read_text().split():
        values.append(parse_fortran_float(token) if fortran else float(token))
    return values


def compute_bounds_from_midpoints(midpoints_mev: np.ndarray) -> np.ndarray:
    midpoints = np.asarray(midpoints_mev, dtype=float)
    if midpoints.ndim != 1 or midpoints.size == 0:
        raise ValueError("midpoints must be a non-empty one-dimensional array")
    ascending = midpoints[0] <= midpoints[-1]
    if not ascending:
        reversed_bounds = compute_bounds_from_midpoints(midpoints[::-1])
        return reversed_bounds[::-1]
    bounds = np.empty(midpoints.size + 1, dtype=float)
    bounds[0] = 1.0e-10 if midpoints[0] <= 1.0e-4 else 1.0e-4
    for idx in range(1, bounds.size):
        bounds[idx] = 2.0 * midpoints[idx - 1] - bounds[idx - 1]
    return bounds


def build_spectrum_columns(number_fraction: np.ndarray, bounds_mev: np.ndarray) -> np.ndarray:
    widths_ev = np.abs(np.diff(bounds_mev)) * 1.0e6
    mids_mev = 0.5 * (bounds_mev[:-1] + bounds_mev[1:])
    number_fraction = np.asarray(number_fraction, dtype=float)
    total = number_fraction.sum()
    if total <= 0.0:
        total = 1.0
    number_fraction = number_fraction / total
    energy_fraction = number_fraction * mids_mev * 1.0e6
    esum = energy_fraction.sum()
    if esum > 0.0:
        energy_fraction = energy_fraction / esum
    differential_number = np.divide(number_fraction, widths_ev, out=np.zeros_like(number_fraction), where=widths_ev != 0.0)
    differential_energy = np.divide(energy_fraction, widths_ev, out=np.zeros_like(energy_fraction), where=widths_ev != 0.0)
    integral_number = np.cumsum(number_fraction[::-1])[::-1]
    e_dn_de = differential_number * mids_mev
    return np.column_stack(
        [number_fraction, energy_fraction, differential_number, differential_energy, integral_number, e_dn_de]
    )


def fitmd(x: float, xarray: np.ndarray, yarray: np.ndarray, icode: int) -> float:
    xarray = np.asarray(xarray, dtype=float)
    yarray = np.asarray(yarray, dtype=float)
    if xarray.size != yarray.size:
        raise ValueError("xarray and yarray must have the same length")
    nx = xarray.size
    if nx < 2:
        return float(yarray[0]) if nx == 1 else 0.0
    ipick = 0
    while ipick + 1 < nx and xarray[ipick + 1] == xarray[ipick]:
        ipick += 1
    if ipick + 1 >= nx:
        if x == xarray[0]:
            return float(yarray[0])
        raise ValueError("fitmd cannot interpolate over a constant x array")
    direction = 1 if xarray[ipick + 1] > xarray[ipick] else -1
    if (x < xarray[0] and direction == 1) or (x > xarray[0] and direction == -1):
        return 0.0
    if (x < xarray[-1] and direction == -1) or (x > xarray[-1] and direction == 1):
        return 0.0
    bracket = None
    if direction == 1:
        for idx in range(nx):
            if x - xarray[idx] < 0.0:
                bracket = idx - 1
                break
    else:
        for idx in range(nx):
            if xarray[idx] - x < 0.0:
                bracket = idx - 1
                break
    if bracket is None:
        bracket = nx - 2
    bracket = max(0, min(bracket, nx - 2))
    return terp1(xarray[bracket], yarray[bracket], xarray[bracket + 1], yarray[bracket + 1], x, icode)


def terp1(x1: float, y1: float, x2: float, y2: float, x: float, icode: int) -> float:
    if y1 == y2:
        return y1
    if icode == 1:
        return y1
    if icode == 2:
        return y1 + (x - x1) * (y2 - y1) / (x2 - x1)
    if icode == 3:
        return y1 + math.log(x / x1) * (y2 - y1) / math.log(x2 / x1)
    if icode == 4:
        return y1 * math.exp((x - x1) * math.log(y2 / y1) / (x2 - x1))
    if icode == 5:
        if y1 == 0.0:
            return y1
        return y1 * math.exp(math.log(x / x1) * math.log(y2 / y1) / math.log(x2 / x1))
    raise ValueError(f"unsupported interpolation code: {icode}")


def average_stddev(values: np.ndarray, stddev_pct: np.ndarray, weights: np.ndarray) -> float:
    numer = np.sum(values * stddev_pct * weights)
    denom = np.sum(values * weights)
    return float(numer / denom) if denom != 0.0 else 0.0


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
