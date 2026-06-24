from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import CovarianceData
from .utils import ensure_parent, fitmd, parse_fortran_float, read_fixed_width_floats


def _flatten_floats(lines: list[str]) -> list[float]:
    values: list[float] = []
    for line in lines:
        for token in line.split():
            values.append(parse_fortran_float(token))
    return values


def read_snlcov(path: Path) -> CovarianceData:
    lines = path.read_text().splitlines()
    nenergy = int(lines[0].split()[0])
    floats = _flatten_floats(lines[1:])
    cursor = 0
    bounds = np.asarray(floats[cursor : cursor + nenergy + 1], dtype=float)
    cursor += nenergy + 1
    values1 = np.asarray(floats[cursor : cursor + nenergy], dtype=float)
    cursor += nenergy
    std1 = np.asarray(floats[cursor : cursor + nenergy], dtype=float)
    cursor += nenergy
    values2 = np.asarray(floats[cursor : cursor + nenergy], dtype=float)
    cursor += nenergy
    std2 = np.asarray(floats[cursor : cursor + nenergy], dtype=float)
    cursor += nenergy
    covariance = np.asarray(floats[cursor : cursor + nenergy * nenergy], dtype=float).reshape(nenergy, nenergy)
    cursor += nenergy * nenergy
    correlation = np.asarray(floats[cursor : cursor + nenergy * nenergy], dtype=float).reshape(nenergy, nenergy)
    return CovarianceData(
        bounds,
        values1,
        std1,
        values2,
        std2,
        covariance,
        correlation,
        bool(np.allclose(values1, values2)),
        path.stem,
    )


def _read_lsl_block(lines: list[str], start: int, count: int) -> tuple[np.ndarray, int]:
    values: list[float] = []
    index = start
    while index < len(lines) and len(values) < count:
        stripped = lines[index].strip()
        if stripped.startswith("*") and values:
            break
        if stripped and not stripped.startswith("*"):
            values.extend(parse_fortran_float(token) for token in stripped.split())
        index += 1
    if len(values) < count:
        raise ValueError("LSL block ended before the requested value count was read")
    return np.asarray(values[:count], dtype=float), index


def _parse_lsl(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lines = path.read_text().splitlines()
    npoints_plus_one = abs(int(lines[2].split()[0]))
    nenergy = npoints_plus_one - 1
    bounds_ev, index = _read_lsl_block(lines, 4, npoints_plus_one)
    while index < len(lines) and not lines[index].strip().startswith("*"):
        index += 1
    values, index = _read_lsl_block(lines, index + 1, nenergy)
    while index < len(lines) and not lines[index].strip().startswith("*"):
        index += 1
    stddev, index = _read_lsl_block(lines, index + 1, nenergy)
    while index < len(lines) and not lines[index].strip().startswith("*"):
        index += 1
    corr = np.zeros((nenergy, nenergy), dtype=float)
    index += 1
    for row in range(nenergy):
        segment, index = _read_lsl_block(lines, index, nenergy - row)
        corr[row, row:] = segment
        corr[row:, row] = segment
    return bounds_ev * 1.0e-6, values, stddev, corr


def read_lsl_dosimetry(path: Path) -> CovarianceData:
    bounds, values, stddev, corr = _parse_lsl(path)
    diag = np.diag(corr)
    scale = 1.0
    if np.nanmax(diag) > 1100.0:
        scale = 1.0e-4
    elif np.nanmax(diag) > 1.1:
        scale = 1.0e-2
    corr = corr * scale
    covariance = correlation_to_covariance(values, stddev, values, stddev, corr)
    return CovarianceData(bounds, values, stddev, values.copy(), stddev.copy(), covariance, corr, True, path.stem)


def read_lsl_spectrum_covariance(path: Path) -> CovarianceData:
    bounds, values, stddev, corr = _parse_lsl(path)
    corr = corr * 1.0e-2
    total = values.sum()
    if total > 0.0:
        values = values / total
    covariance = correlation_to_covariance(values, stddev, values, stddev, corr)
    return CovarianceData(bounds, values, stddev, values.copy(), stddev.copy(), covariance, corr, True, path.stem)


def read_endf_absolute_covariance(path: Path) -> CovarianceData:
    lines = path.read_text().splitlines()
    npoints_plus_one = int(lines[1].split()[0])
    nenergy = npoints_plus_one - 1
    first_block = _flatten_floats(lines[2 : 2 + nenergy])
    values = np.asarray(first_block[3::8][:nenergy], dtype=float)
    raw_std = np.asarray(first_block[4::8][:nenergy], dtype=float)
    tail = _flatten_floats(lines[7:])
    bounds = np.asarray(tail[:npoints_plus_one], dtype=float) * 1.0e-6
    tri = np.asarray(tail[npoints_plus_one : npoints_plus_one + nenergy * (nenergy + 1) // 2], dtype=float)
    corr = np.zeros((nenergy, nenergy), dtype=float)
    cursor = 0
    stddev = raw_std.copy()
    for row in range(nenergy):
        for col in range(row, nenergy):
            abs_cov = tri[cursor]
            cursor += 1
            if row == col and values[row] != 0.0 and stddev[row] != 0.0:
                corr_diag = abs_cov / (values[row] * values[row] * (stddev[row] * 0.01) ** 2)
                if corr_diag > 0.0:
                    stddev[row] = raw_std[row] * np.sqrt(corr_diag)
    cursor = 0
    for row in range(nenergy):
        for col in range(row, nenergy):
            abs_cov = tri[cursor]
            cursor += 1
            denom = values[row] * values[col] * (stddev[row] * 0.01) * (stddev[col] * 0.01)
            corr_val = abs_cov / denom if denom != 0.0 else (1.0 if row == col else 0.0)
            corr[row, col] = corr[col, row] = corr_val
    covariance = correlation_to_covariance(values, stddev, values, stddev, corr)
    return CovarianceData(bounds, values, stddev, values.copy(), stddev.copy(), covariance, corr, True, path.stem)


def read_pdf_covariance(bounds_mev: np.ndarray, number_fraction: np.ndarray, pdf_points: np.ndarray) -> CovarianceData:
    total = number_fraction.sum()
    values = number_fraction / total if total > 0.0 else number_fraction.copy()
    stddev = np.zeros_like(values)
    corr = np.eye(values.size, dtype=float)
    covariance = np.zeros_like(corr)
    covdata = CovarianceData(bounds_mev, values, stddev, values.copy(), stddev.copy(), covariance, corr, True, "pdf")
    setattr(covdata, "pdf", pdf_points)
    return covdata


def read_fcov(path: Path) -> CovarianceData:
    lines = path.read_text().splitlines()
    npoints_plus_one = int(lines[6].split()[0])
    nenergy = npoints_plus_one - 1
    bounds = np.asarray(_flatten_floats([lines[7]]), dtype=float)[:npoints_plus_one] * 1.0e-6
    stddev = np.asarray(_flatten_floats([lines[8]]), dtype=float)[:nenergy]
    corr = np.zeros((nenergy, nenergy), dtype=float)
    tri_values = _flatten_floats(lines[9:])
    cursor = 0
    for row in range(nenergy):
        count = nenergy - row
        segment = np.asarray(tri_values[cursor : cursor + count], dtype=float) * 1.0e-3
        cursor += count
        corr[row, row:] = segment
        corr[row:, row] = segment
    values = np.full(nenergy, 1.0 / max(nenergy, 1), dtype=float)
    covariance = correlation_to_covariance(values, stddev, values, stddev, corr)
    return CovarianceData(bounds, values, stddev, values.copy(), stddev.copy(), covariance, corr, True, path.stem)


def correlation_to_covariance(
    values1: np.ndarray,
    stddev1_pct: np.ndarray,
    values2: np.ndarray,
    stddev2_pct: np.ndarray,
    corr: np.ndarray,
) -> np.ndarray:
    sigma1 = values1 * stddev1_pct * 0.01
    sigma2 = values2 * stddev2_pct * 0.01
    return corr * np.outer(sigma1, sigma2)


def repair_positive_semidefinite(covdata: CovarianceData, normalize_spectrum: bool = False) -> CovarianceData:
    corr = 0.5 * (covdata.correlation + covdata.correlation.T)
    np.fill_diagonal(corr, np.where(np.diag(corr) == 0.0, 1.0, np.diag(corr)))
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    clipped = np.clip(eigenvalues, 1.0e-35, None)
    repaired = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    repaired = 0.5 * (repaired + repaired.T)
    diag = np.sqrt(np.clip(np.diag(repaired), 1.0e-35, None))
    repaired = repaired / np.outer(diag, diag)
    repaired = np.clip(repaired, -1.0, 1.0)
    np.fill_diagonal(repaired, 1.0)
    result = covdata.copy()
    result.correlation = repaired
    result.covariance = correlation_to_covariance(result.values1, result.stddev1_pct, result.values2, result.stddev2_pct, repaired)
    if normalize_spectrum:
        apply_spectrum_normalization(result)
    return result


def apply_spectrum_normalization(covdata: CovarianceData) -> None:
    values = covdata.values1.copy()
    total = values.sum()
    if total > 0.0:
        values = values / total
    sigma = values * covdata.stddev1_pct * 0.01
    abs_cov = covdata.correlation * np.outer(sigma, sigma)
    row_norm = abs_cov.sum(axis=1)
    if np.max(np.abs(row_norm)) <= 1.0e-5:
        covdata.values1 = values
        covdata.values2 = values.copy()
        covdata.covariance = abs_cov
        return
    delta = np.eye(values.size) - values[:, np.newaxis]
    abs2_cov = delta @ abs_cov @ delta.T
    new_std = np.zeros_like(values)
    nonzero = values != 0.0
    new_std[nonzero] = np.sqrt(np.clip(np.diag(abs2_cov)[nonzero], 0.0, None)) * 100.0 / values[nonzero]
    corr = np.zeros_like(abs2_cov)
    denom = np.outer(values * new_std * 0.01, values * new_std * 0.01)
    np.divide(abs2_cov, denom, out=corr, where=denom != 0.0)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, np.where(values != 0.0, 1.0, 0.0))
    covdata.values1 = values
    covdata.values2 = values.copy()
    covdata.stddev1_pct = new_std
    covdata.stddev2_pct = new_std.copy()
    covdata.covariance = abs2_cov
    covdata.correlation = corr


def regrid_covariance(covdata: CovarianceData, target_bounds_mev: np.ndarray, is_spectrum: bool) -> CovarianceData:
    if covdata.bounds_mev.shape == target_bounds_mev.shape and np.allclose(covdata.bounds_mev, target_bounds_mev):
        return covdata.copy()
    target_n = target_bounds_mev.size - 1
    src_bounds_ev = covdata.bounds_mev * 1.0e6
    tgt_bounds_ev = target_bounds_mev * 1.0e6
    std1 = _histogram_expand(covdata.stddev1_pct, src_bounds_ev, tgt_bounds_ev, differential=False)
    std2 = _histogram_expand(covdata.stddev2_pct, src_bounds_ev, tgt_bounds_ev, differential=False)
    values1 = _histogram_expand(covdata.values1, src_bounds_ev, tgt_bounds_ev, differential=is_spectrum)
    values2 = _histogram_expand(covdata.values2, src_bounds_ev, tgt_bounds_ev, differential=is_spectrum)
    src_mids = 0.5 * (src_bounds_ev[:-1] + src_bounds_ev[1:])
    mapping = np.searchsorted(src_bounds_ev[1:], 0.5 * (tgt_bounds_ev[:-1] + tgt_bounds_ev[1:]), side="right")
    mapping = np.clip(mapping, 0, covdata.nenergy - 1)
    corr = covdata.correlation[np.ix_(mapping, mapping)]
    covariance = correlation_to_covariance(values1, std1, values2, std2, corr)
    result = CovarianceData(target_bounds_mev, values1, std1, values2, std2, covariance, corr, covdata.is_self_covariance, covdata.source_name)
    if is_spectrum:
        apply_spectrum_normalization(result)
    return result


def _histogram_expand(values: np.ndarray, src_bounds_ev: np.ndarray, tgt_bounds_ev: np.ndarray, differential: bool) -> np.ndarray:
    src = values.copy()
    if differential:
        widths = np.diff(src_bounds_ev)
        src = np.divide(src, widths, out=np.zeros_like(src), where=widths != 0.0)
    xarray = np.concatenate([[0.0], src_bounds_ev[:-1], [src_bounds_ev[-1]]])
    yarray = np.concatenate([[0.0], src, [0.0]])
    out = np.array([fitmd(bound, xarray, yarray, 1) for bound in tgt_bounds_ev[:-1]], dtype=float)
    if differential:
        out = out * np.diff(tgt_bounds_ev)
        total = out.sum()
        if total > 0.0:
            out = out / total
    return out


def write_snlcov(path: Path, covdata: CovarianceData) -> None:
    ensure_parent(path)
    n = covdata.nenergy
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{n:10d}\n")
        _write_float_block(handle, covdata.bounds_mev)
        _write_float_block(handle, covdata.values1)
        _write_float_block(handle, covdata.stddev1_pct)
        _write_float_block(handle, covdata.values2)
        _write_float_block(handle, covdata.stddev2_pct)
        _write_float_block(handle, covdata.covariance.reshape(-1))
        _write_float_block(handle, covdata.correlation.reshape(-1))


def write_lsl(path: Path, covdata: CovarianceData, spectrum_mode: bool = False) -> None:
    ensure_parent(path)
    label = "*Number Fractions " if spectrum_mode else "*Cross Section (barns)"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("*COR    (LIBRARY)    (MAT.#)    (TEMP)K\n")
        handle.write("*Number of Energies plus 1\n")
        handle.write(f"{covdata.nenergy + 1:5d}\n")
        handle.write("*Energy Grid ( eV )\n")
        _write_float_block(handle, covdata.bounds_mev * 1.0e6)
        handle.write(f"{label}\n")
        _write_float_block(handle, covdata.values1)
        handle.write("*% Standard Deviation\n")
        _write_float_block(handle, covdata.stddev1_pct)
        handle.write("*Correlation Coefficient -- Upper Triangular\n")
        for row in range(covdata.nenergy):
            _write_float_block(handle, covdata.correlation[row, row:] * 100.0)


def write_plot_interfaces(stem: Path, covdata: CovarianceData) -> None:
    std_path = stem.with_name(stem.name + ".std_pct_plt")
    cor_path = stem.with_name(stem.name + ".corplt")
    ensure_parent(std_path)
    with std_path.open("w", encoding="utf-8") as handle:
        first = covdata.stddev1_pct[0] * 1.0e-3
        handle.write(f"{covdata.bounds_mev[0]:14.7E} {first:14.7E}\n")
        for idx in range(covdata.nenergy):
            handle.write(f"{covdata.bounds_mev[idx]:14.7E} {covdata.stddev1_pct[idx]:14.7E}\n")
            handle.write(f"{covdata.bounds_mev[idx + 1]:14.7E} {covdata.stddev1_pct[idx]:14.7E}\n")
        last = covdata.stddev1_pct[-1] * 1.0e-3
        handle.write(f"{covdata.bounds_mev[-1]:14.7E} {last:14.7E}\n")
    with cor_path.open("w", encoding="utf-8") as handle:
        for row in range(covdata.nenergy):
            row_values = ", ".join(f"{value:12.4G}" for value in covdata.correlation[row, :])
            handle.write(f"{covdata.bounds_mev[row]:12.4G}, {covdata.bounds_mev[row]:12.4G}, {row_values}\n")


def _write_float_block(handle, values: np.ndarray, per_line: int = 5) -> None:
    flat = np.asarray(values, dtype=float).reshape(-1)
    for start in range(0, flat.size, per_line):
        chunk = " ".join(f"{value:14.7E}" for value in flat[start : start + per_line])
        handle.write(f" {chunk}\n")
