from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


FloatArray = np.ndarray


@dataclass
class CurveTable:
    bounds_mev: FloatArray
    values: FloatArray
    title: str = ""
    comments: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.bounds_mev = np.asarray(self.bounds_mev, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim == 1:
            self.values = self.values[:, np.newaxis]
        if self.bounds_mev.ndim != 1:
            raise ValueError("bounds_mev must be one-dimensional")
        if self.values.ndim != 2:
            raise ValueError("values must be two-dimensional")
        if self.values.shape[0] != self.bounds_mev.size - 1:
            raise ValueError("values row count must equal number of energy bins")

    @property
    def nenergy(self) -> int:
        return self.bounds_mev.size - 1

    @property
    def midpoints_mev(self) -> FloatArray:
        return 0.5 * (self.bounds_mev[:-1] + self.bounds_mev[1:])

    def column(self, index_1_based: int) -> FloatArray:
        if index_1_based < 1 or index_1_based > self.values.shape[1]:
            raise IndexError("column index out of range")
        return self.values[:, index_1_based - 1].copy()

    def with_values(self, values: FloatArray) -> "CurveTable":
        return CurveTable(self.bounds_mev.copy(), values, self.title, list(self.comments))

    def reversed(self) -> "CurveTable":
        bounds = self.bounds_mev[::-1].copy()
        values = self.values[::-1].copy()
        return CurveTable(bounds, values, self.title, list(self.comments))


@dataclass
class CovarianceData:
    bounds_mev: FloatArray
    values1: FloatArray
    stddev1_pct: FloatArray
    values2: FloatArray
    stddev2_pct: FloatArray
    covariance: FloatArray
    correlation: FloatArray
    is_self_covariance: bool = True
    source_name: str = ""

    def __post_init__(self) -> None:
        self.bounds_mev = np.asarray(self.bounds_mev, dtype=float)
        self.values1 = np.asarray(self.values1, dtype=float)
        self.stddev1_pct = np.asarray(self.stddev1_pct, dtype=float)
        self.values2 = np.asarray(self.values2, dtype=float)
        self.stddev2_pct = np.asarray(self.stddev2_pct, dtype=float)
        self.covariance = np.asarray(self.covariance, dtype=float)
        self.correlation = np.asarray(self.correlation, dtype=float)
        n = self.bounds_mev.size - 1
        if any(arr.size != n for arr in (self.values1, self.stddev1_pct, self.values2, self.stddev2_pct)):
            raise ValueError("vector lengths do not match covariance energy grid")
        if self.covariance.shape != (n, n) or self.correlation.shape != (n, n):
            raise ValueError("covariance matrices must be square with matching dimension")

    @property
    def nenergy(self) -> int:
        return self.bounds_mev.size - 1

    @property
    def midpoints_mev(self) -> FloatArray:
        return 0.5 * (self.bounds_mev[:-1] + self.bounds_mev[1:])

    def copy(self) -> "CovarianceData":
        return CovarianceData(
            self.bounds_mev.copy(),
            self.values1.copy(),
            self.stddev1_pct.copy(),
            self.values2.copy(),
            self.stddev2_pct.copy(),
            self.covariance.copy(),
            self.correlation.copy(),
            self.is_self_covariance,
            self.source_name,
        )


@dataclass
class PathContext:
    repo_root: Path
    work_dir: Path
    input_dir: Path
    output_dir: Path
    punch_dir: Path
    response_dir: Path
    spectrum_dir: Path
    covar_dir: Path
    library_dir: Path
    documentation_dir: Path
    groupr_support_dir: Path
    correlation_support_dir: Path
    self_shield_dir: Path
    lsl_output_dir: Path


@dataclass
class ActionContext:
    icons: list[int]
    title: str
    spectrum_tag: str = "default"
    response_tag: str = "default"
