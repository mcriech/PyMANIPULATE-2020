from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .covariance import (
    CovarianceData,
    apply_spectrum_normalization,
    correlation_to_covariance,
    read_endf_absolute_covariance,
    read_fcov,
    read_lsl_dosimetry,
    read_lsl_spectrum_covariance,
    read_pdf_covariance,
    read_snlcov,
    regrid_covariance,
    repair_positive_semidefinite,
    write_lsl,
    write_plot_interfaces,
    write_snlcov,
)
from .formats import (
    CurveTable,
    build_spectrum_columns,
    make_rebin_lines,
    read_energy_grid,
    read_grspin_table,
    read_groupr_table,
    read_histogram_response,
    read_list_spectrum,
    read_lsl_spectrum,
    read_mf5_spectrum,
    read_plain_midpoint_table,
    read_user_flux,
    write_pair_table,
)
from .legacy_input import InputCursor
from .models import ActionContext
from .paths import (
    OFFICIAL_GRID_BY_ICON5,
    PathResolver,
    official_grid_path,
    resolve_any,
    resolve_correlation_stem,
    resolve_group_source,
    resolve_response_like,
    resolve_self_shield,
    resolve_spectrum_like,
    resolve_work_relative,
)
from .runtime import LegacyRuntime
from .utils import average_stddev, build_spectrum_columns, fitmd, parse_fortran_float


ICODE_TO_ICON5 = {1: 0, 2: 1, 3: 2, 4: 3, 5: 5}


@dataclass
class LoadedCurve:
    table: CurveTable
    selected: np.ndarray
    file_name: str


class JobRunner:
    def __init__(self, repo_root: Path) -> None:
        self.paths = PathResolver(Path(repo_root)).build()

    def run(self, job: str | Path) -> LegacyRuntime:
        input_path = self._resolve_job(job)
        runtime = LegacyRuntime(self.paths, input_path.name)
        cursor = InputCursor.from_path(input_path)
        title = cursor.next_line(skip_blank=False).rstrip()
        runtime.log(f"MANIPULATE Python Port")
        runtime.log(f"title: {title}")
        spectrum_tag = "default"
        response_tag = "default"
        while True:
            icons = cursor.next_ints()
            if len(icons) < 40:
                icons.extend([0] * (40 - len(icons)))
            runtime.log("control parameters: " + " ".join(str(value) for value in icons[:40]))
            if icons[0] == 9:
                break
            if icons[13] == 1:
                tokens = cursor.next_tokens()
                if len(tokens) >= 2:
                    spectrum_tag, response_tag = tokens[0], tokens[1]
            ctx = ActionContext(icons=icons, title=title, spectrum_tag=spectrum_tag, response_tag=response_tag)
            self._dispatch(ctx, cursor, runtime)
        runtime.finalize()
        return runtime

    def _resolve_job(self, job: str | Path) -> Path:
        path = Path(job)
        if path.exists():
            return path.resolve()
        candidate = self.paths.input_dir / str(job)
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(job)

    def _dispatch(self, ctx: ActionContext, cursor: InputCursor, runtime: LegacyRuntime) -> None:
        icon1 = ctx.icons[0]
        if icon1 == -8:
            self._handle_covlate(ctx, cursor, runtime)
        elif icon1 == 14:
            self._handle_covariance_combine(ctx, cursor, runtime)
        elif icon1 in (3, -3):
            self._handle_groupr_extract(ctx, cursor, runtime)
        elif icon1 in (4, 5, -5):
            self._handle_curve_collection(ctx, cursor, runtime)
        elif icon1 == 6:
            self._handle_fold(ctx, cursor, runtime)
        elif abs(icon1) == 7:
            self._handle_weighted_pair(ctx, cursor, runtime)
        elif icon1 == 8:
            self._handle_difference(ctx, cursor, runtime)
        elif icon1 in (10, 11):
            self._handle_interrogation(ctx, cursor, runtime)
        elif icon1 in (12, 13):
            self._handle_grid_expansion(ctx, cursor, runtime)
        elif icon1 == 1:
            runtime.log("icon(1)=1 is a no-op in the legacy source as well")
        elif icon1 in (15, 16):
            runtime.log(f"icon(1)={icon1} PKA support is not yet implemented in this Python port")
        elif icon1 == 2:
            self._handle_curve_collection(ctx, cursor, runtime)
        else:
            raise NotImplementedError(f"icon(1)={icon1} is not implemented")

    def _read_table(self, ctx: ActionContext, file_name: str, imode: int, select_material: int, *, is_response: bool) -> CurveTable:
        icon5 = ctx.icons[4]
        inhibit_renorm = ctx.icons[5] == 1
        reverse_data = ctx.icons[11] == 1
        material_index = 1 if select_material <= 0 else abs(select_material)
        if imode in (0, 12, 23):
            if material_index > 1:
                path = resolve_group_source(self.paths, file_name)
                table = read_groupr_table(path, material_index)
            else:
                path = resolve_response_like(self.paths, file_name) if is_response else resolve_spectrum_like(self.paths, file_name)
                table = read_grspin_table(path, reverse_output=False)
            if icon5 in OFFICIAL_GRID_BY_ICON5:
                grid = read_energy_grid(official_grid_path(self.paths, icon5))
                if grid.size == table.bounds_mev.size:
                    table = CurveTable(grid, table.values, table.title, list(table.comments))
            values = table.column(1)
            if imode in (12, 23):
                widths_ev = np.abs(np.diff(table.bounds_mev)) * 1.0e6
                values = np.abs(values * widths_ev)
                total = values.sum()
                if total <= 0.0:
                    total = 1.0
                values = values / total
                if imode == 23:
                    columns = build_spectrum_columns(values, table.bounds_mev)
                    table = CurveTable(table.bounds_mev, columns, table.title, list(table.comments))
                else:
                    table = CurveTable(table.bounds_mev, values[:, np.newaxis], table.title, list(table.comments))
            elif not is_response and not inhibit_renorm:
                total = values.sum()
                if total <= 0.0:
                    total = 1.0
                values = values / total
                table = CurveTable(table.bounds_mev, values[:, np.newaxis], table.title, list(table.comments))
        elif imode in (1, 2):
            path = resolve_spectrum_like(self.paths, file_name)
            table = read_list_spectrum(path)
        elif imode == 4:
            path = resolve_any(self.paths, file_name)
            table = read_plain_midpoint_table(path, nenergy_first=True)
        elif imode == -4:
            path = resolve_any(self.paths, file_name)
            table = read_plain_midpoint_table(path, nenergy_first=True).reversed()
        elif imode == 9:
            path = resolve_spectrum_like(self.paths, file_name if file_name.endswith(".flux") else f"{file_name}.flux")
            table = read_user_flux(path)
        elif imode == 10:
            path = resolve_response_like(self.paths, file_name)
            table = read_histogram_response(path, official_grid_path(self.paths, icon5), reverse_storage=True)
        elif imode == -10:
            path = resolve_response_like(self.paths, file_name)
            table = read_histogram_response(path, official_grid_path(self.paths, icon5), reverse_storage=False)
        elif imode == 11:
            path = resolve_response_like(self.paths, file_name)
            grid = resolve_response_like(self.paths, f"{file_name}.nrg")
            table = read_histogram_response(path, grid, reverse_storage=False)
        elif imode == 24:
            path = resolve_spectrum_like(self.paths, file_name)
            table = read_mf5_spectrum(path)
        elif imode == 25:
            path = resolve_spectrum_like(self.paths, file_name)
            table = read_lsl_spectrum(path)
        else:
            raise NotImplementedError(f"imode={imode} is not implemented")
        if reverse_data:
            table = table.reversed()
        return table

    def _load_curve(self, ctx: ActionContext, file_name: str, imode: int, select_material: int, *, is_response: bool) -> LoadedCurve:
        table = self._read_table(ctx, file_name, imode, select_material, is_response=is_response)
        index = 1 if select_material <= 0 else min(abs(select_material), table.values.shape[1])
        return LoadedCurve(table=table, selected=table.column(index), file_name=file_name)

    def _align_to(self, reference: CurveTable, candidate: CurveTable) -> CurveTable:
        if candidate.nenergy != reference.nenergy:
            raise ValueError("energy grid sizes do not match")
        ref_mid = reference.midpoints_mev
        cand_mid = candidate.midpoints_mev
        if np.allclose(ref_mid, cand_mid, rtol=1.0e-3, atol=1.0e-12):
            return candidate
        reversed_candidate = candidate.reversed()
        if np.allclose(ref_mid, reversed_candidate.midpoints_mev, rtol=1.0e-3, atol=1.0e-12):
            return reversed_candidate
        raise ValueError("energy grids do not match")

    def _write_output_curve(self, outfile: str, mids_mev: np.ndarray, values: np.ndarray) -> Path:
        path = (self.paths.response_dir / outfile).resolve()
        write_pair_table(path, mids_mev, values, reverse=False, scale=1.0)
        return path

    def _handle_groupr_extract(self, ctx: ActionContext, cursor: InputCursor, runtime: LegacyRuntime) -> None:
        file_name, irxn, outfile, scale = cursor.next_tokens()[:4]
        reaction = int(irxn)
        scale_value = parse_fortran_float(scale)
        if ctx.icons[0] == 3:
            table = read_groupr_table(resolve_group_source(self.paths, file_name), reaction)
        else:
            table = self._read_table(ctx, file_name, 0, 1, is_response=True)
        if ctx.icons[2] == 1:
            out_path = self._write_output_curve(f"{outfile}.{reaction}", table.midpoints_mev, table.column(1) * scale_value)
            runtime.log(f"wrote {out_path}")
            runtime.pun2(ctx.title[:60])
            runtime.pun2(f"file = {outfile}")
            runtime.pun2(f"{1:6d} {table.nenergy:6d}")
            for value in table.column(1) * scale_value:
                runtime.pun2(f"{value:14.7E}")
        runtime.ext(f"EXTRACT RSP {outfile} {reaction} {float(np.sum(table.column(1) * scale_value)):.7E}")

    def _handle_curve_collection(self, ctx: ActionContext, cursor: InputCursor, runtime: LegacyRuntime) -> None:
        header = cursor.next_tokens()
        number_of_files = int(header[0])
        scale = parse_fortran_float(header[1])
        outfile = header[2]
        curves: list[LoadedCurve] = []
        for _ in range(number_of_files):
            tokens = cursor.next_tokens()
            if ctx.icons[0] == 4:
                file_name, imode, select_material = tokens[:3]
                fraction = 1.0
            else:
                file_name, imode, select_material, fraction = tokens[:4]
            loaded = self._load_curve(ctx, file_name, int(imode), int(select_material), is_response=False)
            loaded = LoadedCurve(loaded.table, loaded.selected * parse_fortran_float(str(fraction)), loaded.file_name)
            if ctx.icons[9] == 1:
                cursor.next_tokens()
            curves.append(loaded)
        if ctx.icons[0] == -5:
            self._write_statistics(curves, outfile, runtime)
            return
        if not curves:
            return
        reference = curves[0].table
        aligned = [LoadedCurve(self._align_to(reference, curve.table), curve.selected, curve.file_name) for curve in curves]
        if ctx.icons[0] == 5:
            values = np.sum([curve.selected for curve in aligned], axis=0)
            if ctx.icons[17] == 1:
                max_element = np.max(np.abs(np.vstack([curve.selected for curve in aligned])), axis=0)
                mask = np.divide(np.abs(values), max_element, out=np.zeros_like(values), where=max_element != 0.0) < 1.0e-4
                values[mask] = 0.0
        else:
            values = aligned[0].selected.copy()
        out_path = self._write_output_curve(outfile, reference.midpoints_mev, values * scale)
        runtime.log(f"wrote {out_path}")
        runtime.set_rebin(make_rebin_lines(CurveTable(reference.bounds_mev, values[:, np.newaxis], outfile)))

    def _write_statistics(self, curves: list[LoadedCurve], outfile: str, runtime: LegacyRuntime) -> None:
        reference = curves[0].table
        aligned = [LoadedCurve(self._align_to(reference, curve.table), curve.selected, curve.file_name) for curve in curves]
        matrix = np.vstack([curve.selected for curve in aligned])
        avg = matrix.mean(axis=0)
        variance = matrix.var(axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros(reference.nenergy, dtype=float)
        pct_variance = np.sqrt(np.clip(variance, 0.0, None)) * 100.0
        pct_std = np.divide(pct_variance, avg, out=np.zeros_like(avg), where=avg != 0.0)
        avg_path = (self.paths.response_dir / f"{outfile}.avg").resolve()
        pct_path = (self.paths.response_dir / f"{outfile}.pctstd").resolve()
        write_pair_table(avg_path, reference.midpoints_mev, avg)
        write_pair_table(pct_path, reference.midpoints_mev, pct_std)
        runtime.log(f"wrote {avg_path}")
        runtime.log(f"wrote {pct_path}")

    def _handle_fold(self, ctx: ActionContext, cursor: InputCursor, runtime: LegacyRuntime) -> None:
        source_tokens = cursor.next_tokens()
        response_tokens = cursor.next_tokens()
        source = self._load_curve(ctx, source_tokens[0], int(source_tokens[1]), int(source_tokens[2]), is_response=False)
        response = self._load_curve(ctx, response_tokens[0], int(response_tokens[1]), int(response_tokens[2]), is_response=True)
        response_table = self._align_to(source.table, response.table)
        response_values = response_table.column(1)
        if ctx.icons[12] == 1:
            alpha = parse_fortran_float(cursor.next_tokens()[0])
            attenuation_tokens = cursor.next_tokens()
            attenuation = self._load_curve(ctx, attenuation_tokens[0], int(attenuation_tokens[1]), int(attenuation_tokens[2]), is_response=False)
            attenuation_values = self._align_to(source.table, attenuation.table).column(1)
            response_values = response_values * np.exp(-alpha * attenuation_values)
        elif ctx.icons[12] == 2:
            corr_name = cursor.next_tokens()[0]
            response_values = self._apply_self_shield(response_values, source.table.nenergy, corr_name)
        increment = source.selected * response_values
        total = float(np.sum(increment))
        above_10kev = source.table.midpoints_mev > 1.0e-2
        above_3mev = source.table.midpoints_mev > 3.0
        sum2 = float(np.sum(increment[above_10kev]))
        sum3 = float(np.sum(increment[above_3mev]))
        flu1 = float(np.sum(source.selected[above_10kev]))
        flu2 = float(np.sum(response_values[above_10kev]))
        tot1 = float(np.sum(source.selected))
        tot2 = float(np.sum(response_values))
        cumulative = np.cumsum(increment)
        if total != 0.0:
            cumulative = cumulative / total
        e05 = fitmd(0.05, cumulative, source.table.midpoints_mev, 2)
        e10 = fitmd(0.10, cumulative, source.table.midpoints_mev, 2)
        e25 = fitmd(0.25, cumulative, source.table.midpoints_mev, 2)
        e50 = fitmd(0.50, cumulative, source.table.midpoints_mev, 2)
        e75 = fitmd(0.75, cumulative, source.table.midpoints_mev, 2)
        e90 = fitmd(0.90, cumulative, source.table.midpoints_mev, 2)
        e95 = fitmd(0.95, cumulative, source.table.midpoints_mev, 2)
        runtime.log(f"folded response = {total:.7E}")
        runtime.ext(
            f"EXTRACT ENG {ctx.spectrum_tag} {ctx.response_tag} {total:.7E} {e05:.7E} {e10:.7E} "
            f"{e25:.7E} {e50:.7E} {e75:.7E} {e90:.7E} {e95:.7E}"
        )
        runtime.log(f">10 keV fold = {sum2:.7E}")
        runtime.log(f">3 MeV fold = {sum3:.7E}")
        runtime.log(f"source integral = {tot1:.7E}")
        runtime.log(f"response integral = {tot2:.7E}")
        if flu1 != 0.0:
            runtime.log(f"normalized >10 keV fold = {sum2 / flu1:.7E}")
        runtime.set_rebin(make_rebin_lines(CurveTable(source.table.bounds_mev, source.selected[:, np.newaxis], source.file_name)))

    def _apply_self_shield(self, response_values: np.ndarray, nenergy: int, correction_name: str) -> np.ndarray:
        if correction_name == "null#-void-bare":
            return response_values.copy()
        path = resolve_self_shield(self.paths, correction_name)
        lines = path.read_text().splitlines()
        factors = np.asarray([parse_fortran_float(tok) for tok in " ".join(lines[1:]).split()], dtype=float)
        corrected = response_values.copy()
        limit = min(factors.size, corrected.size)
        corrected[:limit] = corrected[:limit] * factors[:limit]
        return corrected

    def _handle_weighted_pair(self, ctx: ActionContext, cursor: InputCursor, runtime: LegacyRuntime) -> None:
        header = cursor.next_tokens()
        scale = parse_fortran_float(header[1])
        outfile = header[2]
        first = cursor.next_tokens()
        second = cursor.next_tokens()
        curve1 = self._load_curve(ctx, first[0], int(first[1]), int(first[2]), is_response=True)
        curve2 = self._load_curve(ctx, second[0], int(second[1]), int(second[2]), is_response=True)
        second_table = self._align_to(curve1.table, curve2.table)
        rhs = second_table.column(1)
        if ctx.icons[0] == 7:
            values = curve1.selected * rhs
        else:
            values = np.divide(curve1.selected, rhs, out=np.full_like(curve1.selected, -9.999e-30), where=rhs != 0.0)
        out_path = self._write_output_curve(outfile, curve1.table.midpoints_mev, values * scale)
        runtime.log(f"wrote {out_path}")

    def _handle_difference(self, ctx: ActionContext, cursor: InputCursor, runtime: LegacyRuntime) -> None:
        first = cursor.next_tokens()
        second = cursor.next_tokens()
        curve1 = self._load_curve(ctx, first[0], int(first[1]), int(first[2]), is_response=False)
        curve2 = self._load_curve(ctx, second[0], int(second[1]), int(second[2]), is_response=False)
        table2 = self._align_to(curve1.table, curve2.table)
        rhs = table2.column(1)
        denom = np.where(np.abs(curve1.selected) > 1.0e-33, curve1.selected, rhs)
        denom = np.where(np.abs(denom) > 1.0e-33, denom, 1.0)
        percent = (curve1.selected - rhs) / (denom / 100.0)
        runtime.log("file difference")
        runtime.log(f"max percent diff = {np.max(np.abs(percent)):.7E}")

    def _handle_interrogation(self, ctx: ActionContext, cursor: InputCursor, runtime: LegacyRuntime) -> None:
        file_name = cursor.next_tokens()[0]
        table = self._read_table(ctx, file_name, 11 if ctx.icons[0] == 11 else 10, 1, is_response=True)
        method, direction, ipoint, value = cursor.next_tokens()[:4]
        point = int(ipoint)
        point = 1 if point < 1 or point > table.values.shape[1] else point
        xdata = table.midpoints_mev if int(direction) == 1 else table.column(point)
        ydata = table.column(point) if int(direction) == 1 else table.midpoints_mev
        result = fitmd(parse_fortran_float(value), xdata, ydata, int(method))
        runtime.log(f"interrogation result for {file_name} = {result:.7E}")

    def _handle_grid_expansion(self, ctx: ActionContext, cursor: InputCursor, runtime: LegacyRuntime) -> None:
        if ctx.icons[0] == 13:
            source_table = None
        else:
            file_name = cursor.next_tokens()[0]
            source_table = self._read_table(ctx, file_name, 11, 1, is_response=True)
        method_tokens = cursor.next_tokens()
        method = int(method_tokens[0])
        direction = int(method_tokens[1]) if len(method_tokens) > 1 else 1
        target_grid = read_energy_grid(official_grid_path(self.paths, ctx.icons[4]))
        output_path = self.paths.work_dir / "file.interpolation"
        lines = ["    640"]
        mids = 0.5 * (target_grid[:-1] + target_grid[1:])
        for mid in mids:
            if ctx.icons[0] == 13:
                x = mid * 1.0e3
                result = 0.872670 + -0.187469 * np.log10(x) + 1.237178e-7 * (x**2) * np.log10(x) + -0.060753 * (np.log10(x) ** 2)
                if x <= 1.0e-1:
                    result = 1.0
                if x >= 500.0:
                    result = 0.01
            else:
                xdata = source_table.midpoints_mev if direction == 1 else source_table.column(1)
                ydata = source_table.column(1) if direction == 1 else source_table.midpoints_mev
                result = fitmd(mid, xdata, ydata, method)
            lines.append(f"{mid:14.7E}      {result:14.7E}")
        output_path.write_text("\n".join(lines) + "\n")
        runtime.log(f"wrote {output_path}")

    def _load_covariance(self, fmt_code: int, stem: str, icode: int, *, spectrum_mode: bool) -> CovarianceData:
        if fmt_code == 2:
            try:
                cov = read_lsl_dosimetry(resolve_correlation_stem(self.paths, stem, ".lsl"))
            except FileNotFoundError:
                cov = read_snlcov(resolve_correlation_stem(self.paths, stem, ".snlcov"))
        elif fmt_code == 3:
            cov = read_snlcov(resolve_correlation_stem(self.paths, stem, ".snlcov"))
        elif fmt_code == 5:
            try:
                cov = read_lsl_spectrum_covariance(resolve_correlation_stem(self.paths, stem, ".lsl"))
            except FileNotFoundError:
                cov = read_snlcov(resolve_correlation_stem(self.paths, stem, ".snlcov"))
        elif fmt_code == 6:
            cov = read_endf_absolute_covariance(resolve_correlation_stem(self.paths, stem, ".endf"))
        elif fmt_code == 8:
            cov = read_fcov(resolve_correlation_stem(self.paths, stem, ".lib"))
        else:
            raise NotImplementedError(f"covariance format {fmt_code} is not implemented")
        cov = repair_positive_semidefinite(cov, normalize_spectrum=spectrum_mode)
        if icode != 0:
            target_grid = read_energy_grid(official_grid_path(self.paths, ICODE_TO_ICON5[icode]))
            cov = regrid_covariance(cov, target_grid, is_spectrum=spectrum_mode)
        return cov

    def _covariance_fold(self, fixed_curve: LoadedCurve, covdata: CovarianceData) -> tuple[float, float, float]:
        aligned_table = self._align_to(CurveTable(covdata.bounds_mev, covdata.values1[:, np.newaxis], covdata.source_name), fixed_curve.table)
        if np.allclose(aligned_table.midpoints_mev, fixed_curve.table.midpoints_mev, rtol=1.0e-6, atol=1.0e-12):
            fixed = fixed_curve.selected
        else:
            fixed = fixed_curve.selected[::-1]
        values = covdata.values1
        sigma = covdata.stddev1_pct * 0.01
        total = float(np.sum(fixed * values))
        weighted = fixed * values * sigma
        variance = float(weighted @ covdata.correlation @ weighted)
        variance = max(variance, 0.0)
        uncertainty_pct = np.sqrt(variance) / total * 100.0 if total != 0.0 else 0.0
        avg_std = average_stddev(values, covdata.stddev1_pct, fixed)
        return total, uncertainty_pct, avg_std

    def _handle_covlate(self, ctx: ActionContext, cursor: InputCursor, runtime: LegacyRuntime) -> None:
        icov = cursor.next_ints()
        while len(icov) < 3:
            icov.append(0)
        mode = icov[0]
        if mode == 1:
            cov_fmt1, fmt1, icode = cursor.next_tokens()[:3]
            cov_fmt2, fmt2, imode, select = cursor.next_tokens()[:4]
            covdata = self._load_covariance(int(cov_fmt1), fmt1, int(icode), spectrum_mode=False)
            if icov[1] == 1:
                override_tokens = cursor.next_tokens()
                override = self._load_curve(ctx, override_tokens[0], int(override_tokens[1]), int(override_tokens[2]), is_response=True)
                aligned_override = self._align_to(CurveTable(covdata.bounds_mev, covdata.values1[:, np.newaxis], covdata.source_name), override.table)
                override_values = override.selected if np.allclose(
                    aligned_override.midpoints_mev, override.table.midpoints_mev, rtol=1.0e-6, atol=1.0e-12
                ) else override.selected[::-1]
                if ctx.icons[12] == 2:
                    corr_name = cursor.next_tokens()[0]
                    override_values = self._apply_self_shield(override_values, override_values.size, corr_name)
                covdata.values1 = override_values.copy()
                covdata.values2 = override_values.copy()
                covdata.covariance = correlation_to_covariance(
                    covdata.values1, covdata.stddev1_pct, covdata.values2, covdata.stddev2_pct, covdata.correlation
                )
            fixed = self._load_curve(ctx, fmt2, int(imode), int(select), is_response=False)
            total, unc, avg = self._covariance_fold(fixed, covdata)
            runtime.log(f"covariance fold = {total:.7E}, uncertainty = {unc:.7E} %, avg std = {avg:.7E} %")
            runtime.ext(f"EXTRACT COV1 {ctx.spectrum_tag} {ctx.response_tag} {total:.7E} {unc:.7E} {avg:.7E}")
        elif mode == 4:
            cov_fmt2, fmt2, imode, select = cursor.next_tokens()[:4]
            cov_fmt1, fmt1, icode = cursor.next_tokens()[:3]
            covdata = self._load_covariance(int(cov_fmt1), fmt1, int(icode), spectrum_mode=True)
            fixed = self._load_curve(ctx, fmt2, int(imode), int(select), is_response=True)
            if ctx.icons[12] == 2:
                corr_name = cursor.next_tokens()[0]
                fixed = LoadedCurve(
                    fixed.table,
                    self._apply_self_shield(fixed.selected, fixed.table.nenergy, corr_name),
                    fixed.file_name,
                )
            total, unc, avg = self._covariance_fold(fixed, covdata)
            runtime.log(f"spectrum covariance fold = {total:.7E}, uncertainty = {unc:.7E} %, avg std = {avg:.7E} %")
            runtime.ext(f"EXTRACT COV9 {ctx.spectrum_tag} {ctx.response_tag} {total:.7E} {unc:.7E} {avg:.7E}")
        elif mode == 2:
            cov_fmt1, fmt1, icode = cursor.next_tokens()[:3]
            spectrum_mode = int(cov_fmt1) in (5, 6, 7, 8)
            covdata = self._load_covariance(int(cov_fmt1), fmt1, int(icode), spectrum_mode=spectrum_mode)
            stem = resolve_work_relative(self.paths, fmt1)
            write_plot_interfaces(stem, covdata)
            runtime.log(f"wrote covariance plot interfaces for {fmt1}")
        elif mode == 3:
            runtime.log("icov(1)=3 covariance scaling workflow is not yet implemented")
        elif mode == 5:
            runtime.log("icov(1)=5 covariance combination placeholder retained from legacy source")
        else:
            raise NotImplementedError(f"icov(1)={mode} is not implemented")

    def _handle_covariance_combine(self, ctx: ActionContext, cursor: InputCursor, runtime: LegacyRuntime) -> None:
        icov = cursor.next_ints()
        header = cursor.next_tokens()
        number_of_files = int(header[0])
        overall_scale = parse_fortran_float(header[1])
        outfile = header[2]
        covariances: list[tuple[float, CovarianceData]] = []
        for _ in range(number_of_files):
            fmt_code, stem, icode, weight = cursor.next_tokens()[:4]
            cov = self._load_covariance(int(fmt_code), stem, int(icode), spectrum_mode=int(fmt_code) == 5)
            covariances.append((parse_fortran_float(weight), cov))
        base = covariances[0][1]
        cov_sum = np.zeros_like(base.covariance)
        for weight, cov in covariances:
            if not np.allclose(cov.bounds_mev, base.bounds_mev):
                raise ValueError("covariance grids do not match for combination")
            cov_sum += (weight * overall_scale) ** 2 * cov.covariance
        values = base.values1.copy()
        unc = np.sqrt(np.clip(np.diag(cov_sum), 0.0, None))
        std = np.divide(unc, values, out=np.zeros_like(values), where=values != 0.0) * 100.0
        denom = np.outer(unc, unc)
        corr = np.divide(cov_sum, denom, out=np.zeros_like(cov_sum), where=denom != 0.0)
        np.fill_diagonal(corr, np.where(values != 0.0, 1.0, 0.0))
        combined = CovarianceData(base.bounds_mev, values, std, values.copy(), std.copy(), cov_sum, corr, True, outfile)
        out_path = resolve_work_relative(self.paths, outfile)
        write_lsl(out_path, combined, spectrum_mode=False)
        write_plot_interfaces(out_path, combined)
        runtime.log(f"wrote {out_path}")
