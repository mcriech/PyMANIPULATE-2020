from __future__ import annotations

import difflib
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_PORT_ROOT = REPO_ROOT / "python_port"
VALIDATION_ROOT = PYTHON_PORT_ROOT / "preliminary_validation"
TRACKED_DIRS = ["output", "response", "covar"]
REGRESSION_JOBS = [
    "example_NJOY_groupr_convert",
    "example_cross_section_covariance_verification",
    "example_NJOY_groupr_response_convert",
    "example_resp_unc_spectrum_averaged_response",
    "example_NJOY_groupr_spectrum_convert",
    "example_response_fold",
    "example_NJOY_groupr_xsec_convert",
    "example_spct_unc_spectrum_averaged_response",
    "example_NJOY_response_combination",
    "example_composite_uncertainty",
]
TEXT_SUFFIXES = {
    "",
    ".261",
    ".corplt",
    ".csv",
    ".dat",
    ".ext",
    ".inp",
    ".jnb",
    ".json",
    ".log",
    ".lsl",
    ".nrg",
    ".out",
    ".plt",
    ".prl",
    ".sg33",
    ".sg44",
    ".sg55",
    ".std_pct_plt",
    ".txt",
}


def rel_repo_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_state() -> dict[str, dict[str, int | str]]:
    state: dict[str, dict[str, int | str]] = {}
    for folder_name in TRACKED_DIRS:
        root = REPO_ROOT / folder_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            state[rel_repo_path(path)] = {
                "sha256": hash_file(path),
                "size": path.stat().st_size,
            }
    return state


def compare_states(
    before: dict[str, dict[str, int | str]],
    after: dict[str, dict[str, int | str]],
) -> dict[str, list[str]]:
    before_keys = set(before)
    after_keys = set(after)
    added = sorted(after_keys - before_keys)
    removed = sorted(before_keys - after_keys)
    modified = sorted(
        path for path in before_keys & after_keys if before[path]["sha256"] != after[path]["sha256"]
    )
    return {"added": added, "removed": removed, "modified": modified}


def copy_baseline(snapshot_root: Path) -> None:
    for folder_name in TRACKED_DIRS:
        source = REPO_ROOT / folder_name
        destination = snapshot_root / folder_name
        if source.exists():
            shutil.copytree(source, destination)


def is_probably_text(path: Path) -> bool:
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    try:
        sample = path.read_bytes()[:4096]
    except OSError:
        return False
    return b"\x00" not in sample


def baseline_status(rel_path: str, baseline_root: Path, current_state: dict[str, dict[str, int | str]]) -> str:
    baseline_path = baseline_root / Path(rel_path)
    current_path = REPO_ROOT / Path(rel_path)
    if rel_path not in current_state:
        return "removed_from_workspace"
    if not baseline_path.exists():
        return "new_file"
    if hash_file(baseline_path) == hash_file(current_path):
        return "matches_prerun_reference"
    return "differs_from_prerun_reference"


def write_diff(rel_path: str, baseline_root: Path, diff_root: Path) -> str | None:
    baseline_path = baseline_root / Path(rel_path)
    current_path = REPO_ROOT / Path(rel_path)
    if not baseline_path.exists() or not current_path.exists():
        return None
    if not is_probably_text(baseline_path) or not is_probably_text(current_path):
        return None
    try:
        baseline_lines = baseline_path.read_text().splitlines()
        current_lines = current_path.read_text().splitlines()
    except UnicodeDecodeError:
        return None
    diff_lines = list(
        difflib.unified_diff(
            baseline_lines,
            current_lines,
            fromfile=f"baseline/{rel_path}",
            tofile=f"current/{rel_path}",
            lineterm="",
        )
    )
    if not diff_lines:
        return None
    diff_path = diff_root / f"{rel_path}.diff"
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    diff_path.write_text("\n".join(diff_lines) + "\n")
    return rel_repo_path(diff_path)


def copy_current_file(rel_path: str, destination_root: Path) -> str | None:
    source = REPO_ROOT / Path(rel_path)
    if not source.exists():
        return None
    destination = destination_root / rel_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return rel_repo_path(destination)


def write_job_log(job_log_path: Path, lines: list[str]) -> None:
    job_log_path.parent.mkdir(parents=True, exist_ok=True)
    job_log_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = VALIDATION_ROOT / f"run_{run_stamp}"
    logs_root = run_root / "logs"
    diffs_root = run_root / "diffs"
    files_root = run_root / "files"
    run_root.mkdir(parents=True, exist_ok=False)

    sys.path.insert(0, str(PYTHON_PORT_ROOT / "src"))
    from manipulate_py.runner import JobRunner

    baseline_parent = Path(tempfile.mkdtemp(prefix="manipulate_validation_"))
    baseline_root = baseline_parent / "baseline"

    overall_started = time.time()
    baseline_state = scan_state()
    copy_baseline(baseline_root)

    results: dict[str, object] = {
        "run_started_utc_epoch": overall_started,
        "repo_root": str(REPO_ROOT),
        "tracked_dirs": TRACKED_DIRS,
        "regression_jobs": [],
    }

    try:
        runner = JobRunner(REPO_ROOT)
        current_state = baseline_state
        for job_name in REGRESSION_JOBS:
            before_state = current_state
            started = time.perf_counter()
            succeeded = False
            output_path = None
            ext_path = None
            runtime_error = None
            traceback_text = None
            try:
                runtime = runner.run(job_name)
                output_path = rel_repo_path(runtime.paths.output_dir / f"{runtime.job_name}.out")
                ext_path = rel_repo_path(runtime.paths.output_dir / f"{runtime.job_name}.ext")
                succeeded = True
            except Exception as exc:  # pragma: no cover - validation harness
                runtime_error = str(exc)
                traceback_text = traceback.format_exc()
            elapsed_seconds = time.perf_counter() - started
            after_state = scan_state()
            delta = compare_states(before_state, after_state)

            changed_paths = sorted(delta["added"] + delta["modified"] + delta["removed"])
            changed_files: list[dict[str, str | None]] = []
            for rel_path in changed_paths:
                comparison = baseline_status(rel_path, baseline_root, after_state)
                copied_path = copy_current_file(rel_path, files_root / job_name)
                diff_path = None
                if comparison == "differs_from_prerun_reference":
                    diff_path = write_diff(rel_path, baseline_root, diffs_root / job_name)
                changed_files.append(
                    {
                        "path": rel_path,
                        "comparison_to_prerun_reference": comparison,
                        "copied_artifact": copied_path,
                        "diff_artifact": diff_path,
                    }
                )

            job_log_lines = [
                f"job: {job_name}",
                f"status: {'passed' if succeeded else 'failed'}",
                f"elapsed_seconds: {elapsed_seconds:.3f}",
                f"output_path: {output_path}",
                f"ext_path: {ext_path}",
                f"changed_file_count: {len(changed_files)}",
            ]
            if runtime_error:
                job_log_lines.append(f"error: {runtime_error}")
            if traceback_text:
                job_log_lines.extend(["traceback:", traceback_text.rstrip()])
            for file_result in changed_files:
                job_log_lines.append(
                    f"changed: {file_result['path']} [{file_result['comparison_to_prerun_reference']}]"
                )

            write_job_log(logs_root / f"{job_name}.log", job_log_lines)
            results["regression_jobs"].append(
                {
                    "job_name": job_name,
                    "status": "passed" if succeeded else "failed",
                    "elapsed_seconds": elapsed_seconds,
                    "output_path": output_path,
                    "ext_path": ext_path,
                    "error": runtime_error,
                    "traceback": traceback_text,
                    "delta": delta,
                    "changed_files": changed_files,
                }
            )
            current_state = after_state

        smoke_started = time.perf_counter()
        smoke_process = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=PYTHON_PORT_ROOT,
            capture_output=True,
            text=True,
        )
        smoke_elapsed_seconds = time.perf_counter() - smoke_started
        smoke_log = (
            f"command: {sys.executable} -m unittest discover -s tests -v\n"
            f"exit_code: {smoke_process.returncode}\n"
            f"elapsed_seconds: {smoke_elapsed_seconds:.3f}\n\n"
            "[stdout]\n"
            f"{smoke_process.stdout}\n"
            "[stderr]\n"
            f"{smoke_process.stderr}\n"
        )
        (logs_root / "smoke_suite.log").write_text(smoke_log)

        final_state = scan_state()
        final_delta = compare_states(baseline_state, final_state)
        final_differences: list[dict[str, str | None]] = []
        for rel_path in sorted(final_delta["added"] + final_delta["modified"] + final_delta["removed"]):
            comparison = baseline_status(rel_path, baseline_root, final_state)
            diff_path = None
            if comparison == "differs_from_prerun_reference":
                diff_path = write_diff(rel_path, baseline_root, diffs_root / "final_state")
            final_differences.append(
                {
                    "path": rel_path,
                    "comparison_to_prerun_reference": comparison,
                    "diff_artifact": diff_path,
                }
            )

        results["smoke_suite"] = {
            "status": "passed" if smoke_process.returncode == 0 else "failed",
            "exit_code": smoke_process.returncode,
            "elapsed_seconds": smoke_elapsed_seconds,
            "log_path": rel_repo_path(logs_root / "smoke_suite.log"),
        }
        results["final_workspace_delta"] = {
            "delta": final_delta,
            "files": final_differences,
        }
        results["run_completed_utc_epoch"] = time.time()
        results["run_root"] = rel_repo_path(run_root)
        results["baseline_manifest"] = baseline_state

        (run_root / "validation_results.json").write_text(json.dumps(results, indent=2))
    finally:
        shutil.rmtree(baseline_parent, ignore_errors=True)

    print(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
