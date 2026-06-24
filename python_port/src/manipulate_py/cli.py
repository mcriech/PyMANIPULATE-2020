from __future__ import annotations

import argparse
from pathlib import Path

from .runner import JobRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Python port of MANIPULATE-2020")
    parser.add_argument("job", help="Job name from input/ or a path to a legacy input file")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3], help="Repository root")
    args = parser.parse_args()
    runner = JobRunner(args.repo_root)
    runtime = runner.run(args.job)
    print(f"Wrote {runtime.paths.output_dir / (runtime.job_name + '.out')}")
