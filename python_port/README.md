# MANIPULATE Python Port

This directory contains a Python reimplementation of the legacy `MANIPULATE-2020` FORTRAN workflow.

The port keeps the original control-card style input files and directory conventions, but organizes the implementation into a maintainable Python package.

## Scope

- Legacy input parsing for the `input/` job files
- Spectrum, response, GROUPR-like, LSL, and SNLCOV file handling
- Folding, weighted combination, statistical summaries, interpolation, and covariance propagation
- Output generation in the existing repo structure (`response/`, `output/`, `output/punch/`, `covar/`)

## Usage

From this directory:

```powershell
python -m manipulate_py.cli ..\input\example_response_fold
```

Or install the package in editable mode and run the console script:

```powershell
pip install -e .
manipulate-python ..\input\example_response_fold
```

If only the job name is supplied, the runner looks it up beneath the repo `input/` directory.

## Verification

Local smoke tests currently exercise these bundled jobs:

- `example_response_fold`
- `example_cross_section_covariance_verification`
- `example_resp_unc_spectrum_averaged_response`
- `example_spct_unc_spectrum_averaged_response`

Run them with:

```powershell
python -m unittest discover -s tests -v
```

## Notes

- The runner is intentionally path-aware and prefers the files in this repository over the historical hard-coded external Sandia/NJOY paths.
- Covariance and folding workflows were prioritized because they are the primary documented use cases in the bundled manual and examples.
- Some archived input decks reference upstream NJOY support files that are not fully present in this repository. The Python runner will use the local archived support files when available, and otherwise expects the missing source data to be supplied.
