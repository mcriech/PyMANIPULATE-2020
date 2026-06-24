# MANIPULATE-2020

This repository contains the legacy `MANIPULATE-2020` workflow assets and a Python reimplementation in [python_port](python_port/).

## Repository Layout

- `python_port/`: Python package and tests for the port
- `input/`: control-card style example job decks
- `response/`, `output/`, `output/punch/`, `covar/`: runtime data used and produced by the workflow
- `documentation/`: bundled MANIPULATE reference material

## Python Port

The Python port preserves the original job-file conventions while organizing the implementation as a maintainable package.

Current scope includes:

- parsing legacy `input/` job files
- handling spectrum, response, GROUPR-like, LSL, and SNLCOV files
- folding, weighted combination, statistical summaries, interpolation, and covariance propagation
- writing outputs back into the existing repository directory structure

## How To Run

From `python_port/`:

```powershell
python -m manipulate_py.cli ..\input\example_response_fold
```

You can also install the package in editable mode and use the console script:

```powershell
pip install -e .
manipulate-python ..\input\example_response_fold
```

If you pass only a job name such as `example_response_fold`, the runner looks for that file under the repository `input/` directory.

## Verification

The bundled smoke tests currently cover these example jobs:

- `example_response_fold`
- `example_cross_section_covariance_verification`
- `example_resp_unc_spectrum_averaged_response`
- `example_spct_unc_spectrum_averaged_response`

Run them from `python_port/` with:

```powershell
python -m unittest discover -s tests -v
```

## Notes

- The Python runner prefers files in this repository over the historical hard-coded Sandia/NJOY paths.
- The documented covariance and folding workflows were prioritized in the port.
- Some archived job decks still depend on external NJOY support files that are not fully bundled here.
- Additional port-specific details are in [python_port/README.md](python_port/README.md).
