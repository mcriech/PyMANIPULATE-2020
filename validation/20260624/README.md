## Validation Bundle

This folder packages the June 24, 2026 Python-port validation run into a single reviewable bundle.

### What is included

- `job_inputs/`: the 10 regression job decks used by `python_port/tools/run_preliminary_validation.py`
- `workspace_snapshot/output/`: the post-run `output/` directory snapshot
- `workspace_snapshot/response/`: the post-run `response/` directory snapshot
- `workspace_snapshot/covar/`: the covariance inputs used by the jobs
- `validation_run/`: the generated validation manifest, logs, copied artifacts, and diffs from `run_20260624_165628`
- `documentation_reference/`: the bundled MANIPULATE PDF referenced during validation

### Commands run

```powershell
cd python_port
python -m unittest discover -s tests -v
python tools\run_preliminary_validation.py
```

### Result summary

- Smoke suite status: passed
- Smoke tests passed: 4 of 4
- Regression harness status: passed
- Regression jobs passed: 10 of 10

The smoke suite covered:

- `example_cross_section_covariance_verification`
- `example_response_fold`
- `example_resp_unc_spectrum_averaged_response`
- `example_spct_unc_spectrum_averaged_response`

The full regression harness covered:

- `example_NJOY_groupr_convert`
- `example_cross_section_covariance_verification`
- `example_NJOY_groupr_response_convert`
- `example_resp_unc_spectrum_averaged_response`
- `example_NJOY_groupr_spectrum_convert`
- `example_response_fold`
- `example_NJOY_groupr_xsec_convert`
- `example_spct_unc_spectrum_averaged_response`
- `example_NJOY_response_combination`
- `example_composite_uncertainty`

### Notes

- The raw validation manifest is in `validation_run/validation_results.json`.
- The smoke log is in `validation_run/logs/smoke_suite.log`.
- The workspace snapshot is intentionally copied after the run so reviewers can inspect the actual generated files without recomputing them.
