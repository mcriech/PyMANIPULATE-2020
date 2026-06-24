from __future__ import annotations

import sys
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "python_port" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from manipulate_py.runner import JobRunner  # noqa: E402


class ExampleSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = JobRunner(REPO_ROOT)

    def test_response_fold_job(self) -> None:
        runtime = self.runner.run("example_response_fold")
        ext_path = runtime.paths.output_dir / "example_response_fold.ext"
        text = ext_path.read_text()
        self.assertIn("EXTRACT ENG", text)
        self.assertIn("Cf252(sf)", text)

    def test_cross_section_covariance_job(self) -> None:
        runtime = self.runner.run("example_cross_section_covariance_verification")
        out_path = runtime.paths.output_dir / "example_Ag109g_IRDFF-II_cor_plot_interface.corplt"
        self.assertTrue(out_path.exists())

    def test_response_uncertainty_job(self) -> None:
        runtime = self.runner.run("example_resp_unc_spectrum_averaged_response")
        ext_path = runtime.paths.output_dir / "example_resp_unc_spectrum_averaged_response.ext"
        text = ext_path.read_text()
        self.assertIn("EXTRACT COV1", text)

    def test_spectrum_uncertainty_job(self) -> None:
        runtime = self.runner.run("example_spct_unc_spectrum_averaged_response")
        ext_path = runtime.paths.output_dir / "example_spct_unc_spectrum_averaged_response.ext"
        text = ext_path.read_text()
        self.assertIn("EXTRACT COV9", text)


if __name__ == "__main__":
    unittest.main()
