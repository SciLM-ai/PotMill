import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from potmill.config import ConfigManager


def _write(d, text):
    path = os.path.join(d, "config.ini")
    with open(path, "w") as f:
        f.write(text)
    return path


class TestConfigManager(unittest.TestCase):
    def test_defaults_and_coercion(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write(d, "[MAIN]\nnconfigurations = 500\n[FitSNAP]\nchem_elem = H Be W\nmlip = ACE\n")
            cfg = ConfigManager(path)
        # user override coerced to int
        self.assertEqual(cfg["MAIN"]["nconfigurations"], 500)
        # default applied for an omitted key
        self.assertEqual(cfg["MAIN"]["fit_gpus_per_node"], 2)
        self.assertEqual(cfg["MAIN"]["fit_device"], "cuda")
        # space-separated -> list
        self.assertEqual(cfg["FitSNAP"]["chem_elem"], ["H", "Be", "W"])

    def test_passthrough_section_is_raw(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write(d, "[FairChemCalculator]\ntask_name = omat\ndevice = cpu\n")
            cfg = ConfigManager(path)
        self.assertEqual(cfg["FairChemCalculator"], {"task_name": "omat", "device": "cpu"})

    def test_unknown_key_warns(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write(d, "[MAIN]\nnot_a_real_key = 3\n")
            buf = io.StringIO()
            with redirect_stdout(buf):
                ConfigManager(path)
        self.assertIn("not_a_real_key", buf.getvalue())

    def test_validate_warns_on_low_pair_style(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write(d, "[RCUT]\nmax_rcut = 6.5\n")
            cfg = ConfigManager(path)
        buf = io.StringIO()
        with redirect_stdout(buf):
            cfg.validate({"REFERENCE": {"pair_style": "zero 5.0"}})
        self.assertIn("WARNING", buf.getvalue())

        buf = io.StringIO()
        with redirect_stdout(buf):
            cfg.validate({"REFERENCE": {"pair_style": "zero 6.6"}})
        self.assertEqual(buf.getvalue(), "")

    def test_missing_file_uses_defaults(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            cfg = ConfigManager("/nonexistent/config.ini")
        self.assertIn("not found", buf.getvalue())
        self.assertEqual(cfg["MAIN"]["n_fold"], 3)


if __name__ == "__main__":
    unittest.main()
