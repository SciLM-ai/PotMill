import os
import tempfile
import unittest

from potmill.config import ConfigManager
from potmill import labeling
from potmill.labeling.uma import uma, uma_batch
from potmill.labeling.vasp import vasp, parse_setups, _VASP_DEFAULTS, _MAGMOM
from potmill.labeling.lammps import lammps


def _cfg(text=""):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "config.ini")
        with open(path, "w") as f:
            f.write(text)
        return ConfigManager(path)


class TestMakeLabeling(unittest.TestCase):
    def test_default_is_fairchem(self):
        lab = labeling.make_labeling(_cfg())
        self.assertIs(lab.per_config, uma)
        self.assertIs(lab.batched, uma_batch)
        self.assertTrue(callable(lab.init_function))

    def test_fairchem_kwargs_defaults_and_override(self):
        self.assertEqual(
            labeling._fairchem_kwargs(_cfg()),
            {"name": "uma-m-1p1", "task_name": "omat", "device": "cuda"},
        )
        kw = labeling._fairchem_kwargs(_cfg("[FAIRChemCalculator]\ndevice = cpu\nname = uma-s-1\n"))
        self.assertEqual(kw["device"], "cpu")
        self.assertEqual(kw["name"], "uma-s-1")
        self.assertEqual(kw["task_name"], "omat")  # default preserved

    def test_vasp_backend(self):
        lab = labeling.make_labeling(_cfg("[ourLabeling]\ncalculator = Vasp\n"))
        self.assertIs(lab.per_config, vasp)
        self.assertIsNone(lab.batched)

    def test_lammps_backend(self):
        lab = labeling.make_labeling(_cfg("[ourLabeling]\ncalculator = LAMMPS\n"))
        self.assertIs(lab.per_config, lammps)
        self.assertIsNone(lab.batched)

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            labeling.make_labeling(_cfg("[ourLabeling]\ncalculator = Nonsense\n"))


class TestVaspSettings(unittest.TestCase):
    def test_defaults_present(self):
        # exact vasp-ase-sp.py single-point settings are the backend defaults
        self.assertEqual(_VASP_DEFAULTS["encut"], 500)
        self.assertEqual(_VASP_DEFAULTS["ismear"], 0)
        self.assertEqual(_VASP_DEFAULTS["ediff"], 1e-6)
        self.assertEqual(_VASP_DEFAULTS["kspacing"], 0.125)
        self.assertEqual(_VASP_DEFAULTS["prec"], "Accurate")

    def test_user_overrides_default(self):
        # the [Vasp] kwargs override the defaults (same merge the backend does)
        merged = {**_VASP_DEFAULTS, **{"encut": 300, "ismear": 1}}
        self.assertEqual(merged["encut"], 300)
        self.assertEqual(merged["ismear"], 1)
        self.assertEqual(merged["prec"], "Accurate")  # untouched default remains

    def test_parse_setups(self):
        self.assertEqual(parse_setups("recommended"), {"base": "recommended"})
        self.assertEqual(
            parse_setups("recommended W:_sv"), {"base": "recommended", "W": "_sv"}
        )
        # interpret_string turns a multi-token value into a list before it reaches us
        self.assertEqual(
            parse_setups(["recommended", "W:_sv", "Mo:_pv"]),
            {"base": "recommended", "W": "_sv", "Mo": "_pv"},
        )
        self.assertEqual(parse_setups({"base": "minimal"}), {"base": "minimal"})

    def test_magmom_any_element(self):
        # known elements use their tabulated moment; unknown elements fall back to 1.0
        moments = [_MAGMOM.get(s, _MAGMOM["default"]) for s in ("W", "Be", "Fe", "Xx")]
        self.assertEqual(moments, [1.0, 1.0, 2.5, 1.0])


if __name__ == "__main__":
    unittest.main()
