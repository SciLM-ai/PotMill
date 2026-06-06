import os
import tempfile
import unittest

from potmill.config import ConfigManager
from potmill import labeling
from potmill.labeling.uma import uma, uma_batch
from potmill.labeling.vasp import vasp
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
        self.assertEqual(labeling._fairchem_kwargs(_cfg()),
                         {"name": "uma-m-1p1", "task_name": "omat", "device": "cuda"})
        kw = labeling._fairchem_kwargs(_cfg("[FairChemCalculator]\ndevice = cpu\nname = uma-s-1\n"))
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


if __name__ == "__main__":
    unittest.main()
