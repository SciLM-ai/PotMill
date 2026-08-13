"""Tests for the MD stability stage (``[Main] md``).

The stage's verdict is only as good as three things that are easy to get subtly wrong: the test cell
must be big enough for the dynamics to be real, a collapsed structure must be detected, and an
unstable potential must come back as a RESULT rather than an exception.
"""

import os
import tempfile
import unittest

import numpy as np

from potmill.md.runner import COLLAPSE_COMPRESSION, FUSED_DISTANCE, _min_distance
from potmill.md.stage import merge_md_task
from potmill.md.structure import replicate_for_md

try:
    from ase.build import bulk

    _HAVE_ASE = True
except ImportError:
    _HAVE_ASE = False

try:
    import lammps  # noqa: F401

    _HAVE_LAMMPS = True
except ImportError:
    _HAVE_LAMMPS = False


@unittest.skipUnless(_HAVE_ASE, "ase not installed")
class TestReplication(unittest.TestCase):
    """A 2-atom entropy cell is mostly interacting with its own periodic images -- MD in it is an
    artifact, so the stage must grow it before testing anything."""

    def test_grows_to_min_atoms_and_min_cell(self):
        small = bulk("W", "bcc", a=3.17, cubic=True)  # 2 atoms, 3.17 A cell
        grown, reps = replicate_for_md(small, min_atoms=200, min_cell_length=13.0)
        self.assertGreaterEqual(len(grown), 200)
        lengths = np.linalg.norm(np.asarray(grown.get_cell()), axis=1)
        self.assertTrue(np.all(lengths >= 13.0), f"cell {lengths} shorter than twice the cutoff")
        self.assertEqual(len(grown), len(small) * int(np.prod(reps)))

    def test_leaves_a_big_enough_cell_alone(self):
        big = bulk("W", "bcc", a=3.17, cubic=True) * (6, 6, 6)
        grown, reps = replicate_for_md(big, min_atoms=200, min_cell_length=13.0)
        self.assertEqual(reps, (1, 1, 1))
        self.assertEqual(len(grown), len(big))


@unittest.skipUnless(_HAVE_ASE, "ase not installed")
class TestCandidateSelection(unittest.TestCase):
    """Lowest formation energy is the right ORDER but not a sufficient condition: entropy-maximized
    configurations routinely contain contacts ACE cannot evaluate (measured: 0.885 A at rank 0 on a
    real run), which would make every potential look unstable when none of them is."""

    def _cell(self, closest):
        from ase import Atoms

        return Atoms("H2", positions=[[0, 0, 0], [closest, 0, 0]], cell=np.eye(3) * 12.0, pbc=True)

    def test_skips_candidates_that_are_too_close(self):
        from potmill.md.structure import select_evaluable

        structures = {0: self._cell(0.885), 1: self._cell(2.4)}
        atoms, info = select_evaluable([0, 1], structures, {0: 0.0, 1: 0.4}, min_distance=1.1)
        self.assertEqual(info["job_id"], 1)
        self.assertEqual(info["rank_by_formation_energy"], 1)
        self.assertEqual(info["rejected_too_close"], 1)
        self.assertAlmostEqual(info["min_distance"], 2.4, places=6)
        self.assertEqual(len(atoms), 2)

    def test_takes_the_best_when_it_is_fine(self):
        from potmill.md.structure import select_evaluable

        structures = {0: self._cell(2.5), 1: self._cell(2.4)}
        _, info = select_evaluable([0, 1], structures, {0: 0.0, 1: 0.4}, min_distance=1.1)
        self.assertEqual(info["rank_by_formation_energy"], 0)
        self.assertEqual(info["rejected_too_close"], 0)

    def test_prefers_the_least_compressed_candidate(self):
        """Both are evaluable, but starting from the squeezed one makes every potential look bad --
        measured on a real run as 4/4 stable vs 0/4 from two candidates of the same run."""
        from potmill.md.structure import select_evaluable

        structures = {0: self._cell(1.5), 1: self._cell(2.4)}
        _, info = select_evaluable([0, 1], structures, {0: 0.0, 1: 0.4}, min_distance=1.1)
        self.assertEqual(info["job_id"], 1)
        self.assertGreater(info["compression"], 1.0)

    def test_raises_when_nothing_is_evaluable(self):
        """Never fall back to a structure MD cannot start from -- say so instead."""
        from potmill.md.structure import select_evaluable

        structures = {0: self._cell(0.5), 1: self._cell(0.9)}
        with self.assertRaises(ValueError):
            select_evaluable([0, 1], structures, {0: 0.0, 1: 0.4}, min_distance=1.1)


@unittest.skipUnless(_HAVE_ASE, "ase not installed")
class TestCollapseCriterion(unittest.TestCase):
    """Collapse must be judged as a FRACTION of the pair's covalent bond length, never as an
    absolute distance. Two real false verdicts came from getting this wrong: an absolute 1.1 A floor
    called a perfectly normal H-H contact (H2 is 0.74 A) a collapse, and a floor derived from
    ``rcinner`` became 0.1 A when rcinner was 0, passing a run whose atoms ended up 0.58 A apart.
    """

    def test_short_hydrogen_contact_is_not_a_collapse(self):
        from ase import Atoms

        from potmill.md.structure import compression

        h2 = Atoms("H2", positions=[[0, 0, 0], [0.9, 0, 0]], cell=np.eye(3) * 12.0, pbc=True)
        self.assertLess(0.9, 1.1, "this contact is shorter than the old absolute floor")
        self.assertGreater(compression(h2), COLLAPSE_COMPRESSION)

    def test_squeezed_heavy_contact_is_a_collapse(self):
        from ase import Atoms

        from potmill.md.structure import compression

        w2 = Atoms("W2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=np.eye(3) * 12.0, pbc=True)
        self.assertGreater(1.5, 1.1, "this contact would have PASSED the old absolute floor")
        self.assertLess(compression(w2), COLLAPSE_COMPRESSION)


@unittest.skipUnless(_HAVE_ASE, "ase not installed")
class TestMinDistance(unittest.TestCase):
    def test_none_when_nothing_is_close(self):
        """No pair within the search cutoff must read as 'nothing is close', not as a NaN that
        later gets mistaken for a broken simulation."""
        positions = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
        self.assertIsNone(_min_distance(positions, np.eye(3) * 20.0))

    def test_detects_collapse(self):
        positions = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [8.0, 8.0, 8.0]])
        d = _min_distance(positions, np.eye(3) * 20.0)
        self.assertAlmostEqual(d, 0.3, places=6)
        self.assertLess(d, FUSED_DISTANCE)

    def test_respects_periodic_images(self):
        """Two atoms far apart in the cell can still be neighbours across the boundary."""
        positions = np.array([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]])
        self.assertAlmostEqual(_min_distance(positions, np.eye(3) * 10.0), 0.2, places=6)


class TestMerge(unittest.TestCase):
    def test_writes_md_csv_and_joins_index(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "potentials"))
            index = pd.DataFrame(
                [
                    {"dir": "pot_a", "test_e_rmse": 1.0, "status": "ok"},
                    {"dir": "pot_b", "test_e_rmse": 2.0, "status": "ok"},
                ]
            )
            index.to_csv(os.path.join(root, "potentials", "index.csv"), index=False)
            merge_md_task(
                root,
                {"dir": "pot_a", "md_ok": 1, "md_drift_per_atom_per_ps": 1e-6},
                {"dir": "pot_b", "md_ok": 0, "md_note": "atoms collapsed"},
                None,  # a task with no potential to test returns None and must be ignored
            )
            md = pd.read_csv(os.path.join(root, "potentials", "md.csv"))
            self.assertEqual(sorted(md["dir"]), ["pot_a", "pot_b"])
            joined = pd.read_csv(os.path.join(root, "potentials", "index.csv"))
            self.assertIn("md_ok", joined.columns)
            self.assertEqual(joined.loc[joined["dir"] == "pot_a", "md_ok"].iloc[0], 1)

    def test_rerun_does_not_duplicate_columns(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "potentials"))
            pd.DataFrame([{"dir": "pot_a", "status": "ok"}]).to_csv(
                os.path.join(root, "potentials", "index.csv"), index=False
            )
            for _ in range(2):
                merge_md_task(root, {"dir": "pot_a", "md_ok": 1})
            joined = pd.read_csv(os.path.join(root, "potentials", "index.csv"))
            self.assertEqual(list(joined.columns).count("md_ok"), 1)


@unittest.skipUnless(_HAVE_LAMMPS and _HAVE_ASE, "LAMMPS (ML-PACE) / ASE not installed")
class TestUnstablePotentialIsAResult(unittest.TestCase):
    """An exploding potential must be reported, never raised: the stage screens many potentials and
    the unstable ones are exactly what it exists to find."""

    def test_missing_potential_file_is_reported(self):
        from potmill.md.runner import run_md

        atoms = bulk("W", "bcc", a=3.17, cubic=True) * (2, 2, 2)
        with tempfile.TemporaryDirectory() as root:
            record = run_md(atoms, root, "does_not_exist", ["W"], steps=10, minimize=False)
        self.assertEqual(record["ok"], 0)
        self.assertTrue(record["note"], "a failure must explain itself")


if __name__ == "__main__":
    unittest.main()
