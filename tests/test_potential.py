"""Tests for the LAMMPS potential export (``[Main] potential``).

The two things that can silently produce a wrong potential are checked against ground truth here:

* the ALL-DATA coefficients merged from the per-fold R-factors must equal a one-shot weighted
  least-squares fit over every row (``test_all_data_beta_*``), and
* the written ``.yace`` must make LAMMPS reproduce the fitted model's energy and forces
  (``TestLammpsRoundTrip``, skipped where LAMMPS/FitSNAP are unavailable).
"""

import os
import tempfile
import unittest

import numpy as np

from potmill.bfile import write_b
from potmill.fitting import config_fold, foldfit
from potmill.potential.betas import all_data_beta, all_data_beta_rows, state_n_configs
from potmill.potential.labels import check_against_feature_names, feature_names_from_blist
from potmill.potential.mod import pair_commands

try:
    import torch  # noqa: F401

    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False

try:
    import fitsnap3lib  # noqa: F401
    import lammps  # noqa: F401
    from ase.build import bulk  # noqa: F401

    _HAVE_LAMMPS = True
except ImportError:
    _HAVE_LAMMPS = False


def _write_combined_b(path, configs):
    """Concatenate per-config b blocks (mirrors combine_b's `cat`)."""
    parts = []
    for job_id, energy, forces in configs:
        tmp = path + f".{job_id}"
        write_b(tmp, job_id, energy, forces.shape[0], forces)
        parts.append(np.loadtxt(tmp, delimiter=","))
        os.remove(tmp)
    np.savetxt(path, np.vstack(parts), delimiter=",", fmt=["%i", "%i", "%.10f"])


def _synthetic_batches(rng, n_batches, n_configs, p, start=0):
    """Per-batch (design matrix, b-config tuples) with the pipeline's row layout."""
    batches = []
    job_id = start
    for _ in range(n_batches):
        blocks, configs = [], []
        for _ in range(n_configs):
            n_atoms = int(rng.integers(2, 5))
            blocks.append(rng.standard_normal((1 + 3 * n_atoms, p)))
            configs.append(
                (job_id, float(rng.standard_normal()) * 5.0, rng.standard_normal((n_atoms, 3)))
            )
            job_id += 1
        batches.append((np.vstack(blocks), configs))
    return batches


@unittest.skipUnless(_HAVE_TORCH, "torch not installed")
class TestAllDataBeta(unittest.TestCase):
    """The exported coefficients must be the all-data fit, not a fold's."""

    def _run(self, n_batches):
        rng = np.random.default_rng(7)
        p, n_fold, eweight = 6, 3, 10.0
        n_configs = 24
        subset_hp = [[5.0], [8]]
        feature_names = [[0]] * p  # every column selected by _feature_indices for SNAP
        batches = _synthetic_batches(rng, n_batches, n_configs, p)

        self.assertEqual(
            {config_fold(c, n_fold) for c in range(n_batches * n_configs)},
            set(range(n_fold)),
            "synthetic configs must populate every fold",
        )

        with tempfile.TemporaryDirectory() as root:
            feats = os.path.join(root, "features") + "/"
            state_dir = os.path.join(root, "state")
            fit_dir = os.path.join(root, "fits") + "/"
            os.makedirs(fit_dir, exist_ok=True)
            cwd = os.getcwd()
            prev = None
            try:
                for b, (a_batch, configs) in enumerate(batches):
                    os.makedirs(f"{feats}{b}/5.0", exist_ok=True)
                    np.save(f"{feats}{b}/5.0/a.npy", a_batch)
                    _write_combined_b(f"{feats}{b}/b_batch.csv", configs)
                    prev = foldfit(
                        feats,
                        feature_names,
                        None,
                        subset_hp,
                        [eweight],
                        "SNAP",
                        b,
                        prev,
                        n_fold=n_fold,
                        fit_dir_base=fit_dir,
                        state_dir=state_dir,
                        fit_device="cpu",
                    )
            finally:
                os.chdir(cwd)

            merged = all_data_beta(prev, eweight)
            # The accumulator must know how much data it ate -- this is what index.csv reports as
            # n_configs, and it is the only honest number for a run interrupted between checkpoints.
            self.assertEqual(state_n_configs(prev), n_batches * n_configs)

        # ground truth: one-shot weighted least squares over every row of every batch
        a_all = np.vstack([a for a, _ in batches])
        b_all = np.concatenate(
            [
                np.concatenate([[energy / len(forces)], np.asarray(forces).ravel()])
                for _, configs in batches
                for _, energy, forces in configs
            ]
        )
        is_energy = np.concatenate(
            [
                np.arange(1 + 3 * len(forces)) == 0
                for _, configs in batches
                for _, energy, forces in configs
            ]
        )
        direct = all_data_beta_rows(a_all, b_all, is_energy, eweight)

        np.testing.assert_allclose(merged, direct, rtol=1e-8, atol=1e-9)

    def test_single_batch(self):
        self._run(1)

    def test_multi_batch(self):
        """The merge must stay exact as batches accumulate (each row is in k-1 training sets)."""
        self._run(3)

    def test_rejects_single_fold_state(self):
        import torch

        blob = [
            {
                "solve": {"E": torch.zeros((3, 3)), "F": torch.zeros((3, 3))},
                "Sw": {("tr", "E"): 1.0, ("tr", "F"): 1.0},
            }
        ]
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "state.pt")
            torch.save(blob, path)
            with self.assertRaises(ValueError):
                all_data_beta(path, 10.0)


class TestLabels(unittest.TestCase):
    def test_feature_names_from_blist(self):
        blist = [[1, 0, 0, 1], [2, 0, 0, 2], [3, 1, 1, 1], [4, 1, 1, 2]]
        names = feature_names_from_blist(blist, 2)
        self.assertEqual(names[0], [0])
        self.assertEqual(names[3], [0])  # second element's constant column
        self.assertEqual(len(names), 6)

    def test_mismatch_is_fatal(self):
        """A basis reconstruction that disagrees with featurization must raise, never warn."""
        blist = [[1, 0, 0, 1], [2, 0, 0, 2]]
        good = feature_names_from_blist(blist, 1)
        self.assertTrue(check_against_feature_names(blist, 1, good))
        bad = [list(n) for n in good]
        bad[1][-1] = 99
        with self.assertRaises(ValueError):
            check_against_feature_names(blist, 1, bad)
        with self.assertRaises(ValueError):
            check_against_feature_names(blist, 1, good[:-1])


class TestModWriter(unittest.TestCase):
    def test_zero_reference_explicit_evaluator(self):
        lines = pair_commands(
            "pot.yace", ["W", "Be"], {"pair_style": "zero 6.6"}, evaluator="recursive"
        )
        self.assertEqual(lines[0], "pair_style pace recursive")
        self.assertEqual(lines[1], "pair_coeff * * pot.yace W Be")

    def test_default_selects_the_evaluator_at_runtime(self):
        """recursive ABORTS under KOKKOS (pair_pace_kokkos.cpp:570) and product is ~18% slower on
        CPU, so the written file must branch rather than force either on the user."""
        lines = pair_commands("pot.yace", ["W", "Be"], {"pair_style": "zero 6.6"})
        text = "\n".join(lines)
        self.assertIn("is_active(package,kokkos)", text)
        self.assertIn('"pair_style pace product"', text)
        self.assertIn('"pair_style pace recursive"', text)
        self.assertEqual(lines[-1], "pair_coeff * * pot.yace W Be")

    def test_hybrid_reference_is_passed_through(self):
        ref = {
            "pair_style": "hybrid/overlay zbl 1.0 2.0 zero 6.6",
            "pair_coeff1": "* * zbl 74 74",
            "pair_coeff2": "* * zero",
        }
        lines = pair_commands("pot.yace", ["W"], ref, evaluator="recursive")
        self.assertEqual(lines[0], "pair_style hybrid/overlay zbl 1.0 2.0 pace recursive")
        self.assertIn("pair_coeff * * zbl 74 74", lines)
        self.assertNotIn("pair_coeff * * zero", lines)
        self.assertEqual(lines[-1], "pair_coeff * * pace pot.yace W")

    def test_hybrid_reference_also_branches_by_default(self):
        ref = {"pair_style": "hybrid/overlay zbl 1.0 2.0 zero 6.6", "pair_coeff1": "* * zbl 74 74"}
        text = "\n".join(pair_commands("pot.yace", ["W"], ref))
        self.assertIn('"pair_style hybrid/overlay zbl 1.0 2.0 pace product"', text)
        self.assertIn('"pair_style hybrid/overlay zbl 1.0 2.0 pace recursive"', text)

    def test_unknown_reference_raises(self):
        """An unsupported reference would silently drop the term the fit subtracted."""
        with self.assertRaises(ValueError):
            pair_commands("pot.yace", ["W"], {"pair_style": "eam/alloy"})


ACE_TEST_CONFIG = {
    "ACE": {
        "numTypes": "1",
        "type": "W",
        "ranks": "1 2",
        "lmin": "0 0",
        "lmax": "0 1",
        "nmax": "4 2",
        "nmaxbase": "12",
        "rcutfac": "5.0",
        "lambda": "1.5",
        "rcinner": "1.0",
        "drcinner": "0.01",
        "bzeroflag": "0",
    },
    "CALCULATOR": {"calculator": "LAMMPSPACE", "energy": "1", "force": "1", "stress": "0"},
    "ESHIFT": {"W": "0.0"},
    "GROUPS": {
        "group_sections": "name training_size testing_size eweight fweight vweight",
        "group_types": "str float float float float float",
        "smartweights": "0",
        "random_sampling": "0",
        "All": "1.0 0.0 1.0 1.0 0.0",
    },
    "SOLVER": {"solver": "SVD", "compute_testerrs": "1", "detailed_errors": "1"},
    "REFERENCE": {
        "units": "metal",
        "atom_style": "atomic",
        "pair_style": "zero 6.0",
        "pair_coeff": "* *",
    },
    "MEMORY": {"override": "0"},
    "SCRAPER": {"scraper": "JSON"},
    "OUTFILE": {"output_style": "PACE"},
}


@unittest.skipUnless(_HAVE_LAMMPS, "LAMMPS (ML-PACE) / FitSNAP / ASE not installed")
class TestLammpsRoundTrip(unittest.TestCase):
    """LAMMPS running the written potential must reproduce natoms*(a_E@beta) and a_F@beta.

    Random coefficients are legitimate here: the identity holds for ANY beta, so this tests the
    potential file, not the quality of a fit.
    """

    def test_energy_and_forces_match_the_model(self):
        import copy

        from ase.build import bulk

        from potmill.potential.ace import write_yace
        from potmill.potential.mod import write_mod
        from potmill.potential.verify import lammps_ef

        rng = np.random.default_rng(3)
        atoms = bulk("W", "bcc", a=3.17, cubic=True) * (2, 2, 2)
        atoms.rattle(0.1, seed=5)
        nmax, lmax, rcuts = [4, 2], [0, 1], [5.0]

        with tempfile.TemporaryDirectory() as root:
            from potmill.featurization.featurize import featurize

            cwd = os.getcwd()
            try:
                names = featurize(
                    [atoms.copy()],
                    {"FitSNAP": {"mlip": "ACE"}},
                    copy.deepcopy(ACE_TEST_CONFIG),
                    rcuts,
                    root,
                )
            finally:
                os.chdir(cwd)
            a_matrix = np.load(os.path.join(root, "a.npy"))
            beta = rng.normal(scale=0.01, size=a_matrix.shape[1])
            natoms = len(atoms)
            ref_e = natoms * float(a_matrix[0] @ beta)
            ref_f = (a_matrix[1 : 1 + 3 * natoms] @ beta).reshape(natoms, 3)

            name = "test_pot"
            write_yace(
                os.path.join(root, name),
                ACE_TEST_CONFIG["ACE"],
                rcuts,
                nmax,
                lmax,
                nmax,
                lmax,
                beta,
                names,
            )
            # Both algorithms, plus the runtime-branching file the exporter actually writes
            # (evaluator=None). They must all reproduce the model identically.
            for evaluator in ("recursive", "product", None):
                write_mod(
                    os.path.join(root, f"{name}.mod"),
                    f"{name}.yace",
                    ["W"],
                    ACE_TEST_CONFIG["REFERENCE"],
                    evaluator=evaluator,
                )
                lmp_e, lmp_f = lammps_ef(atoms, root, f"{name}.mod", ["W"])
                label = evaluator or "runtime-selected"
                self.assertLess(abs(lmp_e - ref_e) / natoms, 1e-9, f"energy ({label})")
                self.assertLess(float(np.max(np.abs(lmp_f - ref_f))), 1e-9, f"forces ({label})")


if __name__ == "__main__":
    unittest.main()
