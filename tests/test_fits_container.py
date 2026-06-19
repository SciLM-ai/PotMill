import glob
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from potmill.fitting.fit import _write_results, read_beta
from potmill.tools import hyperparameters_to_string

_RMSE_KEYS = ("tr_E", "tr_F", "te_E", "te_F", "tr_E_w", "tr_F_w", "te_E_w", "te_F_w")
# columns of results_{pid}.csv: fold, then the ACE combo cols, then the 8 RMSEs (see _write_results)
_COLS = [
    "fold",
    "rcut0",
    "nmax1",
    "nmax2",
    "lmax1",
    "lmax2",
    "eweight",
    "train_e_rmse",
    "train_f_rmse",
    "test_e_rmse",
    "test_f_rmse",
    "train_e_rmse_weighted",
    "train_f_rmse_weighted",
    "test_e_rmse_weighted",
    "test_f_rmse_weighted",
]


class TestFitsContainer(unittest.TestCase):
    def test_container_roundtrip(self):
        """_write_results must (a) produce only per-worker container files (no per-combo dirs),
        (b) let pareto's group-by-combo mean reproduce the old per-combo results.csv mean, and
        (c) let read_beta recover each fitted beta bit-for-bit."""
        rng = np.random.default_rng(0)
        mlip = "ACE"
        n_fold = 3
        # combos 0,1 share rcut/nmax/lmax but differ in eweight; combo 2 differs entirely
        combos = [
            ([5.0], [5, 2], [0, 1], 10.0),
            ([5.0], [5, 2], [0, 1], 20.0),
            ([6.5], [9, 4], [0, 4], 10.0),
        ]
        expected_mean = {}  # combo idx -> {rmse_key: mean over folds}
        betas = {}  # (combo idx, fold) -> beta array

        with tempfile.TemporaryDirectory() as batch_dir:
            for ci, hp in enumerate(combos):
                per_fold = []
                for fold in range(n_fold):
                    res = {k: float(rng.standard_normal()) for k in _RMSE_KEYS}
                    res["beta"] = rng.standard_normal(int(rng.integers(5, 20)))
                    betas[(ci, fold)] = res["beta"]
                    per_fold.append(res)
                    _write_results(batch_dir, mlip, hp, fold, res)
                expected_mean[ci] = {k: np.mean([r[k] for r in per_fold]) for k in _RMSE_KEYS}

            # (a) only container files, no per-combo subdirectories
            entries = sorted(os.listdir(batch_dir))
            self.assertTrue(all(os.path.isfile(os.path.join(batch_dir, e)) for e in entries), entries)
            self.assertTrue(all(e.startswith(("results_", "betas_")) for e in entries), entries)
            self.assertEqual(len(glob.glob(os.path.join(batch_dir, "results_*.csv"))), 1)  # one pid

            # (b) pareto's group-by-combo mean == per-combo mean over folds
            df = pd.concat(
                [pd.read_csv(f, header=None) for f in glob.glob(f"{batch_dir}/results_*.csv")],
                ignore_index=True,
            )
            df.columns = _COLS
            grouped = df.groupby(_COLS[1:7], as_index=False)[_COLS[7:]].mean()
            self.assertEqual(len(grouped), len(combos))
            for ci, (rcut, _nmax, _lmax, ew) in enumerate(combos):
                row = grouped[(grouped.rcut0 == rcut[0]) & (grouped.eweight == ew)]
                self.assertEqual(len(row), 1)
                # results.csv stores RMSEs at %.10f, so compare within that precision
                self.assertAlmostEqual(
                    row.test_e_rmse.iloc[0], expected_mean[ci]["te_E"], places=7
                )
                self.assertAlmostEqual(
                    row.test_f_rmse_weighted.iloc[0], expected_mean[ci]["te_F_w"], places=7
                )

            # (c) read_beta recovers each fitted beta bit-for-bit
            for ci, hp in enumerate(combos):
                combo_string = hyperparameters_to_string(mlip, hp, delimiter="_")
                for fold in range(n_fold):
                    got = read_beta(batch_dir, combo_string, fold)
                    np.testing.assert_array_equal(got, betas[(ci, fold)].astype("<f8"))


if __name__ == "__main__":
    unittest.main()
