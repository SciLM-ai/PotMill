import os
import glob
import tempfile
import unittest

import numpy as np

from potmill.bfile import write_b
from potmill.fitting import fit, foldfit, config_fold, _feature_indices


def _write_combined_b(path, configs):
    """Concatenate per-config b blocks (mirrors combine_b's `cat`)."""
    parts = []
    for job_id, energy, forces in configs:
        tmp = path + f".{job_id}"
        write_b(tmp, job_id, energy, forces.shape[0], forces)
        parts.append(np.loadtxt(tmp, delimiter=","))
        os.remove(tmp)
    np.savetxt(path, np.vstack(parts), delimiter=",", fmt=["%i", "%i", "%.10f"])


class TestFitFoldfitEquivalence(unittest.TestCase):
    def test_single_batch_rows_vs_incremental(self):
        rng = np.random.default_rng(0)
        mlip = "SNAP"
        subset_hp = [[5.0], [8]]
        eweight = 10.0
        n_fold = 3
        p = 5
        n_configs = 15
        feature_names = [[0]] * p  # all columns selected by _feature_indices

        # build aligned a / b in config-major order
        a_blocks, configs = [], []
        for c in range(n_configs):
            n_atoms = int(rng.integers(2, 4))
            n_rows = 1 + 3 * n_atoms
            a_blocks.append(rng.standard_normal((n_rows, p)))
            configs.append(
                (c, float(rng.standard_normal()) * 5.0, rng.standard_normal((n_atoms, 3)))
            )
        a = np.vstack(a_blocks)

        # all folds must be populated for the per-fold RMSE means
        folds = {config_fold(c, n_fold) for c in range(n_configs)}
        self.assertEqual(folds, set(range(n_fold)))

        with tempfile.TemporaryDirectory() as root:
            feats = os.path.join(root, "features") + "/"
            os.makedirs(f"{feats}0/5.0", exist_ok=True)
            np.save(f"{feats}0/5.0/a.npy", a)
            _write_combined_b(f"{feats}0/b_batch.csv", configs)
            _write_combined_b(f"{feats}b{n_configs}.csv", configs)

            fit_dir = os.path.join(root, "rows")
            os.makedirs(fit_dir, exist_ok=True)
            cwd = os.getcwd()
            try:
                fit(
                    feats,
                    feature_names,
                    list(range(n_configs)),
                    subset_hp + [eweight],
                    mlip,
                    batch_ID=0,
                    n_fold=n_fold,
                    fit_directory=os.path.join(fit_dir, "combo"),
                    fit_device="cpu",
                    fit_method="svd",
                )
            finally:
                os.chdir(cwd)

            fold_base = os.path.join(root, "incr") + "/"
            os.makedirs(fold_base, exist_ok=True)  # __main__ makes the per-batch fits/{i}/ dir
            state_dir = os.path.join(root, "state")
            foldfit(
                feats,
                feature_names,
                None,
                subset_hp,
                [eweight],
                mlip,
                0,
                None,
                n_fold=n_fold,
                fit_dir_base=fold_base,
                state_dir=state_dir,
                fit_device="cpu",
                fit_method="svd",
            )

            rows_csv = np.loadtxt(glob.glob(os.path.join(fit_dir, "results_*.csv"))[0], delimiter=",")
            incr_csv = np.loadtxt(glob.glob(f"{fold_base}results_*.csv")[0], delimiter=",")

        rows_csv = rows_csv[rows_csv[:, 0].argsort()]
        incr_csv = incr_csv[incr_csv[:, 0].argsort()]
        self.assertEqual(rows_csv.shape, incr_csv.shape)
        np.testing.assert_allclose(rows_csv, incr_csv, rtol=1e-6, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
