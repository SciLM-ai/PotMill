"""Tests for the streaming POPS uncertainty estimator and the artifact shipped with a potential.

Three things must hold for this to be trustworthy: it must be the SAME estimator the POPS library
implements (validated against it where installed), its streaming form must equal the in-memory form
it replaces (the whole point is that we never materialize the n x p corrections), and the box that
gets shipped must reproduce exactly what an ensemble drawn from it would have said.
"""

import os
import tempfile
import unittest

import numpy as np

from potmill.uq.artifact import calibrate, load_uq, save_uq
from potmill.uq.pops import POPSPosterior, evidence_ridge_from_gram, fit_pops_streaming
from potmill.uq.stage import WeightedChunks, fit_weights, uq_for_potential

try:
    from sklearn.linear_model import BayesianRidge

    _HAVE_SKLEARN = True
except ImportError:
    _HAVE_SKLEARN = False

try:
    from popsregression import POPSRegression

    _HAVE_POPS = True
except ImportError:
    _HAVE_POPS = False


def _misspecified_problem(n=1500, p=20, seed=0):
    """Deliberately misspecified: a quadratic term a linear model cannot represent, tiny noise --
    the regime POPS targets (misspecification dominates aleatoric error)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = X @ rng.normal(size=p) + 0.3 * X[:, 0] ** 2 + 0.001 * rng.normal(size=n)
    return X, y


class _Chunks:
    """Re-iterable batches: the hypercube posterior needs a second pass over the data."""

    def __init__(self, X, y, k=5):
        self.X, self.y, self.k = X, y, k

    def __iter__(self):
        for sel in np.array_split(np.arange(self.X.shape[0]), self.k):
            yield self.X[sel], self.y[sel]


def _anchor(X, y):
    """The (beta, sigma, alpha) anchor POPS is built around, from the sufficient statistics."""
    _, sigma, alpha, _ = evidence_ridge_from_gram(X.T @ X, X.T @ y, float(y @ y), X.shape[0])
    return np.linalg.lstsq(X, y, rcond=None)[0], sigma, alpha


class TestEvidenceRidgeFromGram(unittest.TestCase):
    @unittest.skipUnless(_HAVE_SKLEARN, "scikit-learn not installed")
    def test_matches_sklearn_from_sufficient_statistics_only(self):
        """The Bayesian half must come from X^T X, X^T y and y^T y alone -- that is what makes it
        free from the fit's accumulated R-factors, with no design-matrix pass."""
        X, y = _misspecified_problem()
        ref = BayesianRidge(fit_intercept=False).fit(X, y)
        coef, sigma, alpha, _ = evidence_ridge_from_gram(X.T @ X, X.T @ y, float(y @ y), X.shape[0])
        np.testing.assert_allclose(coef, ref.coef_, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(sigma, ref.sigma_, rtol=1e-5, atol=1e-12)
        self.assertAlmostEqual(alpha / ref.alpha_, 1.0, places=6)

    def test_survives_an_ill_conditioned_gram(self):
        """Real ACE design matrices reach condition numbers ~1e18, where forming alpha * eigvals
        overflows float64 -- the failure this iteration was rewritten to avoid."""
        rng = np.random.default_rng(1)
        p = 30
        basis = np.linalg.qr(rng.normal(size=(p, p)))[0]
        eigvals = np.logspace(0, -18, p)  # spans 18 orders of magnitude
        gram = (basis * eigvals) @ basis.T
        coef, sigma, alpha, lam = evidence_ridge_from_gram(
            gram, basis @ (eigvals * rng.normal(size=p)), 1.0, 1000
        )
        for name, value in (("coef", coef), ("sigma", sigma)):
            self.assertTrue(np.all(np.isfinite(value)), f"{name} went non-finite")
        self.assertTrue(np.isfinite(alpha) and np.isfinite(lam))


class TestStreamingEquivalence(unittest.TestCase):
    def test_streaming_matches_in_memory(self):
        """Chunking must not change the answer -- the estimator is an accumulation of rank-1 terms."""
        X, y = _misspecified_problem()
        beta, sigma, alpha = _anchor(X, y)
        one = fit_pops_streaming(_Chunks(X, y, k=1), beta, sigma, alpha, posterior="ensemble")
        many = fit_pops_streaming(_Chunks(X, y, k=17), beta, sigma, alpha, posterior="ensemble")
        np.testing.assert_allclose(many.sigma_miss, one.sigma_miss, rtol=1e-10, atol=1e-14)
        self.assertEqual(many.n_rows_used, one.n_rows_used)

    def test_std_factor_matches_the_quadratic_form(self):
        """std() uses an eigen-factor (one BLAS matmul, 270x faster on 100k structures); it must
        equal the explicit x^T Sigma x it replaces."""
        rng = np.random.default_rng(3)
        p = 12
        A = rng.normal(size=(p, p))
        cov = A @ A.T
        post = POPSPosterior(cov, None, None, None, 0.5 * cov, "ensemble", 10, 10)
        X = rng.normal(size=(50, p))
        explicit = np.sqrt(np.einsum("ij,jk,ik->i", X, post.sigma_total, X))
        np.testing.assert_allclose(post.std(X), explicit, rtol=1e-10, atol=1e-12)

    def test_selection_threshold_is_residual_based(self):
        """Rows are selected by |residual| >= minimum_relative_error * RMSE (the library's current
        criterion); a threshold above every residual must raise rather than return an empty fit."""
        X, y = _misspecified_problem(n=400, p=8)
        beta, sigma, alpha = _anchor(X, y)
        loose = fit_pops_streaming(
            _Chunks(X, y), beta, sigma, alpha, posterior="ensemble", minimum_relative_error=0.0
        )
        tight = fit_pops_streaming(
            _Chunks(X, y), beta, sigma, alpha, posterior="ensemble", minimum_relative_error=1.0
        )
        self.assertEqual(loose.n_rows_used, X.shape[0])
        self.assertLess(tight.n_rows_used, loose.n_rows_used)
        with self.assertRaises(ValueError):
            fit_pops_streaming(_Chunks(X, y), beta, sigma, alpha, minimum_relative_error=1e6)


class TestHypercube(unittest.TestCase):
    """The box IS the shipped object, so its analytic quantities must equal the sampled ones."""

    def setUp(self):
        self.X, self.y = _misspecified_problem(n=2000, p=15)
        beta, sigma, alpha = _anchor(self.X, self.y)
        self.beta = beta
        self.post = fit_pops_streaming(
            _Chunks(self.X, self.y), beta, sigma, alpha, posterior="hypercube"
        )

    def test_bracket_contains_most_residuals(self):
        """The paper's claim, in miniature: the bracket over the whole set should contain the actual
        error for the large majority of points, which the covariance alone does not."""
        low, high = self.post.bounds(self.X)
        residual = self.X @ self.beta - self.y
        covered = float(np.mean((residual >= low) & (residual <= high)))
        self.assertGreater(covered, 0.8, f"bracket covered only {covered:.1%} of residuals")
        self.assertGreater(self.post.std(self.X).mean(), 0.0)

    def test_analytic_bounds_are_the_exact_corner_extremes(self):
        """The bound is a maximum over the box, not over a sample of it: no ensemble, however large,
        may exceed it, and a large one must approach it."""
        ensemble = self.post.sample(4000, seed=0)  # p x n
        predictions = self.X[:50] @ ensemble
        low, high = self.post.bounds(self.X[:50])
        self.assertTrue(np.all(predictions.min(axis=1) >= low - 1e-12))
        self.assertTrue(np.all(predictions.max(axis=1) <= high + 1e-12))

    def test_analytic_sigma_matches_a_large_ensemble(self):
        """sigma comes from the box's exact second moment; sampling it only ever added noise."""
        ensemble = self.post.sample(20000, seed=1)
        sampled = np.sqrt(
            np.einsum("ij,ij->i", self.X[:200] @ ensemble, self.X[:200] @ ensemble) / 20000
            + np.einsum("ij,jk,ik->i", self.X[:200], self.post.sigma_epi, self.X[:200])
        )
        np.testing.assert_allclose(self.post.std(self.X[:200]), sampled, rtol=0.02)

    def test_ensemble_posterior_has_no_box(self):
        X, y = _misspecified_problem(n=300, p=6)
        beta, sigma, alpha = _anchor(X, y)
        post = fit_pops_streaming(_Chunks(X, y), beta, sigma, alpha, posterior="ensemble")
        self.assertIsNone(post.projections)
        for call in (post.bounds, post.sample):
            with self.assertRaises(ValueError):
                call(X) if call is post.bounds else call(10)


class TestArtifact(unittest.TestCase):
    def test_round_trip_preserves_predictions(self):
        """float32 storage must not move a standard deviation or a bracket that anyone would notice
        -- and beta must come back EXACTLY, since it has to match the .yace it ships beside."""
        X, y = _misspecified_problem(n=800, p=10)
        beta, sigma, alpha = _anchor(X, y)
        post = fit_pops_streaming(_Chunks(X, y), beta, sigma, alpha)
        errors = np.abs(X @ beta - y)
        calibration = calibrate(post.std(X), errors)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_uq(
                os.path.join(tmp, "p.uq.npz"),
                post,
                beta,
                np.arange(10),
                calibration,
                {"rcut": "5.0", "n_configs": np.int64(800)},
            )
            back, beta_back, columns, meta = load_uq(path)
        np.testing.assert_array_equal(beta_back, beta)
        np.testing.assert_array_equal(columns, np.arange(10))
        np.testing.assert_allclose(back.std(X), post.std(X), rtol=1e-5)
        np.testing.assert_allclose(back.bounds(X)[0], post.bounds(X)[0], rtol=1e-4, atol=1e-8)
        self.assertEqual(meta["rcut"], "5.0")
        self.assertEqual(meta["n_configs"], 800)
        self.assertEqual(meta["calib_n"], calibration["calib_n"])

    def test_calibration_hits_its_coverage(self):
        """A q68 factor means what it says: scaling by it must put 68% of the errors inside."""
        rng = np.random.default_rng(5)
        sigma = rng.uniform(0.1, 1.0, size=5000)
        errors = np.abs(rng.normal(scale=sigma))
        out = calibrate(sigma, errors)
        for level in (68, 95):
            covered = np.mean(errors <= out[f"calib_q{level}"] * sigma)
            self.assertAlmostEqual(covered, level / 100.0, places=2)
        self.assertGreater(out["calib_q95"], out["calib_q68"])
        self.assertAlmostEqual(out["raw_coverage"], float(np.mean(errors <= sigma)), places=12)
        with self.assertRaises(ValueError):
            calibrate(np.zeros(10), np.ones(10))


class TestStageWeighting(unittest.TestCase):
    def test_weights_are_the_fit_s_own(self):
        """The stage must weight energy rows exactly as fitting/fit.py does -- exp(-E/5) normalized
        to sum to eweight -- or POPS is anchored on a loss beta does not minimize."""
        targets = np.array([-3.0, -1.0, 0.5, 2.0])
        weights = fit_weights(targets, 10.0)
        expected = np.exp(-targets / 5.0)
        np.testing.assert_allclose(weights, expected / expected.sum() * 10.0)
        self.assertAlmostEqual(weights.sum(), 10.0)

    def test_weighted_chunks_reassemble_the_weighted_matrix(self):
        rng = np.random.default_rng(7)
        rows, targets = rng.normal(size=(53, 4)), rng.normal(size=53)
        weights = fit_weights(targets, 3.0)
        chunks = WeightedChunks(rows, targets, weights, n_chunks=7)
        stacked = np.concatenate([r for r, _ in chunks])
        np.testing.assert_allclose(stacked, rows * weights[:, None])
        np.testing.assert_allclose(
            np.concatenate([t for _, t in chunks]), targets * weights, rtol=1e-12
        )

    def test_end_to_end_uncertainty_ranks_and_calibrates(self):
        """The stage's contract on data where the answer is known: sigma must rank held-out errors
        (a region the model fits badly must come out more uncertain) and q68 must be a real 68%."""
        rng = np.random.default_rng(11)
        n, p = 900, 6
        rows = rng.normal(size=(n, p))
        rows[:150] *= 4.0  # an extrapolative corner the linear model cannot cover
        coef = rng.normal(size=p)
        targets = rows @ coef + 0.2 * rows[:, 0] ** 2
        jobs = np.arange(n)
        beta = np.linalg.lstsq(rows, targets, rcond=None)[0]
        fold_betas = [beta + 0.01 * rng.normal(size=p) for _ in range(3)]
        settings = {
            "posterior": "hypercube",
            "minimum_relative_error": 0.01,
            "percentile_clipping": 0.0,
        }
        post, calibration, stats = uq_for_potential(
            rows, targets, jobs, 10.0, beta, settings, fold_betas, 3
        )
        self.assertGreater(stats["uq_spearman"], 0.3)
        self.assertGreater(stats["uq_n_modes"], 0)
        sigma = post.std(rows)
        held_out = np.abs(rows @ fold_betas[0] - targets)
        self.assertAlmostEqual(
            float(np.mean(held_out <= calibration["calib_q68"] * sigma)), 0.68, delta=0.15
        )


@unittest.skipUnless(_HAVE_POPS, "popsregression not installed")
class TestAgainstTheLibrary(unittest.TestCase):
    def test_reproduces_popsregression_given_the_same_anchor(self):
        """Ground truth: with the library's own coefficients and covariance as the anchor, our
        streaming misspecification covariance and predictive std must be its own."""
        X, y = _misspecified_problem(n=3000, p=18)
        lib = POPSRegression(posterior="ensemble", leverage_percentile=0.0).fit(X, y)
        ours = fit_pops_streaming(
            _Chunks(X, y, k=7),
            lib.coef_,
            lib.sigma_,
            lib.alpha_,
            posterior="ensemble",
            minimum_relative_error=0.0,
        )
        scale = np.abs(lib.misspecification_sigma_).max()
        self.assertLess(np.abs(ours.sigma_miss - lib.misspecification_sigma_).max() / scale, 1e-10)
        _, std_lib = lib.predict(X[:200], return_std=True)
        np.testing.assert_allclose(ours.std(X[:200]), std_lib, rtol=1e-10, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
