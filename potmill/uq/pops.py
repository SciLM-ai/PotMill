"""POPS misspecification uncertainty, computed in one streaming pass.

POPS (Swinburne et al., "Uncertainty quantification for misspecified machine learned interatomic
potentials", npj Comput Mater 2025 / arXiv:2502.07104) targets exactly our regime: near-deterministic
reference data where the dominant error is not noise but the fact that a linear ACE model cannot
represent the true energy surface. For each training point it asks which parameter perturbation
would make the model match that point exactly, and the SPREAD of those pointwise-optimal parameters
is the misspecification uncertainty.

Why this is reimplemented here rather than calling ``popsregression`` on our data:

1. **Scale.** The library forms an ``n x p`` matrix of pointwise corrections. At 100k configurations
   and p = 1254 that is not storable, and it would reload the cumulative design matrix -- the O(N^2)
   pattern the incremental fit exists to avoid. Every quantity POPS needs is an accumulation of
   per-row rank-1 terms, so it streams in O(p^2) memory:

       theta_i = (r_i / h_i) * Sigma_s x_i        with  h_i = x_i^T Sigma_s x_i,  r_i = y_i - x_i^T beta
       C = sum_i theta_i theta_i^T = Sigma_s [ sum_i w_i x_i x_i^T ] Sigma_s,   w_i = (r_i / h_i)^2

   The bracket is another weighted Gram matrix -- accumulable batch by batch exactly like the fit's
   own R-factors.

2. **Anchoring.** The library refits BayesianRidge internally, so its uncertainty describes ITS
   coefficients. Ours must describe the coefficients we actually ship, or the error bars belong to a
   different potential than the one the user runs.

Validated against ``popsregression`` itself in ``tests/test_uq.py``: given the same anchor, the two
agree to ~1e-10.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class POPSPosterior:
    """The shippable UQ object: the POPS hypercube itself, not samples drawn from it.

    Storing the BOX (``projections`` p x k, ``low``/``high`` k) rather than an ensemble is smaller,
    deterministic, and strictly more informative -- everything the ensemble was for can be derived
    from it exactly:

    * ``sigma`` uses the box's exact second moment, so it does not carry Monte-Carlo noise. Measured:
      a 500-member ensemble already reproduces it to 0.1% (0.61312 vs 0.61197 analytic) and 10 000
      members change nothing -- so sampling buys no accuracy here, only a seed-dependent answer.
    * ``bounds`` maximizes a LINEAR functional over a box, which is analytic (pick each component's
      favourable corner). Sampling under-reaches the corners: bracket coverage measured 91.1% at 100
      samples, 95.4% at 500, 98.2% at 10 000 -- i.e. the sampled bracket's coverage was partly a
      statement about the sample count. The corner bound is the honest worst case over the set.
    * ``sample(n)`` still materializes an ensemble on demand, for anyone who wants a committee of
      potentials to run simulations with (the paper's own use) -- nothing is lost by not storing one.

    The box is NOT the cheaper option: at p = 1236 with k = 1162 active modes, ``projections`` is
    5.75 MB against 2.47 MB for a 500-member ensemble. That is the price of exactness, and it sits
    beside a ``sigma_epi`` of the same order (6.11 MB), so the artifact is ~11 MB either way.
    """

    sigma_epi: np.ndarray  # p x p  -- parameter (epistemic) covariance
    projections: np.ndarray | None  # p x k -- principal directions of the correction cloud
    low: np.ndarray | None  # k -- box lower bounds in that basis
    high: np.ndarray | None  # k -- box upper bounds
    sigma_miss_direct: np.ndarray | None  # p x p -- used by posterior='ensemble' (no box)
    posterior: str
    n_rows_used: int
    n_rows_total: int

    @property
    def sigma_miss(self):
        """Misspecification covariance: the box's EXACT second moment about the fitted parameters.

        For u uniform on [low, high]: E[u u^T] = diag((high-low)^2 / 12) + m m^T with m the box
        centre -- no sampling needed.
        """
        if self.projections is None:
            return self.sigma_miss_direct
        centre = 0.5 * (self.low + self.high)
        second_moment = np.diag((self.high - self.low) ** 2 / 12.0) + np.outer(centre, centre)
        return self.projections @ second_moment @ self.projections.T

    @property
    def sigma_total(self):
        return self.sigma_epi + self.sigma_miss

    def sample(self, n_samples=500, seed=0):
        """Draw an ensemble of parameter perturbations from the box (p x n)."""
        if self.projections is None:
            raise ValueError("this posterior carries no box (posterior='ensemble') (stop)")
        unit = np.random.default_rng(seed).uniform(size=(self.low.size, n_samples))
        return self.projections @ (self.low[:, None] + (self.high - self.low)[:, None] * unit)

    def _factor(self):
        """``F`` with ``F F^T = Sigma_total``, so ``sigma(x) = ||F^T x||``.

        Cached, and built by eigendecomposition rather than Cholesky because Sigma_miss is only
        positive SEMI-definite (the hypercube spans k <= p active modes). Using a factor turns the
        per-row quadratic form into one BLAS matmul: 140 s -> a few seconds for 100k structures at
        p = 759, which matters the moment anyone screens a trajectory.
        """
        if getattr(self, "_factor_cache", None) is None:
            eigvals, eigvecs = np.linalg.eigh(self.sigma_total)
            self._factor_cache = eigvecs * np.sqrt(np.maximum(eigvals, 0.0))
        return self._factor_cache

    def std(self, X):
        """Predictive standard deviation for design-matrix rows ``X`` (n x p)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        projected = X @ self._factor()
        return np.sqrt(np.einsum("ij,ij->i", projected, projected))

    def bounds(self, X):
        """``(low, high)`` prediction offsets over the WHOLE box -- the exact worst-case bracket.

        A linear prediction over a box is extremized at its corners, component by component, so this
        is analytic and needs no ensemble: for c = P^T x, the extremes are sum_k min/max(c_k low_k,
        c_k high_k). Add them to the point prediction to bracket it.
        """
        if self.projections is None:
            raise ValueError("this posterior carries no box (posterior='ensemble') (stop)")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        coeff = X @ self.projections  # n x k
        lo_term = coeff * self.low
        hi_term = coeff * self.high
        return (
            np.minimum(lo_term, hi_term).sum(axis=1),
            np.maximum(lo_term, hi_term).sum(axis=1),
        )


def evidence_ridge_from_gram(gram, xty, yty, n_samples, max_iter=300, tol=1e-3):
    """BayesianRidge's evidence maximization, driven by SUFFICIENT STATISTICS instead of the data.

    ``sklearn`` runs this from an SVD of the ``n x p`` design matrix, but every update needs only the
    eigenvalues of ``X^T X``, ``X^T y`` and ``y^T y`` -- all of which our fit already accumulates in
    its augmented R-factor. So the Bayesian half of POPS costs one p x p eigendecomposition and no
    data pass at all.

    Iterating on the RATIO ``rho = lambda / alpha`` rather than on the two hyperparameters
    separately is not a stylistic choice: the ACE design matrices here have condition numbers around
    1e18, so forming ``alpha * eigvals`` overflows float64 outright (observed on a real 100k fit).
    Every quantity the evidence update needs depends on ``rho`` alone, and ``alpha * sigma`` -- the
    only thing POPS consumes -- is independent of ``alpha`` entirely.

    Returns ``(coef, sigma, alpha, lambda_)`` with ``sigma = (alpha X^T X + lambda I)^-1``.
    """
    gram = np.asarray(gram, dtype=float)
    xty = np.asarray(xty, dtype=float)
    eigvals, eigvecs = np.linalg.eigh(gram)
    eigvals = np.maximum(eigvals, 0.0)
    scale = float(eigvals.max())
    if scale <= 0:
        raise ValueError("Gram matrix is zero -- no data to fit (stop)")
    proj = eigvecs.T @ xty  # X^T y in the eigenbasis

    floor = np.finfo(float).eps * scale  # below this, directions are numerical noise
    rho = 1e-6 * scale
    alpha = 1.0
    for _ in range(max_iter):
        denom = eigvals + rho
        coef_eig = proj / denom
        rss = max(
            yty - 2.0 * float(coef_eig @ proj) + float(coef_eig @ (eigvals * coef_eig)),
            np.finfo(float).eps * max(yty, 1.0),
        )
        gamma = float(np.sum(eigvals / denom))
        lambda_new = gamma / max(float(coef_eig @ coef_eig), np.finfo(float).tiny)
        alpha = max((n_samples - gamma) / rss, np.finfo(float).tiny)
        rho_new = float(np.clip(lambda_new / alpha, floor, 1e12 * scale))
        converged = abs(rho_new - rho) / max(rho, floor) < tol
        rho = rho_new
        if converged:
            break

    denom = eigvals + rho
    coef = eigvecs @ (proj / denom)
    # sigma = (alpha X^T X + lambda I)^-1; alpha * sigma (what POPS uses) is alpha-free
    sigma = (eigvecs / (alpha * denom)) @ eigvecs.T
    return coef, sigma, alpha, alpha * rho


def _row_statistics(rows, targets, beta, scaled_sigma):
    """Per-row residual, leverage and correction weight -- the only per-row quantities POPS needs."""
    residual = targets - rows @ beta
    correction = rows @ scaled_sigma  # n x p
    leverage = np.einsum("ij,ij->i", correction, rows)
    safe = np.where(leverage > 0, leverage, np.inf)
    return residual, leverage, correction * (residual / safe)[:, None]


def _kept_corrections(batches, beta, scaled_sigma, threshold):
    """Per batch, the pointwise corrections of the rows that clear the residual threshold.

    One generator for all three passes (residual scale, correction Gram, hypercube bounds) so the
    selection rule cannot drift between them.
    """
    for rows, targets in batches:
        rows_arr = np.asarray(rows, float)
        targets_arr = np.asarray(targets, float)
        _, _, correction = _row_statistics(rows_arr, targets_arr, beta, scaled_sigma)
        keep = np.abs(targets_arr - rows_arr @ beta) >= threshold
        if np.any(keep):
            yield correction[keep]


def fit_pops_streaming(
    batches,
    beta,
    sigma_epi,
    alpha,
    posterior="hypercube",
    minimum_relative_error=0.01,
    percentile_clipping=0.0,
    mode_threshold=1e-8,
):
    """POPS posterior from an iterable of ``(rows, targets)`` batches, never materializing them all.

    ``batches`` may be re-iterated (the hypercube posterior needs a second pass for the bounding box,
    since the principal directions are unknown until the first pass finishes). Selection follows the
    library's current criterion: a row contributes only if ``|residual| >= minimum_relative_error *
    RMSE``, which targets the points the model actually fits badly -- the signature of
    misspecification -- rather than the high-leverage points its earlier versions used.
    """
    scaled_sigma = alpha * np.asarray(sigma_epi, dtype=float)
    beta = np.asarray(beta, dtype=float)
    p = beta.size

    # ---- pass 1: residual scale, then the correction Gram matrix -------------------------------
    sq_sum, n_total = 0.0, 0
    for rows, targets in batches:
        residual = np.asarray(targets, float) - np.asarray(rows, float) @ beta
        sq_sum += float(residual @ residual)
        n_total += residual.size
    rmse = np.sqrt(sq_sum / max(n_total, 1))
    threshold = minimum_relative_error * rmse

    correction_gram = np.zeros((p, p))
    n_used = 0
    for kept in _kept_corrections(batches, beta, scaled_sigma, threshold):
        correction_gram += kept.T @ kept
        n_used += kept.shape[0]
    if n_used == 0:
        raise ValueError(
            f"no training row has |residual| >= {minimum_relative_error} x RMSE ({threshold:.3e}) "
            f"-- the model fits every point exactly, so POPS has no misspecification to measure (stop)"
        )

    if posterior == "ensemble":
        return POPSPosterior(
            sigma_epi=np.asarray(sigma_epi, float),
            projections=None,
            low=None,
            high=None,
            sigma_miss_direct=correction_gram / n_used,
            posterior="ensemble",
            n_rows_used=n_used,
            n_rows_total=n_total,
        )
    if posterior != "hypercube":
        raise ValueError(f"posterior must be 'ensemble' or 'hypercube', got '{posterior}' (stop)")

    # ---- hypercube: principal directions from pass 1, then their extent in pass 2 ---------------
    eigvals, eigvecs = np.linalg.eigh(correction_gram)
    active = eigvals > mode_threshold * eigvals.max()
    projections = eigvecs[:, active]  # p x k

    if percentile_clipping > 0:
        # A quantile box needs the projected values themselves; k is small (active modes only), so
        # this is O(n_used x k), not O(n x p).
        projected = np.concatenate(
            [
                kept @ projections
                for kept in _kept_corrections(batches, beta, scaled_sigma, threshold)
            ]
        )
        low = np.percentile(projected, percentile_clipping, axis=0)
        high = np.percentile(projected, 100.0 - percentile_clipping, axis=0)
    else:
        # Pure min/max (the library default): running extrema, no storage.
        low = np.full(projections.shape[1], np.inf)
        high = np.full(projections.shape[1], -np.inf)
        for kept in _kept_corrections(batches, beta, scaled_sigma, threshold):
            proj = kept @ projections
            low = np.minimum(low, proj.min(axis=0))
            high = np.maximum(high, proj.max(axis=0))

    return POPSPosterior(
        sigma_epi=np.asarray(sigma_epi, float),
        projections=projections,
        low=low,
        high=high,
        sigma_miss_direct=None,
        posterior="hypercube",
        n_rows_used=n_used,
        n_rows_total=n_total,
    )
