"""All-data coefficients for the exported LAMMPS potential.

The pipeline only ever SOLVES per CV fold: fold ``f``'s coefficients are fitted on the ``k-1``/``k``
of the data that is not fold ``f``'s test set.  A potential a user runs MD with should use ALL the
labeled data, so this module produces that coefficient vector -- without ever re-reading the
cumulative design matrix (the O(N^2) reload the incremental engine exists to avoid).

Incremental engine (production).  ``foldfit`` already accumulates, per fold, an augmented TSQR
R-factor of that fold's weighted TRAINING rows (``state.pt``).  Because the k test sets PARTITION
the data, every row appears in exactly ``k-1`` of the k training sets, so stacking the k R-factors
and re-triangularizing gives the R-factor of ``sqrt(k-1) x`` the full weighted design matrix::

    R_full = qr_r([R_tr(0); R_tr(1); ...; R_tr(k-1)])        # = sqrt(k-1) * R(all rows)

A least-squares solution read off an AUGMENTED R is invariant under an overall scale factor
(``R -> cR`` implies ``d -> cd`` implies ``x`` unchanged), so the ``sqrt(k-1)`` cancels and the
result is the same estimator a one-shot all-data fit would give.  The per-row weights
(``exp(-E/5)``, ``1/max(3,|f|)``) are fold-independent and already baked into the R's; only the
global normalizations must be rescaled, from the stored per-fold sums::

    Sw_full = sum_f Sw_f / (k - 1)

Validated against a direct one-shot weighted ``lstsq`` in ``tests/test_potential.py``.

Row engine (``fit_engine = rows``) has no accumulated state, but it already loads the cumulative
design matrix, so ``all_data_beta_rows`` computes the identical estimator directly from it.
"""

import os

import numpy as np


def _qr_r(mat):
    import torch

    return torch.linalg.qr(mat, mode="r").R


def _solve_augmented(R_solve, p, rcond):
    """SVD least-squares from an augmented R (``[A | b]`` triangularized) -- the same truncated
    solve ``_FoldState.solve_and_rmse`` uses, so an exported potential matches the fitted model."""
    import torch

    R = R_solve[:p, :p]
    d = R_solve[:p, p]
    U, S, Vh = torch.linalg.svd(R, full_matrices=False)
    S_inv = torch.where(rcond * S[0] < S, 1.0 / S, torch.zeros_like(S))
    return Vh.mT @ (S_inv * (U.mT @ d))


_MERGE_CACHE = {}


def merge_state(state_path):
    """Merge a subset's per-fold R-factors into the full-data solve inputs.

    Returns ``{R_E, R_F, Sw_E, Sw_F, p, n_configs}``, where ``n_configs`` is how many configurations
    this accumulator has actually eaten -- the ground truth for what the exported coefficients were
    fitted on, which matters when a run was interrupted between checkpoints.

    Cached per (path, mtime): the eweight enters only AFTER this merge, so every eweight swept for
    a subset reuses one merge instead of repeating the QR (which is the expensive part -- a
    ``k(p+1) x (p+1)`` factorization, and ``p`` reaches a few thousand columns).
    """
    import torch

    key = (os.path.abspath(state_path), os.path.getmtime(state_path))
    if key in _MERGE_CACHE:
        return _MERGE_CACHE[key]

    blobs = torch.load(state_path, map_location="cpu", weights_only=False)
    k = len(blobs)
    if k < 2:
        raise ValueError(
            f"{state_path} holds {k} fold state(s): the all-data merge needs n_fold >= 2 "
            f"(every row must appear in exactly k-1 training sets) (stop)"
        )

    parts_E = [b["solve"]["E"].to(dtype=torch.float64) for b in blobs]
    parts_F = [b["solve"]["F"].to(dtype=torch.float64) for b in blobs]
    p = parts_E[0].shape[1] - 1  # augmented: p descriptor columns + the target column
    for part in parts_E + parts_F:
        if part.shape[1] != p + 1:
            raise ValueError(
                f"{state_path}: inconsistent R widths ({part.shape[1]} vs {p + 1}) (stop)"
            )

    # Energy rows are one per configuration, and each fold splits every configuration into exactly
    # one of train/test -- so tr_E + te_E is that fold's view of ALL configurations eaten so far.
    # Every fold must therefore agree; if they do not, the state file is not self-consistent.
    per_fold = {int(b["n"][("tr", "E")]) + int(b["n"][("te", "E")]) for b in blobs}
    if len(per_fold) != 1:
        raise ValueError(
            f"{state_path}: folds disagree on how many configurations were folded in "
            f"({sorted(per_fold)}) -- the accumulated state is inconsistent (stop)"
        )

    R_E = _qr_r(torch.cat(parts_E, dim=0))
    R_F = _qr_r(torch.cat(parts_F, dim=0))
    Sw_E = sum(float(b["Sw"][("tr", "E")]) for b in blobs) / (k - 1)
    Sw_F = sum(float(b["Sw"][("tr", "F")]) for b in blobs) / (k - 1)
    if Sw_E <= 0 or Sw_F <= 0:
        raise ValueError(
            f"{state_path}: non-positive weight sums (Sw_E={Sw_E}, Sw_F={Sw_F}) (stop)"
        )

    _MERGE_CACHE.clear()  # one subset at a time; the R's are O(p^2) and p can be thousands
    _MERGE_CACHE[key] = {
        "R_E": R_E,
        "R_F": R_F,
        "Sw_E": Sw_E,
        "Sw_F": Sw_F,
        "p": p,
        "n_configs": per_fold.pop(),
    }
    return _MERGE_CACHE[key]


def state_n_configs(state_path):
    """How many configurations a subset's accumulated fit state has eaten (cached with the merge)."""
    return merge_state(state_path)["n_configs"]


def all_data_beta(state_path, eweight, rcond=1e-13):
    """Coefficients fitted on ALL labeled rows, merged from a subset's per-fold R-factors.

    ``state_path`` is a ``fits/_state/subset_<s>/state.pt`` written by ``foldfit``. Returns a 1-D
    float64 numpy array of length ``p`` (the subset's selected column count).
    """
    import torch

    merged = merge_state(state_path)
    alpha = eweight / merged["Sw_E"]
    beta_f = 1.0 / merged["Sw_F"]
    x = _solve_augmented(
        _qr_r(torch.cat([alpha * merged["R_E"], beta_f * merged["R_F"]], dim=0)),
        merged["p"],
        rcond,
    )
    if not bool(torch.all(torch.isfinite(x))):
        raise ValueError(
            f"non-finite all-data coefficients from {state_path} (rank-deficient design?) (stop)"
        )
    return x.detach().cpu().numpy().astype(np.float64)


def all_data_beta_rows(a_matrix, b_values, is_energy, eweight, rcond=1e-13):
    """The same estimator computed directly from the cumulative design matrix (row engine).

    Reproduces ``fit.fit``'s weighting with the train side taken as EVERY row: energy rows weighted
    by ``exp(-E/5)`` normalized to sum to ``eweight``, force rows by ``1/max(3,|f|)`` normalized to
    sum to 1.
    """
    a_matrix = np.asarray(a_matrix, dtype=np.float64)
    b_values = np.asarray(b_values, dtype=np.float64)
    is_energy = np.asarray(is_energy, dtype=bool)
    is_force = ~is_energy

    ew = np.exp(-b_values[is_energy] / 5)
    ew = ew / ew.sum() * eweight
    fw = 1.0 / np.clip(np.abs(b_values[is_force]), 3.0, None)
    fw = fw / fw.sum()

    a_stack = np.vstack([ew[:, None] * a_matrix[is_energy], fw[:, None] * a_matrix[is_force]])
    b_stack = np.concatenate([ew * b_values[is_energy], fw * b_values[is_force]])
    beta, *_ = np.linalg.lstsq(a_stack, b_stack, rcond=rcond)
    if not np.all(np.isfinite(beta)):
        raise ValueError("non-finite all-data coefficients from the row-engine solve (stop)")
    return beta
