"""The ``<name>.uq.npz`` shipped beside each potential, and how to read it back.

The file carries the POPS hypercube rather than an ensemble sampled from it (see
``POPSPosterior``): smaller, deterministic, and enough to derive the standard deviation exactly, the
worst-case bracket exactly, and an ensemble of any size on demand.

Everything is float32. A standard deviation does not need sixteen digits, and at p = 1254 the
epistemic covariance alone is 12.6 MB in float64 -- against a 0.34 MB potential. What the file must
never lose is provenance: which columns it belongs to, what it was calibrated against, and how much
of the variance the epistemic term actually carried, so a reader can tell whether the approximations
below were reasonable for their potential.
"""

import numpy as np

FORMAT_VERSION = 1


def save_uq(path, posterior, beta, column_indices, calibration, provenance):
    """Write ``<name>.uq.npz``. ``calibration`` and ``provenance`` are flat dicts of scalars/strings."""
    payload = {
        "format_version": np.int32(FORMAT_VERSION),
        "beta": np.asarray(beta, dtype=np.float64),  # must match the .yace exactly -> float64
        "column_indices": np.asarray(column_indices, dtype=np.int32),
        "sigma_epi": np.asarray(posterior.sigma_epi, dtype=np.float32),
        "posterior": str(posterior.posterior),
        "n_rows_used": np.int64(posterior.n_rows_used),
        "n_rows_total": np.int64(posterior.n_rows_total),
    }
    if posterior.projections is not None:
        payload |= {
            "projections": np.asarray(posterior.projections, dtype=np.float32),
            "low": np.asarray(posterior.low, dtype=np.float32),
            "high": np.asarray(posterior.high, dtype=np.float32),
        }
    else:
        payload["sigma_miss_direct"] = np.asarray(posterior.sigma_miss, dtype=np.float32)
    for key, value in (calibration | provenance).items():
        payload[key] = np.asarray(value)
    np.savez_compressed(path, **payload)
    return path


def load_uq(path):
    """Read a ``.uq.npz`` back into a ``(POPSPosterior, beta, column_indices, meta)`` tuple."""
    from potmill.uq.pops import POPSPosterior

    z = np.load(path, allow_pickle=False)
    version = int(z["format_version"])
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{path} is UQ format version {version}, this PotMill writes/reads {FORMAT_VERSION} "
            f"(stop, do not guess the layout)"
        )
    has_box = "projections" in z
    posterior = POPSPosterior(
        sigma_epi=z["sigma_epi"].astype(np.float64),
        projections=z["projections"].astype(np.float64) if has_box else None,
        low=z["low"].astype(np.float64) if has_box else None,
        high=z["high"].astype(np.float64) if has_box else None,
        sigma_miss_direct=(None if has_box else z["sigma_miss_direct"].astype(np.float64)),
        posterior=str(z["posterior"]),
        n_rows_used=int(z["n_rows_used"]),
        n_rows_total=int(z["n_rows_total"]),
    )
    meta = {
        k: (z[k].item() if z[k].ndim == 0 else z[k])
        for k in z.files
        if k
        not in {
            "beta",
            "column_indices",
            "sigma_epi",
            "projections",
            "low",
            "high",
            "sigma_miss_direct",
        }
    }
    return posterior, z["beta"], z["column_indices"], meta


def calibrate(sigma, errors, levels=(0.68, 0.95)):
    """Split-conformal scale factors: quantiles of ``|error| / sigma`` on HELD-OUT data.

    Multiplying ``sigma`` by ``q_level`` gives an interval that contains that fraction of held-out
    errors, which is what turns a raw model spread into "+/- X eV/atom, ~68% of structures". The raw
    coverage is recorded alongside so the size of the correction is visible rather than absorbed.
    """
    sigma = np.asarray(sigma, dtype=float)
    errors = np.asarray(errors, dtype=float)
    usable = sigma > 0
    if not np.any(usable):
        raise ValueError("every sigma is zero -- nothing to calibrate against (stop)")
    ratio = errors[usable] / sigma[usable]
    out = {
        "calib_n": np.int64(int(usable.sum())),
        # One number, not one per level: the raw coverage is "how many errors the UNSCALED sigma
        # already covered", which has nothing to do with the level being calibrated for.
        "raw_coverage": np.float64(np.mean(ratio <= 1.0)),
    }
    for level in levels:
        out[f"calib_q{int(round(level * 100))}"] = np.float64(np.quantile(ratio, level))
    return out
