"""Uncertainty quantification for the exported potentials (POPS, streaming).

``pops.py`` is the estimator, ``artifact.py`` the ``<name>.uq.npz`` shipped beside each potential,
``stage.py`` the ``[Main] uq`` pipeline stage, and ``calculator.py`` the ASE calculator users point
at a potential directory to get ``get_uncertainty()`` alongside ``get_forces()``.
"""

from potmill.uq.artifact import calibrate, load_uq, save_uq
from potmill.uq.pops import POPSPosterior, evidence_ridge_from_gram, fit_pops_streaming

__all__ = [
    "POPSPosterior",
    "PotMillCalculator",
    "calibrate",
    "evidence_ridge_from_gram",
    "fit_pops_streaming",
    "load_uq",
    "save_uq",
]


def __getattr__(name):
    # Imported lazily: the calculator pulls in ASE, and the estimator must stay importable without it.
    if name == "PotMillCalculator":
        from potmill.uq.calculator import PotMillCalculator

        return PotMillCalculator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
