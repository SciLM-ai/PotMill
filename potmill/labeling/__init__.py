"""Labeling backends. The backend is selected by [ourLabeling] calculator and configured by the
matching passthrough section ([FAIRChemCalculator] / [Vasp] / [LAMMPS]); make_labeling() returns
everything __main__ needs to wire the block-allocated labeling executor.

Each backend's init_function returns a dict whose keys (calc / predictor / vasp_kwargs / ...) are
auto-injected by executorlib into the label function by matching parameter names, so all per-config
label functions share the signature ``(start_path, atoms, job_id, dirpath, <injected>)``.
"""

from collections import namedtuple

from potmill.labeling.grace import grace, grace_batch, make_init_grace_calculator
from potmill.labeling.lammps import lammps, make_init_lammps
from potmill.labeling.uma import make_init_uma_calculator, make_init_uma_predictor, uma, uma_batch
from potmill.labeling.vasp import make_init_vasp, vasp

Labeling = namedtuple("Labeling", ["init_function", "per_config", "batched"])


def _fairchem_kwargs(config):
    kwargs = dict(config.get("FAIRChemCalculator", {}))
    kwargs.setdefault("name", "uma-m-1p1")
    kwargs.setdefault("task_name", "omat")
    kwargs.setdefault("device", "cuda")
    return kwargs


def _grace_kwargs(config):
    kwargs = dict(config.get("GRACE", {}))
    kwargs.setdefault("model", "GRACE-2L-SMAX-OMAT-large")
    kwargs.setdefault("min_dist", 0.3)
    kwargs["min_dist"] = float(kwargs["min_dist"])  # config.ini values arrive as strings
    return kwargs


def make_labeling(config):
    """Return the Labeling(init_function, per_config, batched) for the configured backend."""
    name = config.get_value("ourLabeling", "calculator", "FAIRChemCalculator")
    batched = config["ourLabeling"]["label_batch_size"] > 1
    if name == "FAIRChemCalculator":
        kwargs = _fairchem_kwargs(config)
        init = make_init_uma_predictor(kwargs) if batched else make_init_uma_calculator(kwargs)
        return Labeling(init, uma, uma_batch)
    if name == "Vasp":
        return Labeling(make_init_vasp(config), vasp, None)
    if name == "LAMMPS":
        return Labeling(make_init_lammps(config), lammps, None)
    if name == "GRACE":
        init = make_init_grace_calculator(_grace_kwargs(config))
        return Labeling(init, grace, grace_batch)
    raise ValueError(
        f"Unknown [ourLabeling] calculator '{name}' "
        "(supported: FAIRChemCalculator, Vasp, LAMMPS, GRACE)"
    )
