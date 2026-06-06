"""Labeling backends. The backend is selected by [ourLabeling] calculator and configured by the
matching passthrough section ([FairChemCalculator] / [Vasp] / [LAMMPS]); make_labeling() returns
everything __main__ needs to wire the block-allocated labeling executor.

Each backend's init_function returns a dict whose keys (calc / predictor / vasp_kwargs / ...) are
auto-injected by executorlib into the label function by matching parameter names, so all per-config
label functions share the signature ``(start_path, atoms, job_id, dirpath, <injected>)``.
"""

from collections import namedtuple

from potmill.labeling.uma import (
    make_init_uma_calculator, make_init_uma_predictor, uma, uma_batch)
from potmill.labeling.vasp import make_init_vasp, vasp
from potmill.labeling.lammps import make_init_lammps, lammps
from potmill.labeling.fake import fake_vasp

Labeling = namedtuple("Labeling", ["init_function", "per_config", "batched"])


def _fairchem_kwargs(config):
    kwargs = dict(config.get("FairChemCalculator", {}))
    kwargs.setdefault("name", "uma-m-1p1")
    kwargs.setdefault("task_name", "omat")
    kwargs.setdefault("device", "cuda")
    return kwargs


def make_labeling(config):
    """Return the Labeling(init_function, per_config, batched) for the configured backend."""
    name = config.get_value("ourLabeling", "calculator", "FairChemCalculator")
    batched = config["MAIN"]["label_batch_size"] > 1
    if name == "FairChemCalculator":
        kwargs = _fairchem_kwargs(config)
        init = make_init_uma_predictor(kwargs) if batched else make_init_uma_calculator(kwargs)
        return Labeling(init, uma, uma_batch)
    if name == "Vasp":
        return Labeling(make_init_vasp(config), vasp, None)
    if name == "LAMMPS":
        return Labeling(make_init_lammps(config), lammps, None)
    raise ValueError(f"Unknown [ourLabeling] calculator '{name}' "
                     f"(supported: FairChemCalculator, Vasp, LAMMPS)")
