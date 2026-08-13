"""Prove an exported potential reproduces the fitted model, by running LAMMPS with it.

The fitted model's prediction for a configuration is defined by the design matrix FitSNAP builds
and the coefficients the fit produced.  With ``bzeroflag = 0`` and no stress rows, FitSNAP's rows
for one configuration are::

    row 0        energy:  [count_el/natoms, sum_atoms B_el/natoms, ...]   ->  b = E_total/natoms
    rows 1..3N   forces:  [0, dB/dr, ...]                                 ->  b = F component

so the model predicts ``E_total = natoms * (a_energy @ beta)`` and ``F = a_force @ beta`` exactly.
This module computes that reference, runs LAMMPS with the written potential on the same structure,
and compares -- energy and every force component.  Any real discrepancy is a bug in the export,
not something to be explained away.
"""

import contextlib
import copy
import os
import tempfile

import numpy as np

from potmill.fitting.fit import _feature_indices
from potmill.tools import lmaxes_to_string, nmaxes_to_string


@contextlib.contextmanager
def _in_dir(path):
    """``featurize`` and LAMMPS both resolve paths against the working directory."""
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(prev)


def sample_structures(start_path, n=3):
    """A few labeled structures from a run, for the cross-check. Raises if none can be found --
    the verification is never quietly skipped."""
    import glob

    from ase.io import read

    start_path = start_path.rstrip("/") + "/"
    out = []
    for path in sorted(glob.glob(start_path + "labeling/labeled_*.traj")):
        for atoms in read(path, index=f":{n - len(out)}"):
            atoms.calc = None
            out.append(atoms)
        if len(out) >= n:
            return out[:n]
    for path in sorted(glob.glob(start_path + "labeling/*/atoms_*.traj"))[: n - len(out)]:
        atoms = read(path, index=0)
        atoms.calc = None
        out.append(atoms)
    if not out:
        raise FileNotFoundError(
            f"No labeled structures under {start_path}labeling/ to verify against (stop)"
        )
    return out[:n]


def reference_ef(
    atoms, config_dict, fitsnap_config, combo, beta, full_nmax, full_lmax, workdir=None
):
    """``(E_total, forces)`` from the FITTED MODEL: a fresh FitSNAP featurization of this structure
    at the full swept basis, column-filtered exactly as the fit filtered it, times beta."""
    from potmill.featurization.featurize import featurize

    workdir = workdir or tempfile.mkdtemp()
    os.makedirs(workdir, exist_ok=True)
    fc = copy.deepcopy(fitsnap_config)
    fc["ACE"]["nmax"] = nmaxes_to_string(full_nmax)
    fc["ACE"]["lmax"] = lmaxes_to_string(full_lmax)

    probe = atoms.copy()
    probe.calc = None  # a labeled calculator lacks stress and would trip the ASE scraper
    prev = os.getcwd()
    try:
        names = featurize([probe], config_dict, fc, list(combo["rcuts"]), workdir)
    finally:
        os.chdir(prev)  # featurize chdirs into its feature directory and does not return

    a_matrix = np.load(os.path.join(workdir, "a.npy"))
    indices = _feature_indices(
        "ACE", names, [list(combo["rcuts"]), list(combo["nmaxes"]), list(combo["lmaxes"])]
    )
    if len(indices) != len(beta):
        raise ValueError(f"selected columns {len(indices)} != beta length {len(beta)} (stop)")
    a_sel = a_matrix[:, indices]
    natoms = len(probe)
    if a_sel.shape[0] < 1 + 3 * natoms:
        raise ValueError(
            f"design matrix has {a_sel.shape[0]} rows, expected at least {1 + 3 * natoms} "
            f"(1 energy + 3N force) (stop)"
        )
    energy = float(natoms * (a_sel[0] @ beta))
    forces = (a_sel[1 : 1 + 3 * natoms] @ beta).reshape(natoms, 3)
    return energy, forces, names


def lammps_ef(atoms, pot_dir, mod_filename, elements, units="metal", atom_style="atomic"):
    """``(E_total, forces)`` from LAMMPS actually running the exported potential."""
    from ase.io import write
    from lammps import lammps

    with _in_dir(pot_dir):  # the .mod references its .yace by bare filename
        with tempfile.TemporaryDirectory(dir=pot_dir) as tmp:
            data_file = os.path.join(tmp, "structure.data")
            write(data_file, atoms, format="lammps-data", specorder=list(elements), masses=True)
            lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
            try:
                lmp.commands_list(
                    [
                        f"units {units}",
                        f"atom_style {atom_style}",
                        "atom_modify map array",
                        "boundary p p p",
                        f"read_data {data_file}",
                        f"include {mod_filename}",
                        "run 0",
                    ]
                )
                energy = lmp.get_thermo("pe")
                natoms = lmp.get_natoms()
                # gather_atoms returns exactly natoms values ordered by atom ID (= the order ASE
                # wrote them). extract_atom("f") must NOT be used here: its array is dimensioned
                # [0:nmax] and includes ghost atoms, so it is longer than the structure.
                forces = np.array(lmp.gather_atoms("f", 1, 3)).reshape(natoms, 3)
            finally:
                lmp.close()
    return float(energy), forces


def compare(
    atoms_list,
    pot_dir,
    name,
    elements,
    config_dict,
    fitsnap_config,
    combo,
    beta,
    full_nmax,
    full_lmax,
    units="metal",
    atom_style="atomic",
):
    """Reference vs LAMMPS over several structures. Returns per-structure and worst-case errors."""
    per_structure = []
    for i, atoms in enumerate(atoms_list):
        ref_e, ref_f, _ = reference_ef(
            atoms, config_dict, fitsnap_config, combo, beta, full_nmax, full_lmax
        )
        lmp_e, lmp_f = lammps_ef(
            atoms, pot_dir, f"{name}.mod", elements, units=units, atom_style=atom_style
        )
        de = abs(lmp_e - ref_e)
        df = float(np.max(np.abs(lmp_f - ref_f))) if len(ref_f) else 0.0
        per_structure.append(
            {
                "i": i,
                "natoms": len(atoms),
                "ref_E": ref_e,
                "lmp_E": lmp_e,
                "dE": de,
                "dE_per_atom": de / len(atoms),
                "max_dF": df,
                "max_absF": float(np.max(np.abs(ref_f))) if len(ref_f) else 0.0,
            }
        )
    return {
        "per_structure": per_structure,
        "max_dE_per_atom": max(s["dE_per_atom"] for s in per_structure),
        "max_dF": max(s["max_dF"] for s in per_structure),
    }


def verify_written(run_dir, result, n_structures=3, md_steps=0, tolerance=1e-6):
    """Cross-check every potential ``export_potentials`` wrote, against the fitted model.

    Shared by the ``[ourPotential] verify`` pipeline stage and the ``--verify`` CLI flag. Returns
    one record per potential; ``ok`` is False if energies or forces disagree beyond ``tolerance``
    (eV/atom and eV/A -- the agreement measured on real runs is ~1e-13, so this is a huge margin).
    """
    from potmill.config import ConfigManager, load_fitsnap_config

    run_dir = os.path.abspath(run_dir) + "/"
    config = ConfigManager(run_dir + "config.ini")
    fitsnap_config = load_fitsnap_config(run_dir + config["FitSNAP"]["filename"])
    hp = config["ourHyperparameters"]
    full_nmax = hp["max_nmax"] if isinstance(hp["max_nmax"], list) else [hp["max_nmax"]]
    full_lmax = hp["max_lmax"] if isinstance(hp["max_lmax"], list) else [hp["max_lmax"]]
    reference = fitsnap_config.get("REFERENCE", {})
    units = str(reference.get("units", "metal")).strip()
    atom_style = str(reference.get("atom_style", "atomic")).strip()
    structures = sample_structures(run_dir, n=n_structures)

    records = []
    for entry in result["written"]:
        res = compare(
            structures,
            entry["dir"],
            entry["name"],
            entry["elements"],
            config.as_dict,
            fitsnap_config,
            entry["combo"],
            entry["beta"],
            full_nmax,
            full_lmax,
            units=units,
            atom_style=atom_style,
        )
        ok = res["max_dE_per_atom"] < tolerance and res["max_dF"] < tolerance
        record = {
            "name": entry["name"],
            "max_dE_per_atom": res["max_dE_per_atom"],
            "max_dF": res["max_dF"],
            "n_structures": len(structures),
            "ok": ok,
        }
        print(
            f"{'VERIFY' if ok else 'ERROR: VERIFY FAILED'} {entry['name']}: max |dE|/atom = "
            f"{res['max_dE_per_atom']:.3e} eV, max |dF| = {res['max_dF']:.3e} eV/A over "
            f"{len(structures)} structures",
            flush=True,
        )
        if md_steps:
            record["md"] = md_stability(
                structures[0],
                entry["dir"],
                entry["name"],
                entry["elements"],
                nsteps=md_steps,
                units=units,
                atom_style=atom_style,
            )
            print(
                f"VERIFY {entry['name']}: MD {md_steps} steps, drift "
                f"{record['md']['drift_per_atom_per_ps']:.3e} eV/atom/ps, T_final "
                f"{record['md']['final_temperature']:.1f} K",
                flush=True,
            )
        records.append(record)
    return records


def md_stability(
    atoms,
    pot_dir,
    name,
    elements,
    nsteps=2000,
    timestep=0.0005,
    temperature=300.0,
    units="metal",
    atom_style="atomic",
    seed=12345,
):
    """Run NVE MD with the exported potential. Returns total-energy drift per atom per ps.

    A potential that evaluates correctly at a single point but is unusable for dynamics (bad
    derivatives, NaNs, exploding forces) fails here.
    """
    from ase.io import write
    from lammps import lammps

    with _in_dir(pot_dir):
        with tempfile.TemporaryDirectory(dir=pot_dir) as tmp:
            data_file = os.path.join(tmp, "md.data")
            write(data_file, atoms, format="lammps-data", specorder=list(elements), masses=True)
            lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
            try:
                lmp.commands_list(
                    [
                        f"units {units}",
                        f"atom_style {atom_style}",
                        "atom_modify map array",
                        "boundary p p p",
                        f"read_data {data_file}",
                        f"include {name}.mod",
                        f"velocity all create {temperature} {seed} rot yes mom yes",
                        f"timestep {timestep}",
                        "fix 1 all nve",
                        "thermo 100",
                        "run 0",
                    ]
                )
                e0 = lmp.get_thermo("etotal")
                lmp.command(f"run {nsteps}")
                e1 = lmp.get_thermo("etotal")
                temp = lmp.get_thermo("temp")
            finally:
                lmp.close()
    picoseconds = nsteps * timestep
    return {
        "nsteps": nsteps,
        "etotal_start": float(e0),
        "etotal_end": float(e1),
        "drift_per_atom_per_ps": float((e1 - e0) / len(atoms) / picoseconds),
        "final_temperature": float(temp),
    }
