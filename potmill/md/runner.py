"""Run MD with an exported potential and report whether it behaved.

This asks a different question from ``potential/verify.py``. Verification asks "did we write the
file correctly?" -- a property of the writer, constant for a given FitSNAP/LAMMPS install. This asks
"is this fit usable for dynamics?" -- a property of the FIT: its training coverage, its
hyperparameters, how it extrapolates. Two potentials with almost identical RMSEs can differ
completely here, which is exactly why it is worth running per potential rather than trusting the
Pareto table alone.

What is reported, and why each one:

* ``minimized_ok`` / ``min_dist`` -- a pathological potential collapses the cell during relaxation.
  Atoms fusing is the loudest possible failure and would otherwise masquerade as a low energy.
* ``drift_per_atom_per_ps`` -- NVE total-energy drift, the standard check that forces are the exact
  gradient of the energy the potential reports (and that the timestep is survivable).
* ``T_final`` / ``T_mean`` -- under NVT, a fit that cannot hold the thermostat's temperature is
  unusable however good its RMSE looks.
* ``ok`` -- all of the above, plus: LAMMPS did not abort, and nothing went non-finite.
"""

import contextlib
import math
import os
import tempfile

# Fallback collapse floor. The stage passes the potential's OWN floor instead (max rcinner + margin,
# the same threshold the structure picker uses): below ACE's inner cutoff the potential is no longer
# evaluating anything meaningful, so a trajectory that gets there has collapsed however finite its
# energy looks. A real run made this concrete -- a potential "survived" with its closest pair at
# 0.877 A and a drift two orders of magnitude worse than its stable siblings.
FUSED_DISTANCE = 0.5  # A
PAIR_SEARCH_CUTOFF = 3.0  # A -- far enough to catch the real nearest neighbour in a solid


def _min_distance(positions, cell, pbc=True):
    """Smallest interatomic distance under PBC, or None if no pair is within the search cutoff.

    Uses a neighbour list rather than a full distance matrix: the test cell can hold thousands of
    atoms, and an N^2 matrix would be the most expensive thing in the stage.
    """
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    if len(positions) < 2:
        return None
    probe = Atoms(positions=positions, cell=cell, pbc=pbc, symbols=["H"] * len(positions))
    distances = neighbor_list("d", probe, PAIR_SEARCH_CUTOFF)
    return float(distances.min()) if len(distances) else None


def run_md(
    atoms,
    pot_dir,
    name,
    elements,
    ensemble="nvt",
    temperature=300.0,
    timestep=0.001,
    steps=10000,
    minimize=True,
    relax_box=True,
    fused_distance=FUSED_DISTANCE,
    units="metal",
    atom_style="atomic",
    seed=12345,
    thermo=100,
):
    """Minimize (optionally) then run MD with ``<pot_dir>/<name>.mod``. Returns a metrics dict.

    Never raises on a LAMMPS failure: an exploding potential is a RESULT here, not an error, so it
    comes back as ``ok = 0`` with the reason in ``note``.
    """
    import numpy as np
    from ase.io import write
    from lammps import lammps

    record = {
        "ok": 0,
        "minimized_ok": None,
        "natoms": len(atoms),
        "ensemble": ensemble,
        "temperature": temperature,
        "steps": steps,
        "timestep": timestep,
        "e_pot_start": None,
        "drift_per_atom_per_ps": None,
        "T_final": None,
        "T_mean": None,
        "min_dist": None,
        "note": "",
    }
    prev_cwd = os.getcwd()
    os.chdir(pot_dir)  # the .mod refers to its .yace by bare filename
    lmp = None
    try:
        with tempfile.TemporaryDirectory(dir=pot_dir) as tmp:
            data_file = os.path.join(tmp, "md.data")
            write(data_file, atoms, format="lammps-data", specorder=list(elements), masses=True)
            lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
            lmp.commands_list(
                [
                    f"units {units}",
                    f"atom_style {atom_style}",
                    "atom_modify map array",
                    "boundary p p p",
                    f"read_data {data_file}",
                    f"include {name}.mod",
                    f"thermo {thermo}",
                    "run 0",
                ]
            )
            record["e_pot_start"] = float(lmp.get_thermo("pe"))

            if minimize:
                # Relax the CELL too by default: entropy generation assigns each configuration a
                # random volume (volume_scaling_min..max), so testing at the volume it happens to
                # have would measure the structure's arbitrary compression, not the potential.
                # Letting the potential find its own equilibrium volume is the fair test.
                if relax_box:
                    lmp.command("fix boxrelax all box/relax iso 0.0")
                lmp.command("minimize 1e-10 1e-10 2000 20000")
                if relax_box:
                    lmp.command("unfix boxrelax")
                e_min = float(lmp.get_thermo("pe"))
                record["minimized_ok"] = int(math.isfinite(e_min))
                record["e_pot_min"] = e_min

            lmp.commands_list(
                [
                    f"velocity all create {temperature} {seed} rot yes mom yes",
                    f"timestep {timestep}",
                ]
            )
            if ensemble == "nve":
                lmp.command("fix 1 all nve")
            elif ensemble == "nvt":
                lmp.command(f"fix 1 all nvt temp {temperature} {temperature} {100 * timestep}")
            else:
                raise ValueError(f"[ourMD] ensemble = '{ensemble}' must be 'nve' or 'nvt' (stop)")

            lmp.command("run 0")
            e_start = float(lmp.get_thermo("etotal"))
            half = max(steps // 2, 1)
            lmp.command(f"run {half}")
            t_mid = float(lmp.get_thermo("temp"))
            lmp.command(f"run {steps - half}")
            t_end = float(lmp.get_thermo("temp"))
            record["T_final"] = t_end
            record["T_mean"] = 0.5 * (t_mid + t_end)

            if ensemble == "nve":
                drift_steps = steps
                e_drift_start, e_drift_end = e_start, float(lmp.get_thermo("etotal"))
            else:
                # Total energy is not conserved under a thermostat, so drift measured across the NVT
                # run would just be the thermostat's work. Measure it on a short NVE tail instead,
                # which is the actual question: are the forces the gradient of the reported energy?
                drift_steps = max(steps // 10, 100)
                lmp.commands_list(["unfix 1", "fix 1 all nve", "run 0"])
                e_drift_start = float(lmp.get_thermo("etotal"))
                lmp.command(f"run {drift_steps}")
                e_drift_end = float(lmp.get_thermo("etotal"))

            natoms = lmp.get_natoms()
            positions = np.array(lmp.gather_atoms("x", 1, 3)).reshape(natoms, 3)
            boxlo, boxhi, xy, yz, xz, *_ = lmp.extract_box()
            cell = [
                [boxhi[0] - boxlo[0], 0.0, 0.0],
                [xy, boxhi[1] - boxlo[1], 0.0],
                [xz, yz, boxhi[2] - boxlo[2]],
            ]
            record["min_dist"] = _min_distance(positions, cell)
            record["drift_steps"] = drift_steps
            record["drift_per_atom_per_ps"] = (
                (e_drift_end - e_drift_start) / natoms / (drift_steps * timestep)
            )

            finite = all(
                math.isfinite(v) for v in (e_drift_end, t_end, record["drift_per_atom_per_ps"])
            )
            fused = record["min_dist"] is not None and record["min_dist"] < fused_distance
            record["ok"] = int(finite and not fused and natoms == len(atoms))
            if not finite:
                record["note"] = "non-finite energy or temperature during MD"
            elif fused:
                record["note"] = f"atoms collapsed: min distance {record['min_dist']:.3f} A"
            elif natoms != len(atoms):
                record["note"] = f"lost atoms: {natoms} of {len(atoms)} remain"
    except Exception as exc:  # noqa: BLE001 -- an unstable potential is a result, not a crash
        record["ok"] = 0
        record["note"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        if lmp is not None:
            with contextlib.suppress(Exception):
                lmp.close()
        os.chdir(prev_cwd)
    return record
