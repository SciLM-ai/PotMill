"""LAMMPS labeling backend (label with a fitted ACE potential), configured via [LAMMPS]."""

import os
import traceback

from ase import Atoms
from ase.calculators.lammpsrun import LAMMPS
from ase.io import read, write

from potmill.bfile import b_rows, write_b


def make_init_lammps(config):
    """executorlib init_function: forward the [LAMMPS] kwargs to every labeling task on this worker."""
    kwargs = dict(config.get("LAMMPS", {}))
    kwargs.setdefault("pot_file", "pot.yace")
    kwargs.setdefault("atom_types", ["Be"])
    kwargs.setdefault("command", "run_lammps_ase.sh")

    def init_lammps():
        return {"lammps_kwargs": kwargs}

    return init_lammps


def lammps(start_path, input_file, job_id, dirpath, lammps_kwargs):
    os.makedirs(dirpath, exist_ok=True)
    os.chdir(dirpath)
    os.environ["ASE_LAMMPSRUN_COMMAND"] = start_path + lammps_kwargs["command"]
    atom_types = lammps_kwargs["atom_types"]
    if isinstance(atom_types, str):
        atom_types = [atom_types]
    ace_file = start_path + lammps_kwargs["pot_file"]
    calc = LAMMPS(
        files=[ace_file],
        pair_style="pace",
        pair_coeff=["* * " + ace_file + " " + " ".join(atom_types)],
        keep_tmp_files=True,
        tmp_dir="lammps_temp",
        log_file="log.lammps",
    )

    atoms = (
        input_file
        if isinstance(input_file, Atoms)
        else read(start_path + input_file, index=0, format="vasp")
    )
    atoms.pbc = True
    atoms.calc = calc
    rows = None
    try:
        energy, forces = atoms.get_potential_energy(), atoms.get_forces()
        rows = b_rows(job_id, energy, len(atoms), forces)
        write_b("b", job_id, energy, len(atoms), forces)
        write(f"atoms_{job_id}.traj", images=atoms, format="traj")
    except Exception:
        print(f"Error while running LAMMPS or writing the output for job {job_id}", flush=True)
        traceback.print_exc()

    atoms.calc = None
    return {"job_ID": job_id, "b_rows": rows, "atoms": atoms}
