"""GRACE (tensorpotential) labeling backend, configured via the [GRACE] section.

Loads a GRACE foundation model (e.g. GRACE-2L-SMAX-OMAT-large) as an ASE calculator once per worker
and labels each configuration's energy/forces -- a drop-in alternative to the UMA backend. GRACE is
TensorFlow-based; the worker process imports tensorpotential (never jax), so it coexists with the
jax entropy / torch fitting workers, which run in separate processes. Requires the potmill-grace
env (see $WORK/conda_envs/potmill-grace.SETUP.md)."""

import os

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from ase.io.trajectory import Trajectory

from potmill.bfile import b_rows


def make_init_grace_calculator(kwargs):
    """executorlib init_function: load a GRACE foundation-model ASE calculator once per worker."""

    def init_grace_calculator():
        from tensorpotential.calculator import grace_fm

        # min_dist below our structure-gen floor (0.6 A) so real close-contact configs are evaluated
        # as-is, not clamped by GRACE's neighbor-list guard.
        return {"calc": grace_fm(kwargs["model"], min_dist=kwargs["min_dist"])}

    return init_grace_calculator


def grace(start_path, input_file, job_id, dirpath, calc):
    atoms = (
        input_file
        if isinstance(input_file, Atoms)
        else read(start_path + input_file, index=0, format="vasp")
    )
    atoms.pbc = True
    atoms.calc = calc

    energy, forces = atoms.get_potential_energy(), atoms.get_forces()
    rows = b_rows(job_id, energy, len(atoms), forces)
    # Keep labeled structures as one trajectory per worker (appended), not one file per config.
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    atoms.info["job_id"] = int(
        job_id
    )  # self-describing labeled traj (downstream keys composition on this)
    traj = Trajectory(f"labeled_{os.getpid()}.traj", "a")
    traj.write(atoms)
    traj.close()

    atoms.calc = None
    return {"job_ID": job_id, "b_rows": rows, "atoms": atoms}


def grace_batch(start_path, atoms_list, job_ids, labeling_dir, calc):
    """Label a chunk of structures by looping the GRACE ASE calculator within one task. GRACE has no
    single-forward batch API like UMA's predict_unit, so this does NOT fuse the batch into one GPU
    pass -- it evaluates configs one at a time, but amortizes executorlib TASK overhead (one task per
    label_batch_size configs, matching the UMA batched path). Returns a LIST of N dicts."""
    if job_ids is None:
        job_ids = [item["job_id"] if isinstance(item, dict) else None for item in atoms_list]

    results, labeled = [], []
    for item, job_id in zip(atoms_list, job_ids, strict=False):
        atoms = item if isinstance(item, Atoms) else item["atoms"]
        atoms.pbc = True
        atoms.calc = calc
        energy, forces = atoms.get_potential_energy(), atoms.get_forces()
        rows = b_rows(job_id, energy, len(atoms), forces)
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        if job_id is not None:
            atoms.info["job_id"] = int(job_id)  # self-describing labeled traj (keys composition)
        labeled.append(atoms)
        results.append({"job_ID": job_id, "b_rows": rows, "atoms": atoms})

    # One labeled trajectory per worker (appended across this worker's batches), not per config.
    traj = Trajectory(f"{labeling_dir}/labeled_{os.getpid()}.traj", "a")
    for atoms in labeled:
        traj.write(atoms)
    traj.close()
    for atoms in labeled:
        atoms.calc = None
    return results
