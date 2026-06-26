"""UMA (fairchem) labeling backend, configured via the [FAIRChemCalculator] section."""

import os

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from ase.io.trajectory import Trajectory

from potmill.bfile import b_rows


def make_init_uma_calculator(kwargs):
    """executorlib init_function (per-config path): load a FAIRChemCalculator once per worker."""

    def init_uma_calculator():
        from fairchem.core import FAIRChemCalculator

        calc = FAIRChemCalculator.from_model_checkpoint(
            kwargs["name"], task_name=kwargs["task_name"], device=kwargs["device"]
        )
        return {"calc": calc}

    return init_uma_calculator


def make_init_uma_predictor(kwargs):
    """executorlib init_function (batched path): load a predict_unit once per worker so uma_batch
    can amortize UMA's ~160 ms fixed forward overhead across label_batch_size configs."""

    def init_uma_predictor():
        from fairchem.core.calculate.pretrained_mlip import get_predict_unit

        return {
            "predictor": get_predict_unit(kwargs["name"], device=kwargs["device"]),
            "task_name": kwargs["task_name"],
        }

    return init_uma_predictor


def uma(start_path, input_file, job_id, dirpath, calc):
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


def uma_batch(start_path, atoms_list, job_ids, labeling_dir, predictor, task_name="omat"):
    """Batch inference: process N structures in one GPU forward pass. Returns a LIST of N dicts."""
    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

    # items are {"atoms":..., "job_id":...} dicts (tagged in __main__ for the batched path)
    if job_ids is None:
        job_ids = [item["job_id"] if isinstance(item, dict) else None for item in atoms_list]

    resolved, data_list = [], []
    for item in atoms_list:
        atoms = item if isinstance(item, Atoms) else item["atoms"]
        atoms.pbc = True
        atoms.calc = None
        data_list.append(AtomicData.from_ase(atoms, task_name=task_name))
        resolved.append(atoms)

    preds = predictor.predict(atomicdata_list_to_batch(data_list))
    energies = preds["energy"].detach().cpu().numpy()
    forces = preds["forces"].detach().cpu().numpy()

    results, labeled = [], []
    offset = 0
    for i, (atoms, job_id) in enumerate(zip(resolved, job_ids, strict=False)):
        n_atoms = len(atoms)
        f = forces[offset : offset + n_atoms]
        offset += n_atoms
        energy = float(energies[i])
        rows = b_rows(job_id, energy, n_atoms, f)
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=f)
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
