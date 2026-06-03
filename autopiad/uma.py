import numpy as np
import os
import time as _time
from ase import Atoms
from ase.io import read, write

_UMA_FIRST_CALL_DONE = False
_UMA_INIT_DONE_TS = None


def init_uma_calculator():
    """executorlib init_function: pre-load UMA calculator once per GPU worker."""
    global _UMA_INIT_DONE_TS
    _t0 = _time.time()
    from fairchem.core import FAIRChemCalculator
    calc = FAIRChemCalculator.from_model_checkpoint("uma-m-1p1", task_name="omat", device="cuda")
    _UMA_INIT_DONE_TS = _time.time()
    print(f"HANDOFF_TIMING: init_uma_calculator DONE pid={os.getpid()} wall_clock={_UMA_INIT_DONE_TS:.3f} init_secs={_UMA_INIT_DONE_TS-_t0:.3f}", flush=True)
    return {"calc": calc}


def init_uma_predictor():
    """executorlib init_function: load UMA model once per GPU worker (for batched inference)."""
    from fairchem.core.calculate.pretrained_mlip import get_predict_unit
    predictor = get_predict_unit("uma-m-1p1", device="cuda")
    return {"predictor": predictor}


def uma(start_path, input_file, job_id, first_index, dirpath, calc):
    global _UMA_FIRST_CALL_DONE
    if not _UMA_FIRST_CALL_DONE:
        _UMA_FIRST_CALL_DONE = True
        _now = _time.time()
        _idle = (_now - _UMA_INIT_DONE_TS) if _UMA_INIT_DONE_TS else -1
        print(f"HANDOFF_TIMING: first uma() call pid={os.getpid()} wall_clock={_now:.3f} idle_since_init={_idle:.3f}s job_id={job_id}", flush=True)

    os.chdir(dirpath)

    if isinstance(input_file, Atoms):
        atoms = input_file
    else:
        atoms = read(start_path+input_file, index=0, format='vasp')
    atoms.pbc = True
    atoms.calc = calc

    print("RUN DIRECTORY: ", os.getcwd(), " INPUT FILE: ", input_file, flush=True)

    ener = atoms.get_potential_energy()
    forces = atoms.get_forces().ravel()

    n_atoms = len(atoms)
    b = np.vstack([np.arange(first_index,first_index+1+3*n_atoms),
                    np.full(1+3*n_atoms,job_id),
                    np.concatenate([np.array([ener])/n_atoms,forces])]).T
    np.savetxt("b", b, delimiter=',', fmt=['%i','%i','%.10f'])

    #write the output in ASE traj format
    write(f"atoms_{job_id}.traj", images=atoms, format='traj')

    atoms.calc = None

    return {"job_ID":job_id, "atoms":atoms}


def uma_batch(start_path, atoms_list, job_ids, labeling_dir, predictor):
    """Batch inference: process N structures in a single GPU forward pass."""
    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

    resolved = []
    data_list = []
    for item in atoms_list:
        atoms = item if isinstance(item, Atoms) else item["atoms"]
        atoms.pbc = True
        atoms.calc = None
        data_list.append(AtomicData.from_ase(atoms, task_name="omat"))
        resolved.append(atoms)

    batch = atomicdata_list_to_batch(data_list)
    preds = predictor.predict(batch)

    energies = preds["energy"].detach().cpu().numpy()
    forces = preds["forces"].detach().cpu().numpy()
    natoms_list = [len(a) for a in resolved]

    results = []
    force_offset = 0
    for i, (atoms, job_id) in enumerate(zip(resolved, job_ids)):
        n_atoms = natoms_list[i]
        ener = float(energies[i])
        f = forces[force_offset:force_offset + n_atoms].ravel()
        force_offset += n_atoms

        dirpath = f"{labeling_dir}/{job_id}/"
        os.makedirs(dirpath, exist_ok=True)
        b = np.vstack([np.arange(0, 1 + 3*n_atoms),
                        np.full(1 + 3*n_atoms, job_id),
                        np.concatenate([np.array([ener])/n_atoms, f])]).T
        np.savetxt(f"{dirpath}/b", b, delimiter=',', fmt=['%i','%i','%.10f'])
        write(f"{dirpath}/atoms_{job_id}.traj", images=atoms, format='traj')
        atoms.calc = None
        results.append({"job_ID": job_id, "atoms": atoms})

    return results
