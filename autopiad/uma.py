import numpy as np
import os
import time as _time
from ase import Atoms
from ase.io import read, write

_UMA_FIRST_CALL_DONE = False
_UMA_INIT_DONE_TS = None
_UMA_LAST_RETURN_TS = None
_UMA_CALL_COUNT = 0


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
    """executorlib init_function: load UMA model once per GPU worker (for batched inference
    via uma_batch -- avoids ASE calculator's per-config overhead by reusing the predict_unit
    across many configs in one .predict(batch) call)."""
    global _UMA_INIT_DONE_TS
    _t0 = _time.time()
    from fairchem.core.calculate.pretrained_mlip import get_predict_unit
    predictor = get_predict_unit("uma-m-1p1", device="cuda")
    _UMA_INIT_DONE_TS = _time.time()
    print(f"HANDOFF_TIMING: init_uma_predictor DONE pid={os.getpid()} wall_clock={_UMA_INIT_DONE_TS:.3f} init_secs={_UMA_INIT_DONE_TS-_t0:.3f}", flush=True)
    return {"predictor": predictor}


def uma(start_path, input_file, job_id, first_index, dirpath, calc):
    """Per-config UMA labeling. Prints LABEL_TIMING with per-phase wall time so the
    bottleneck (GPU forward vs ASE traj write vs b savetxt vs idle/ZMQ) is measurable
    across worker count + node count from grep'ing labeling/flux_*.out files."""
    global _UMA_FIRST_CALL_DONE, _UMA_LAST_RETURN_TS, _UMA_CALL_COUNT
    _t_enter = _time.time()
    _t_idle = (_t_enter - _UMA_LAST_RETURN_TS) if _UMA_LAST_RETURN_TS else 0.0
    if not _UMA_FIRST_CALL_DONE:
        _UMA_FIRST_CALL_DONE = True
        _idle = (_t_enter - _UMA_INIT_DONE_TS) if _UMA_INIT_DONE_TS else -1
        print(f"HANDOFF_TIMING: first uma() call pid={os.getpid()} wall_clock={_t_enter:.3f} idle_since_init={_idle:.3f}s job_id={job_id}", flush=True)

    _t0 = _time.time()
    os.chdir(dirpath)
    if isinstance(input_file, Atoms):
        atoms = input_file
    else:
        atoms = read(start_path+input_file, index=0, format='vasp')
    atoms.pbc = True
    atoms.calc = calc
    n_atoms = len(atoms)
    _t_input = _time.time() - _t0

    _t0 = _time.time()
    ener = atoms.get_potential_energy()
    _t_energy = _time.time() - _t0

    _t0 = _time.time()
    forces = atoms.get_forces().ravel()
    _t_force = _time.time() - _t0

    _t0 = _time.time()
    b = np.vstack([np.arange(first_index,first_index+1+3*n_atoms),
                    np.full(1+3*n_atoms,job_id),
                    np.concatenate([np.array([ener])/n_atoms,forces])]).T
    np.savetxt("b", b, delimiter=',', fmt=['%i','%i','%.10f'])
    _t_write_b = _time.time() - _t0

    _t0 = _time.time()
    write(f"atoms_{job_id}.traj", images=atoms, format='traj')
    _t_write_traj = _time.time() - _t0

    atoms.calc = None
    _t_total = _time.time() - _t_enter
    _UMA_LAST_RETURN_TS = _time.time()
    _UMA_CALL_COUNT += 1
    print(f"LABEL_TIMING pid={os.getpid()} call={_UMA_CALL_COUNT} job={job_id} natoms={n_atoms} "
          f"t_idle={_t_idle*1000:.1f} t_input={_t_input*1000:.1f} "
          f"t_energy={_t_energy*1000:.1f} t_force={_t_force*1000:.1f} "
          f"t_write_b={_t_write_b*1000:.1f} t_write_traj={_t_write_traj*1000:.1f} "
          f"t_total={_t_total*1000:.1f}", flush=True)

    return {"job_ID":job_id, "atoms":atoms}


_UMA_BATCH_FIRST_CALL_DONE = False
_UMA_BATCH_CALL_COUNT = 0
_UMA_BATCH_LAST_RETURN_TS = None


def uma_batch(start_path, atoms_list, job_ids, labeling_dir, predictor):
    """Batch inference: process N structures in a single GPU forward pass.

    Per-call breakdown is logged as LABEL_BATCH_TIMING so per-config cost can be compared
    to single-config uma() (LABEL_TIMING). UMA's forward has ~160 ms fixed overhead and
    only ~1 ms / atom of compute, so batches of 16-32 amortize the overhead 10x+."""
    global _UMA_BATCH_FIRST_CALL_DONE, _UMA_BATCH_CALL_COUNT, _UMA_BATCH_LAST_RETURN_TS
    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

    _t_enter = _time.time()
    _t_idle = (_t_enter - _UMA_BATCH_LAST_RETURN_TS) if _UMA_BATCH_LAST_RETURN_TS else 0.0
    if not _UMA_BATCH_FIRST_CALL_DONE:
        _UMA_BATCH_FIRST_CALL_DONE = True
        _idle = (_t_enter - _UMA_INIT_DONE_TS) if _UMA_INIT_DONE_TS else -1
        print(f"HANDOFF_TIMING: first uma_batch() call pid={os.getpid()} wall_clock={_t_enter:.3f} idle_since_init={_idle:.3f}s n={len(atoms_list)}", flush=True)

    # If items are tagged dicts ({"atoms":..., "job_id":...}) and job_ids is None, extract.
    if job_ids is None:
        job_ids = [item["job_id"] if isinstance(item, dict) else None for item in atoms_list]

    _t0 = _time.time()
    resolved = []
    data_list = []
    for item in atoms_list:
        atoms = item if isinstance(item, Atoms) else item["atoms"]
        atoms.pbc = True
        atoms.calc = None
        data_list.append(AtomicData.from_ase(atoms, task_name="omat"))
        resolved.append(atoms)
    _t_input = _time.time() - _t0

    _t0 = _time.time()
    batch = atomicdata_list_to_batch(data_list)
    _t_collate = _time.time() - _t0

    _t0 = _time.time()
    preds = predictor.predict(batch)
    _t_energy = _time.time() - _t0

    _t0 = _time.time()
    energies = preds["energy"].detach().cpu().numpy()
    forces = preds["forces"].detach().cpu().numpy()
    _t_d2h = _time.time() - _t0

    natoms_list = [len(a) for a in resolved]
    total_atoms = sum(natoms_list)

    _t0 = _time.time()
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
    _t_write = _time.time() - _t0
    _t_total = _time.time() - _t_enter

    _UMA_BATCH_LAST_RETURN_TS = _time.time()
    _UMA_BATCH_CALL_COUNT += 1
    print(f"LABEL_BATCH_TIMING pid={os.getpid()} call={_UMA_BATCH_CALL_COUNT} batch_n={len(atoms_list)} "
          f"total_atoms={total_atoms} t_idle={_t_idle*1000:.1f} t_input={_t_input*1000:.1f} "
          f"t_collate={_t_collate*1000:.1f} t_energy={_t_energy*1000:.1f} t_d2h={_t_d2h*1000:.1f} "
          f"t_write={_t_write*1000:.1f} t_total={_t_total*1000:.1f} "
          f"per_config_ms={_t_total*1000/len(atoms_list):.1f}", flush=True)
    return results
