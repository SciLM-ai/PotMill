"""Orchestration helpers for the __main__ pipeline: entropy worker setup, batch collection,
run-directory preparation, and futures-progress reporting."""

import concurrent.futures
import os
import shutil

from potmill.bfile import write_b_batch

STAGES = (
    "entropy",
    "labeling",
    "b_collecting",
    "featurization",
    "fitting",
    "cost",
    "pareto",
    "pops",
)
RUN_DIRS = {
    "entropy": "entropy",
    "labeling": "labeling",
    "featurize": "features",
    "fit": "fits",
    "pops": "pops",
}


def prepare_run_dirs(config, start_path):
    """Wipe and recreate the output directory of every enabled stage (skipped when resuming)."""
    if config["Main"]["resume"]:
        return
    dirs = [d for mode, d in RUN_DIRS.items() if config["Main"][mode]]
    if config["Main"]["pareto"]:
        dirs += ["costs", "pareto-front"]
    for d in dirs:
        shutil.rmtree(start_path + d, ignore_errors=True)
        os.makedirs(start_path + d)


def make_init_atoms_from_entropy(structuregen_config):
    """Create an init_function closure that captures the structuregen config.

    Each worker gets a unique executorlib_worker_id automatically from executorlib and creates its
    own subdirectory for renorm/optimizer files. RNGs are seeded per-worker for diversity across
    parallel entropy workers. Workers share Phase 1 (renormalization) results and accepted
    descriptors via a shared directory, so the global information matrix reflects all workers'
    discoveries.
    """

    def init_atoms_from_entropy(executorlib_worker_id):
        import os
        import random

        import numpy as np

        shared_dir = os.path.join(os.getcwd(), "shared")
        os.makedirs(shared_dir, exist_ok=True)
        descriptor_dir = os.path.join(shared_dir, "descriptors")
        os.makedirs(descriptor_dir, exist_ok=True)

        os.makedirs(f"worker_{executorlib_worker_id}", exist_ok=True)
        os.chdir(f"worker_{executorlib_worker_id}")

        if executorlib_worker_id > 0:
            random.seed(42 + executorlib_worker_id)
            np.random.seed(42 + executorlib_worker_id)

        worker_config = structuregen_config.copy()
        worker_config["_worker_id"] = executorlib_worker_id
        worker_config["shared_state_dir"] = shared_dir
        worker_config["shared_descriptor_dir"] = descriptor_dir

        from potmill.entropy import max_entropy_atoms_iterator

        return {"entropy_iterator": max_entropy_atoms_iterator(worker_config)}

    return init_atoms_from_entropy


def next_atoms_from_entropy(entropy_iterator, job_id=None):
    """Yield the next entropy-generated config. When job_id is given (label_batch_size>1 path)
    returns a tagged dict so uma_batch can recover the submission index after the futures resolve."""
    result = next(entropy_iterator)
    if job_id is not None:
        return {"atoms": result, "job_id": job_id}
    return result


def combine_b(start_path, labeling_results, labeling_IDs_ready_for_fit, batch_idx):
    """Build this batch's b_batch.csv from the in-memory labeling targets (row-aligned with
    features/{batch_idx}/<rcut>/a.npy for foldfit) and append it to the cumulative b{N}.csv (for the
    row engine). No per-config b files are read -- the targets travel in the labeling result dicts."""
    # Batched labeling (label_batch_size>1) returns a list per task, so exe.batched yields a
    # list-of-lists; flatten back to a flat list of result dicts.
    if labeling_results and isinstance(labeling_results[0], list):
        labeling_results = [item for sublist in labeling_results for item in sublist]
    labeling_IDs_finished = [r["job_ID"] for r in labeling_results]
    print("Starting b.csv file preparation for the fit...", flush=True)
    new_ready = labeling_IDs_ready_for_fit + labeling_IDs_finished
    len1, len2 = len(labeling_IDs_ready_for_fit), len(new_ready)
    os.makedirs(f"{start_path}features/{batch_idx}", exist_ok=True)
    write_b_batch(
        f"{start_path}features/{batch_idx}/b_batch.csv", [r["b_rows"] for r in labeling_results]
    )
    os.system(
        f"cat {start_path}features/b{len1}.csv {start_path}features/{batch_idx}/b_batch.csv "
        f"> {start_path}features/b{len2}.csv"
    )
    return new_ready


def _count_running(futures, list_of_lists=False):
    if list_of_lists:
        return sum(1 for sub in futures for f in sub if f.running())
    return sum(1 for f in futures if f.running())


def check_and_print_status(futures, name, total, list_of_lists=False, count_multiplier=1):
    """Poll futures and print stage progress. count_multiplier > 1 means one future stands for N
    items (batched labeling), keeping printed REMAINING/FINISHED in config units like total."""
    if list_of_lists:
        for i in range(len(futures)):
            if len(futures[i]) != 0:
                done, futures[i] = concurrent.futures.wait(futures[i], timeout=1)
                if len(done) != 0:
                    remaining = len(futures[i]) * count_multiplier
                    print(
                        f"{remaining} {name}S REMAINING  --- {total - remaining} {name}S FINISHED  "
                        f"--- {total} {name}S TOTAL",
                        flush=True,
                    )
                break
    else:
        done, futures = concurrent.futures.wait(futures, timeout=1)
        if len(done) != 0:
            remaining = len(futures) * count_multiplier
            print(
                f"{remaining} {name}S REMAINING  --- {total - remaining} {name}S FINISHED  --- "
                f"{total} {name}S TOTAL",
                flush=True,
            )
    return futures


def task_counts(entropy, labeling, b_collecting, featurization, fitting, cost, pareto, pops):
    """Build the {stage: count, stage_running: count} dict for ResourceMonitor.update_task_counts."""
    flat = {
        "entropy": entropy,
        "labeling": labeling,
        "b_collecting": b_collecting,
        "cost": cost,
        "pareto": pareto,
        "pops": pops,
    }
    nested = {"featurization": featurization, "fitting": fitting}
    counts = {}
    for name, futs in flat.items():
        counts[name] = len(futs)
        counts[f"{name}_running"] = _count_running(futs)
    for name, futs in nested.items():
        counts[name] = sum(len(f) for f in futs)
        counts[f"{name}_running"] = _count_running(futs, list_of_lists=True)
    return counts
