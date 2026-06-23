import os
import pickle
import time

import flux.job
from executorlib import FluxJobExecutor

from potmill.analysis import pareto
from potmill.config import ConfigManager, load_fitsnap_config
from potmill.featurization import featurize, init_featurize
from potmill.fitting import fit, foldfit, init_fit, pops
from potmill.labeling import make_labeling
from potmill.monitor import ResourceMonitor
from potmill.pipeline import (
    check_and_print_status,
    combine_b,
    make_init_atoms_from_entropy,
    next_atoms_from_entropy,
    prepare_run_dirs,
    task_counts,
)
from potmill.resources import query_flux, worker_layout
from potmill.tools import (
    combined_ace_hyperparameters,
    combined_snap_hyperparameters,
    create_eweight_range,
    create_rcut_range,
    hyperparameters_to_string,
    lmaxes_to_string,
    nmaxes_to_string,
    rcuts_to_string,
    twojmaxes_to_string,
)


def main():
    rs, all_ncores, all_ngpus, nnodes = query_flux()
    monitor = ResourceMonitor(
        log_dir=os.getcwd(),
        interval=1.0,
        console_interval=10.0,
        nodelist=str(rs.nodelist),
        n_nodes=nnodes,
    )

    start_path = os.getcwd() + "/"
    config = ConfigManager(start_path + "config.ini")
    fitsnap_config = load_fitsnap_config(start_path + config["FitSNAP"]["filename"])

    mlip = config["FitSNAP"]["mlip"]
    resume_mode = config["Main"]["resume"]
    entropy_mode = config["Main"]["entropy"]
    feature_mode = config["Main"]["featurize"]
    labeling_mode = config["Main"]["labeling"]
    fit_mode = config["Main"]["fit"]
    pareto_mode = config["Main"]["pareto"]
    pops_mode = config["Main"]["pops"]
    nconfigurations = config["Main"]["nconfigurations"]
    batch_size = config["Main"]["batch_size"]
    device = config["Main"]["device"]  # cuda | cpu -- drives labeling + fitting placement
    fit_device = device
    fit_method = config["ourFit"]["fit_method"]
    n_fold = config["ourFit"]["n_fold"]  # k for k-fold CV (test = 1/n_fold)
    fit_engine = config["ourFit"]["fit_engine"]  # 'incremental' (R-collecting) | 'rows'
    # label_batch_size: configs per GPU forward pass. 1 = per-config label() with an ASE calculator;
    # >1 = batched predictor path (UMA), amortizing the fixed forward overhead. Must divide batch_size.
    label_batch_size = config["ourLabeling"]["label_batch_size"]
    assert label_batch_size >= 1, f"label_batch_size must be >=1, got {label_batch_size}"
    if label_batch_size > 1:
        assert batch_size % label_batch_size == 0, (
            f"batch_size ({batch_size}) must be a multiple of label_batch_size ({label_batch_size})"
        )

    res = worker_layout(config, nnodes, all_ncores, all_ngpus)
    labeling = make_labeling(config)
    if label_batch_size > 1 and labeling.batched is None:
        raise ValueError(
            f"[ourLabeling] calculator={config.get_value('ourLabeling', 'calculator')} "
            f"has no batched path; set label_batch_size = 1"
        )
    print(
        f"device={device} | {res.n_label_workers} labeling ({res.label_cores_per_job} cores/job) "
        f"+ {res.n_fit_workers} fitting ({res.fit_cores_per_job} cores/job) workers "
        f"| fit_method={fit_method} | label_batch_size={label_batch_size}",
        flush=True,
    )

    structuregen_config = config.get("ourStructureGen", {})
    structuregen_config.setdefault("elements", config["FitSNAP"]["chem_elem"])
    structuregen_config["n_threads"] = res.entropy_cores_per_job
    if res.n_entropy_workers > 1 and structuregen_config.get("strict_entropy_decrease", 0):
        print(
            "WARNING: strict_entropy_decrease forced to 0 for parallel entropy workers", flush=True
        )
        structuregen_config["strict_entropy_decrease"] = 0

    hp = config["ourHyperparameters"]
    rcuts_list = create_rcut_range(hp["min_rcut"], hp["max_rcut"], hp["num_rcut"])
    if mlip == "ACE":
        hyperparameters_list = combined_ace_hyperparameters(config)
        hyperparameters_list_noeweight = combined_ace_hyperparameters(config, w_eweight=False)
        fitsnap_config["ACE"]["nmax"] = nmaxes_to_string(hp["max_nmax"])
        fitsnap_config["ACE"]["lmax"] = lmaxes_to_string(hp["max_lmax"])
    elif mlip == "SNAP":
        hyperparameters_list = combined_snap_hyperparameters(config)
        hyperparameters_list_noeweight = combined_snap_hyperparameters(config, w_eweight=False)
        fitsnap_config["BISPECTRUM"]["twojmax"] = twojmaxes_to_string(hp["max_twojmax"])

    config.validate(fitsnap_config)
    prepare_run_dirs(config, start_path)

    if resume_mode and os.path.isfile("checkpoint.pkl"):
        with open("checkpoint.pkl", "rb") as f:
            (featurizations, fits_idx, costs) = pickle.load(f)
    elif resume_mode:
        raise NotImplementedError("Resuming without checkpoint file is not implemented yet")
    else:
        featurizations = list(range(len(rcuts_list))) if feature_mode else []
        fits_idx = list(range(len(hyperparameters_list))) if fit_mode else []
        costs = list(range(len(hyperparameters_list_noeweight))) if pareto_mode else []

    entropy_atoms_futures = []
    featurization_futures = []
    labeling_futures = []
    b_futures = [[]]  # [[]] is not a bug it is for b_futures[-1] to work
    fitting_futures = []
    cost_futures = []
    pareto_futures = []
    pops_futures = []

    featurize_cores_per_job = res.featurize_cores_per_job
    print(
        f"Featurize: {res.n_featurize_workers} workers ({res.n_featurize_workers // nnodes}/node) "
        f"x {featurize_cores_per_job} cores",
        flush=True,
    )

    # Device-aware resource_dicts for the labeling + fitting executors. cuda: one GPU per job (the
    # job's MPI/threads run on that GPU). cpu: a single Python process given cores_per_job cores --
    # VASP launches its own MPI inside the command; torch fitting uses cores_per_job threads.
    if device == "cuda":
        labeling_resource_dict = {"cores": 1, "gpus_per_core": 1, "num_nodes": 1}
        fitting_resource_dict = {
            "cores": 1,
            "threads_per_core": res.fit_cores_per_job,
            "gpus_per_core": 1,
            "num_nodes": 1,
        }
    else:  # cpu
        # VASP labeling worker is a NESTED flux instance (flux_executor_nesting below) owning
        # labeling_cores_per_job cores; the [Vasp] `command` runs `flux run -n <labeling_cores_per_job>
        # vasp` INSIDE it, so the VASP ranks land on the worker's own cores -- flux's PMI is the one
        # the Cray binary accepts -- with the broker overlapping, i.e. no separate orchestrator core
        # (benchmarked +47% throughput at 4 ranks/job vs 12, since fewer ranks pack more concurrent
        # jobs and the wasted core is gone).
        labeling_resource_dict = {
            "cores": 1,
            "threads_per_core": res.label_cores_per_job,
            "gpus_per_core": 0,
            "num_nodes": 1,
        }
        # Fitting is in-process torch on CPU: 1 core with fit_cores_per_job threads.
        fitting_resource_dict = {
            "cores": 1,
            "threads_per_core": res.fit_cores_per_job,
            "gpus_per_core": 0,
            "num_nodes": 1,
        }
    labeling_resource_dict.update({"cwd": start_path + "labeling", "error_log_file": "error.out"})
    fitting_resource_dict.update({"cwd": start_path + "fits", "error_log_file": "error.out"})

    with monitor, flux.job.FluxExecutor() as flux_executor:
        with (
            FluxJobExecutor(
                flux_log_files=True,
                max_workers=res.n_entropy_workers,
                flux_executor=flux_executor,
                block_allocation=True,
                init_function=make_init_atoms_from_entropy(structuregen_config),
                restart_limit=3,  # auto-restart dead workers (SaddleMill pattern) so a single worker crash doesn't deadlock _drain_dead_worker
                resource_dict={
                    "cores": 1,
                    "gpus_per_core": 0,
                    "num_nodes": 1,
                    "threads_per_core": res.entropy_cores_per_job,
                    "cwd": start_path + "entropy",
                    "error_log_file": "error.out",
                },
            ) as entropy_exe
        ):
            with FluxJobExecutor(
                flux_log_files=True,
                max_workers=res.n_label_workers,
                flux_executor=flux_executor,
                block_allocation=True,
                # cpu: each worker is a nested flux instance so the [Vasp] `flux run -n N vasp`
                # launches in the worker's own cores (no orchestrator core). cuda labeling (UMA)
                # runs in-process on its GPU and needs no nesting.
                flux_executor_nesting=(device == "cpu"),
                init_function=labeling.init_function,
                restart_limit=3,
                resource_dict=labeling_resource_dict,
            ) as labeling_exe:
                with FluxJobExecutor(
                    flux_log_files=True,
                    max_workers=res.n_featurize_workers,
                    block_allocation=True,
                    flux_executor=flux_executor,
                    init_function=init_featurize,
                    restart_limit=3,
                    resource_dict={
                        "cores": featurize_cores_per_job,
                        "gpus_per_core": 0,
                        "num_nodes": 1,
                        "cwd": start_path + "features",
                        "error_log_file": "error.out",
                    },
                ) as featurize_exe:
                    with (
                        FluxJobExecutor(
                            flux_log_files=True,
                            max_workers=res.n_fit_workers,
                            flux_executor=flux_executor,
                            block_allocation=True,
                            init_function=init_fit,
                            restart_limit=3,
                            resource_dict=fitting_resource_dict,
                        ) as fitting_exe,
                        FluxJobExecutor(flux_log_files=True, flux_executor=flux_executor) as exe,
                    ):
                        # Give block-allocated workers time to submit their Flux jobs. Without this, the main
                        # thread's rapid task submissions hold the GIL, starving worker threads that need GIL
                        # time to call flux_executor.submit().
                        time.sleep(30)

                        if entropy_mode:
                            print("Entropy jobs submission...", flush=True)
                            for i in range(nconfigurations):
                                # When batching labels, tag the entropy result with its submission index so
                                # the batched labeler can recover the job_id after exe.batched reorders futures.
                                if label_batch_size > 1:
                                    fs = entropy_exe.submit(next_atoms_from_entropy, job_id=i)
                                else:
                                    fs = entropy_exe.submit(next_atoms_from_entropy)
                                fs.task_ = i
                                entropy_atoms_futures.append(fs)

                        if labeling_mode:
                            print(
                                f"LABELING jobs submission (label_batch_size={label_batch_size})...",
                                flush=True,
                            )
                            if label_batch_size == 1:
                                # Per-config path: one label task per entropy future. The backend's calc /
                                # kwargs are injected by executorlib from its init_function.
                                for i, entropy_atoms in enumerate(entropy_atoms_futures):
                                    # VASP/LAMMPS create this per-config dir themselves; UMA writes
                                    # no per-config files, so we no longer pre-create it here.
                                    labeling_directory = f"{start_path}labeling/{i}/"
                                    fs = labeling_exe.submit(
                                        labeling.per_config,
                                        start_path,
                                        entropy_atoms,
                                        i,
                                        labeling_directory,
                                    )
                                    fs.task_ = i
                                    labeling_futures.append(fs)
                            else:
                                # Batched path: index-order chunks of label_batch_size consecutive entropy
                                # futures. Each labeling task gets its OWN small future list (L entries) rather
                                # than the full nconfigurations -- avoids the O(N^2) cost of exe.batched at 100k.
                                for i, batch_start in enumerate(
                                    range(0, nconfigurations, label_batch_size)
                                ):
                                    batch_futs = entropy_atoms_futures[
                                        batch_start : batch_start + label_batch_size
                                    ]
                                    fs = labeling_exe.submit(
                                        labeling.batched,
                                        start_path,
                                        batch_futs,
                                        None,
                                        f"{start_path}labeling",
                                    )
                                    fs.task_ = i
                                    labeling_futures.append(fs)

                            # exe.batched batches by COMPLETION order (first n done), so featurize/combine_b start
                            # on the first batch_size that FINISH. For the batched-labeling path each labeling
                            # future already holds label_batch_size results, so combine_b batches by
                            # batch_size/label_batch_size futures (each a list).
                            combine_b_n = (
                                batch_size
                                if label_batch_size == 1
                                else batch_size // label_batch_size
                            )
                            batched_labeling_futures = exe.batched(labeling_futures, n=combine_b_n)
                            for i, batched_labeling_future in enumerate(batched_labeling_futures):
                                fs = exe.submit(
                                    combine_b,
                                    start_path,
                                    batched_labeling_future,
                                    b_futures[-1],
                                    i,
                                    resource_dict={
                                        "cores": 1,
                                        "cwd": start_path + "labeling",
                                        "error_log_file": "error.out",
                                    },
                                )
                                fs.task_ = i
                                b_futures.append(fs)

                        if feature_mode:
                            print("FEATURIZATION jobs submission...", flush=True)
                            for i, batched_labeling_future in enumerate(batched_labeling_futures):
                                featurization_futures_temp = []
                                for j in featurizations:
                                    rcuts = rcuts_list[j]
                                    feature_directory = f"{start_path}features/{i}/{rcuts_to_string(rcuts, delimiter='_')}"
                                    os.makedirs(feature_directory, exist_ok=True)
                                    fs = featurize_exe.submit(
                                        featurize,
                                        batched_labeling_future,
                                        config.as_dict,
                                        fitsnap_config,
                                        rcuts,
                                        feature_directory,
                                    )
                                    fs.task_ = (i, j)
                                    featurization_futures_temp.append(fs)
                                featurization_futures.append(featurization_futures_temp)

                        if fit_mode and fit_engine == "incremental":
                            # Incremental R-collecting: one sequential chain per SUBSET (rcut,nmax,lmax).
                            # foldfit(subset, batch i) folds batch i into the subset's running per-fold state
                            # and emits results for all eweights x folds at this checkpoint. The chain edge is
                            # the prev future per subset; subsets run in parallel across the GPU fit workers.
                            print(
                                "FITTING jobs submission (incremental R-collecting)...", flush=True
                            )
                            eweight_range = create_eweight_range(
                                hp["middle_eweight"],
                                hp["num_eweights"],
                            )
                            n_subsets = len(hyperparameters_list_noeweight)
                            prev_state = [None] * n_subsets
                            state_root = start_path + "fits/_state"
                            os.makedirs(state_root, exist_ok=True)
                            for i, b_future in enumerate(b_futures[1:]):
                                fitting_futures_temp = []
                                for s in range(n_subsets):
                                    subset_hp = hyperparameters_list_noeweight[s]
                                    rcut_idx = rcuts_list.index(subset_hp[0])
                                    state_dir = f"{state_root}/subset_{s}"
                                    fit_dir_base = f"{start_path}fits/{i}/"
                                    os.makedirs(fit_dir_base, exist_ok=True)
                                    fs = fitting_exe.submit(
                                        foldfit,
                                        f"{start_path}features/",
                                        featurization_futures[i][rcut_idx],
                                        b_future,
                                        subset_hp,
                                        eweight_range,
                                        mlip,
                                        i,
                                        prev_state[s],
                                        n_fold=n_fold,
                                        fit_dir_base=fit_dir_base,
                                        state_dir=state_dir,
                                        fit_device=fit_device,
                                        fit_method=fit_method,
                                    )
                                    fs.task_ = (i, s)
                                    prev_state[s] = fs
                                    fitting_futures_temp.append(fs)
                                fitting_futures.append(fitting_futures_temp)

                        if fit_mode and fit_engine != "incremental":
                            # Row-based reference/fallback (O(N^2) cumulative reload; fixed alignment + new CV).
                            print("FITTING jobs submission (row-based)...", flush=True)
                            for i, b_future in enumerate(b_futures[1:]):
                                fitting_futures_temp = []
                                for j in fits_idx:
                                    rcut_idx = rcuts_list.index(hyperparameters_list[j][0])
                                    fit_directory = f"{start_path}fits/{i}/"
                                    fit_directory += hyperparameters_to_string(
                                        mlip, hyperparameters_list[j], delimiter="_"
                                    )
                                    os.makedirs(fit_directory, exist_ok=True)
                                    fs = fitting_exe.submit(
                                        fit,
                                        f"{start_path}features/",
                                        featurization_futures[i][rcut_idx],
                                        b_future,
                                        hyperparameters_list[j],
                                        mlip,
                                        batch_ID=i,
                                        n_fold=n_fold,
                                        fit_directory=fit_directory,
                                        fit_device=fit_device,
                                        fit_method=fit_method,
                                    )
                                    fs.task_ = (i, j)
                                    fitting_futures_temp.append(fs)
                                fitting_futures.append(fitting_futures_temp)

                        if pareto_mode:
                            print("COST jobs submission...", flush=True)
                            atoms4cost = batched_labeling_futures[0]
                            cost_nstructures = (
                                100  # cost is only a featurization-timing probe -> a small slice
                            )
                            # of batch 0, not all ~batch_size configs (which starved combine_b)
                            for i in costs:
                                hyperparams = hyperparameters_list_noeweight[i]
                                rcuts = hyperparams[0]
                                costs_directory = start_path + "costs/"
                                costs_directory += hyperparameters_to_string(
                                    mlip, hyperparams, delimiter="_", w_eweight=False
                                )
                                os.makedirs(costs_directory, exist_ok=True)
                                fs = exe.submit(
                                    featurize,
                                    atoms4cost,
                                    config.as_dict,
                                    fitsnap_config,
                                    rcuts,
                                    costs_directory,
                                    only_cost=True,
                                    hyperparameters_noeweight=hyperparams,
                                    cost_nstructures=cost_nstructures,
                                    resource_dict={
                                        "cores": 1,
                                        "gpus_per_core": 0,
                                        "num_nodes": 1,
                                        "cwd": costs_directory,
                                        "error_log_file": "error.out",
                                        "priority": 8,
                                    },
                                )
                                fs.task_ = i
                                cost_futures.append(fs)

                            print("PARETO jobs submission...", flush=True)
                            for i, fitting_futures_per_b in enumerate(fitting_futures):
                                fs = exe.submit(
                                    pareto,
                                    start_path,
                                    i,
                                    fitting_futures_per_b,
                                    cost_futures,
                                    mlip,
                                    resource_dict={
                                        "cores": 1,
                                        "gpus_per_core": 0,
                                        "num_nodes": 1,
                                        "cwd": start_path + "pareto-front",
                                        "error_log_file": "error.out",
                                        "priority": 8,
                                    },
                                )
                                fs.task_ = i
                                pareto_futures.append(fs)

                        if pops_mode:
                            print("UNCERTAINTY QUANTIFICATION jobs submission...", flush=True)
                            for i in fits_idx:
                                rcut_idx = rcuts_list.index(hyperparameters_list[i][0])
                                pops_directory = f"{start_path}pops/"
                                pops_directory += hyperparameters_to_string(
                                    mlip, hyperparameters_list[i], delimiter="_"
                                )
                                os.makedirs(pops_directory, exist_ok=True)
                                fs = exe.submit(
                                    pops,
                                    start_path + "features/",
                                    featurization_futures[-1][rcut_idx],
                                    b_futures[-1],
                                    hyperparameters_list[i],
                                    mlip,
                                    batch_ID=len(b_futures) - 2,
                                    n_fold=n_fold,
                                    resource_dict={
                                        "cores": 1,
                                        "threads_per_core": res.fit_cores_per_job,
                                        "gpus_per_core": 0,
                                        "num_nodes": 1,
                                        "cwd": pops_directory,
                                        "error_log_file": "error.out",
                                    },
                                )
                                fs.task_ = i
                                pops_futures.append(fs)

                        b_futures = b_futures[1:]
                        num_b_futures = len(b_futures)
                        entropy_exe_shutdown = False
                        labeling_exe_shutdown = False
                        featurize_exe_shutdown = False

                        monitor.update_task_counts(
                            task_counts(
                                entropy_atoms_futures,
                                labeling_futures,
                                b_futures,
                                featurization_futures,
                                fitting_futures,
                                cost_futures,
                                pareto_futures,
                                pops_futures,
                            )
                        )

                        total_n_futures = 1  # enter loop
                        while total_n_futures > 0:
                            if entropy_mode:
                                entropy_atoms_futures = check_and_print_status(
                                    entropy_atoms_futures, "ENTROPY", nconfigurations
                                )
                                if not entropy_exe_shutdown and len(entropy_atoms_futures) == 0:
                                    entropy_exe.shutdown(wait=False)
                                    entropy_exe_shutdown = True
                                    print(
                                        "ENTROPY EXECUTOR SHUT DOWN - resources freed", flush=True
                                    )

                            if feature_mode:
                                featurization_futures = check_and_print_status(
                                    featurization_futures,
                                    "FEATURIZATION",
                                    len(featurizations),
                                    list_of_lists=True,
                                )
                                featurization_futures = [
                                    f for f in featurization_futures if len(f) > 0
                                ]
                                if not featurize_exe_shutdown and len(featurization_futures) == 0:
                                    featurize_exe.shutdown(wait=False)
                                    featurize_exe_shutdown = True
                                    print(
                                        "FEATURIZE EXECUTOR SHUT DOWN - resources freed", flush=True
                                    )

                            if labeling_mode:
                                # Each batched labeling future covers label_batch_size configs -- count_multiplier
                                # keeps the printed REMAINING/FINISHED in config units (matches total).
                                labeling_futures = check_and_print_status(
                                    labeling_futures,
                                    "LABELING",
                                    nconfigurations,
                                    count_multiplier=label_batch_size,
                                )
                                b_futures = check_and_print_status(
                                    b_futures, "B_COLLECTING", num_b_futures
                                )
                                if not labeling_exe_shutdown and len(labeling_futures) == 0:
                                    labeling_exe.shutdown(wait=False)
                                    labeling_exe_shutdown = True
                                    print(
                                        "LABELING EXECUTOR SHUT DOWN - resources freed", flush=True
                                    )
                                    # Take over the just-freed labeling resources for fitting via
                                    # dynamic max_workers on the block-allocated executor (mops up the
                                    # fit tail). Reclaim by RESOURCE, not worker count: add as many fit
                                    # workers as fit in what labeling freed, floored PER NODE, so no
                                    # node oversubscribes and DYNAMIC_RESERVE_CORES stay free for
                                    # combine_b/cost/pareto. cuda: a labeling worker frees 1 GPU and a
                                    # fit worker takes 1 GPU -> n_label (unchanged). cpu: cores differ
                                    # (e.g. label 4, fit 5), so divide freed cores by fit cores.
                                    if fit_mode:
                                        if device == "cuda":
                                            added_fit_workers = res.n_label_workers
                                        else:
                                            freed_cores_per_node = (
                                                res.n_label_workers // nnodes
                                            ) * res.label_cores_per_job
                                            added_fit_workers = (
                                                freed_cores_per_node // res.fit_cores_per_job
                                            ) * nnodes
                                        new_max_fit = res.n_fit_workers + added_fit_workers
                                        fitting_exe.max_workers = new_max_fit
                                        print(
                                            f"FITTING expanded to {new_max_fit} workers "
                                            f"(claimed freed labeling resources)",
                                            flush=True,
                                        )

                            if fit_mode:
                                fitting_futures = check_and_print_status(
                                    fitting_futures, "FITTING", len(fits_idx), list_of_lists=True
                                )
                                fitting_futures = [f for f in fitting_futures if len(f) > 0]

                            if pareto_mode:
                                cost_futures = check_and_print_status(
                                    cost_futures, "COST", len(costs)
                                )
                                pareto_futures = check_and_print_status(
                                    pareto_futures, "PARETO", num_b_futures
                                )

                            if pops_mode:
                                pops_futures = check_and_print_status(
                                    pops_futures, "POPS", len(fits_idx)
                                )

                            total_n_futures = (
                                len(entropy_atoms_futures)
                                + len(labeling_futures)
                                + len(b_futures)
                                + len(featurization_futures)
                                + len(fitting_futures)
                                + len(cost_futures)
                                + len(pareto_futures)
                                + len(pops_futures)
                            )

                            monitor.update_task_counts(
                                task_counts(
                                    entropy_atoms_futures,
                                    labeling_futures,
                                    b_futures,
                                    featurization_futures,
                                    fitting_futures,
                                    cost_futures,
                                    pareto_futures,
                                    pops_futures,
                                )
                            )

                        exe.shutdown(wait=False)

                        # Workaround for an executorlib exit hang: shutdown(wait=False) leaves orphaned Flux
                        # jobs that flux_executor.__exit__ waits on forever, and shutdown(wait=True) also hangs
                        # because cancel() does not terminate the Flux MPI job. All pipeline output is complete
                        # at this point so a force exit is safe.
                        os._exit(0)


if __name__ == "__main__":
    main()
