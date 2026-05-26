import os, pickle
from autopiad.tools import create_rcut_range, rcuts_to_string, nmaxes_to_string, lmaxes_to_string, twojmaxes_to_string
from autopiad.tools import hyperparameters_to_string
from autopiad.tools import combined_ace_hyperparameters, combined_snap_hyperparameters, parse_inputfile, configparse
from autopiad.featurize import featurize, init_featurize
from autopiad.vasp import vasp
from autopiad.uma import uma, init_uma_calculator
from autopiad.lammps import lammps
from autopiad.fake_vasp import fake_vasp
from autopiad.fit import fit, init_fit
from autopiad.pareto import pareto
from autopiad.pops import pops
from autopiad.monitor import ResourceMonitor
import flux
import concurrent.futures
import flux.job
from executorlib import FluxJobExecutor


def _count_running(futures, list_of_lists=False):
    if list_of_lists:
        return sum(1 for sub in futures for f in sub if f.running())
    return sum(1 for f in futures if f.running())


def check_and_print_status(futures, name, total, list_of_lists=False):
    if list_of_lists:
        for i in range(len(futures)):
            if len(futures[i]) != 0:
                done, futures[i] = concurrent.futures.wait(futures[i], timeout=0.01)
                if len(done)!=0:
                    print(f"{len(futures[i])} {name}S REMAINING  --- {total-len(futures[i])} {name}S FINISHED  "
                          f"--- {total} {name}S TOTAL", flush=True)
                break
    else:
        done, futures = concurrent.futures.wait(futures, timeout=0.1)
        if len(done)!=0:
            print(f"{len(futures)} {name}S REMAINING  --- {total-len(futures)} {name}S FINISHED  --- "
                  f"{total} {name}S TOTAL", flush=True)
    return futures


def combine_b(start_path, labeling_results, labeling_IDs_ready_for_fit):
    labeling_IDs_finished = [labeling_result["job_ID"] for labeling_result in labeling_results]
    print("Starting b.csv file preparation for the fit...", flush=True)
    new_b_files = " ".join([f"{job_id}/b" for job_id in labeling_IDs_finished])
    new_labeling_IDs_ready_for_fit = labeling_IDs_ready_for_fit + labeling_IDs_finished
    len1, len2 = len(labeling_IDs_ready_for_fit), len(new_labeling_IDs_ready_for_fit)
    os.system(f"cat {start_path}features/b{len1}.csv {new_b_files} > {start_path}features/b{len2}.csv")
    return new_labeling_IDs_ready_for_fit


def make_init_atoms_from_entropy(structuregen_config):
    """Create an init_function closure that captures the structuregen config.

    Each worker gets a unique executorlib_worker_id automatically from executorlib
    and creates its own subdirectory for renorm/optimizer files. RNGs are seeded
    per-worker for diversity across parallel entropy workers.

    Workers share Phase 1 (renormalization) results and accepted descriptors
    via a shared directory, so the global information matrix reflects all
    workers' discoveries.
    """
    def init_atoms_from_entropy(executorlib_worker_id):
        import os, random
        import numpy as np

        # Set up shared directory for Phase 1 results and descriptor exchange
        shared_dir = os.path.join(os.getcwd(), "shared")
        os.makedirs(shared_dir, exist_ok=True)
        descriptor_dir = os.path.join(shared_dir, "descriptors")
        os.makedirs(descriptor_dir, exist_ok=True)

        worker_dir = f"worker_{executorlib_worker_id}"
        os.makedirs(worker_dir, exist_ok=True)
        os.chdir(worker_dir)

        if executorlib_worker_id > 0:
            random.seed(42 + executorlib_worker_id)
            np.random.seed(42 + executorlib_worker_id)

        worker_config = structuregen_config.copy()
        worker_config['_worker_id'] = executorlib_worker_id
        worker_config['shared_state_dir'] = shared_dir
        worker_config['shared_descriptor_dir'] = descriptor_dir

        from autopiad.entropy import max_entropy_atoms_iterator
        return {"entropy_iterator": max_entropy_atoms_iterator(worker_config)}
    return init_atoms_from_entropy


def next_atoms_from_entropy(entropy_iterator):
    return next(entropy_iterator)


def main():
    handle = flux.Flux()
    rs = flux.resource.status.ResourceStatusRPC(handle).get()
    rl = flux.resource.list.resource_list(handle).get()
    all_ncores = rl.all.ncores
    all_ngpus = rl.all.ngpus
    nnodes = len(list(rs.nodelist))

    print("NODELIST:", rs.nodelist, " #CORES:", all_ncores, " #GPUS:", all_ngpus, flush=True)

    monitor = ResourceMonitor(
        log_dir=os.getcwd(),
        interval=1.0,
        console_interval=10.0,
        nodelist=str(rs.nodelist),
        n_nodes=len(list(rs.nodelist)),
    )

    start_path = os.getcwd()+'/'
    config = parse_inputfile(start_path+"inputfile")
    fitsnap_config = configparse(start_path + config['FitSNAP']['filename'])
    fitsnap_config = {section: dict(fitsnap_config.items(section)) for section in fitsnap_config.sections()}

    mlip = config["FitSNAP"]["mlip"]
    resume_mode = config["MAIN"]["resume"]
    entropy_mode = config["MAIN"]["entropy"]
    feature_mode = config["MAIN"]["featurize"]
    labeling_mode = config["MAIN"]["labeling"]
    fit_mode = config["MAIN"]["fit"]
    pareto_mode = config["MAIN"]["pareto"]
    pops_mode = config["MAIN"]["pops"]
    nconfigurations = config["MAIN"]["nconfigurations"]
    batch_size = config["MAIN"]["batch_size"]
    ncores_per_fit = config["MAIN"]["ncores_per_fit"]
    auto_reduce_hps = config["MAIN"]["auto_reduce_hyperparameters"]
    # GPU split for labeling vs fitting (config-driven; defaults to 2 fit GPUs/node, GPU QR).
    gpus_per_node = all_ngpus // nnodes if nnodes else 0
    fit_gpus_per_node = config["MAIN"].get("fit_gpus_per_node", 2)
    fit_device = config["MAIN"].get("fit_device", "cuda")
    fit_method = config["MAIN"].get("fit_method", "svd")
    assert 0 < fit_gpus_per_node < gpus_per_node, \
        f"fit_gpus_per_node ({fit_gpus_per_node}) must be >0 and leave GPUs for labeling " \
        f"(gpus_per_node={gpus_per_node})"
    n_fit_workers = fit_gpus_per_node * nnodes
    n_label_workers = (gpus_per_node - fit_gpus_per_node) * nnodes
    print(f"GPU split: {n_label_workers} labeling + {n_fit_workers} fitting workers "
          f"({gpus_per_node}/node total) | fit_device={fit_device} fit_method={fit_method}", flush=True)
    structuregen_config = config.get("STRUCTUREGEN", {})
    if "elements" not in structuregen_config:
        # Fall back to chem_elem from FitSNAP section for backwards compatibility
        structuregen_config["elements"] = config["FitSNAP"]["chem_elem"]
    # Parallel entropy worker configuration
    n_entropy_workers = structuregen_config.get("n_entropy_workers", 1) * nnodes  # inputfile = per-node; total = per-node * nnodes
    threads_per_worker = max(1, 32 // n_entropy_workers)
    structuregen_config["n_threads"] = threads_per_worker
    if n_entropy_workers > 1 and structuregen_config.get("strict_entropy_decrease", 0):
        print("WARNING: strict_entropy_decrease forced to 0 for parallel entropy workers", flush=True)
        structuregen_config["strict_entropy_decrease"] = 0

    rcuts_list = create_rcut_range(config["RCUT"]["min_rcut"],config["RCUT"]["max_rcut"],config["RCUT"]["num_rcut"])
    if mlip == "ACE":
        hyperparameters_list = combined_ace_hyperparameters(config)
        hyperparameters_list_noeweight = combined_ace_hyperparameters(config, w_eweight=False)
        fitsnap_config["ACE"]["nmax"] = nmaxes_to_string(config["NMAX"]["max_nmax"])
        fitsnap_config["ACE"]["lmax"] = lmaxes_to_string(config["LMAX"]["max_lmax"])
    elif mlip == "SNAP":
        hyperparameters_list = combined_snap_hyperparameters(config)
        hyperparameters_list_noeweight = combined_snap_hyperparameters(config, w_eweight=False)
        fitsnap_config["BISPECTRUM"]["twojmax"] = twojmaxes_to_string(config["TWOJMAX"]["max_twojmax"])

    if not resume_mode and entropy_mode:
        os.system("rm -rf "+start_path+"entropy")
        os.mkdir(start_path+"entropy")
    if not resume_mode and labeling_mode:
        os.system("rm -rf "+start_path+"labeling")
        os.mkdir(start_path+"labeling")
    if not resume_mode and feature_mode:
        os.system("rm -rf "+start_path+"features")
        os.mkdir(start_path+"features")
    if not resume_mode and fit_mode:
        os.system("rm -rf "+start_path+"fits")
        os.mkdir(start_path+"fits")
    if not resume_mode and pareto_mode:
        os.system("rm -rf "+start_path+"costs")
        os.mkdir(start_path+"costs")
        os.system("rm -rf "+start_path+"pareto-front")
        os.mkdir(start_path+"pareto-front")
    if not resume_mode and pops_mode:
        os.system("rm -rf "+start_path+"pops")
        os.mkdir(start_path+"pops")

    if resume_mode and os.path.isfile("checkpoint.pkl"):
        with open("checkpoint.pkl", "rb") as f:
            (featurizations, fits, costs) = pickle.load(f)
    elif resume_mode:
        raise NotImplementedError("Resuming without checkpoint file is not implemented yet")
    else:
        featurizations = [i for i in range(len(rcuts_list))] if feature_mode else []
        fits = [i for i in range(len(hyperparameters_list))] if fit_mode else []
        costs = [i for i in range(len(hyperparameters_list_noeweight))] if pareto_mode else []

    entropy_atoms_futures = []
    featurization_futures = []
    labeling_futures = []
    b_futures = [[]]  # [[]] is not a bug it is for b_futures[-1] to work
    fitting_futures = []
    cost_futures = []
    pareto_futures = []
    pops_futures = []

    ncores_per_featurization = 4
    # featurize is throughput-bound; allow >1 worker/node (config knob, default 1/node).
    n_featurize_workers = config["MAIN"].get("featurize_workers_per_node", 1) * nnodes
    print(f"Featurize: {n_featurize_workers} workers ({n_featurize_workers//nnodes}/node) x {ncores_per_featurization} cores", flush=True)

    with monitor, flux.job.FluxExecutor() as flux_executor:

      with FluxJobExecutor(flux_log_files=True, max_workers=n_entropy_workers, flux_executor=flux_executor,
                           block_allocation=True, init_function=make_init_atoms_from_entropy(structuregen_config), 
                           resource_dict={"cores": 1, "gpus_per_core": 0, "num_nodes": 1, "threads_per_core": threads_per_worker,
                                          "cwd": start_path+"entropy", "error_log_file":"error.out"}) as entropy_exe:
        
        with FluxJobExecutor(flux_log_files=True, max_workers=n_label_workers, flux_executor=flux_executor,
                             block_allocation=True, init_function=init_uma_calculator,
                             resource_dict={"cores": 1, "gpus_per_core": 1, "num_nodes": 1,
                                            "cwd": start_path+"labeling", "error_log_file": "error.out"}) as labeling_exe:

          with FluxJobExecutor(flux_log_files=True, max_workers=n_featurize_workers, block_allocation=True, flux_executor=flux_executor,
                               init_function=init_featurize,
                               resource_dict={"cores": ncores_per_featurization, "gpus_per_core": 0, "num_nodes": 1,
                                              "cwd": start_path+"features", "error_log_file":"error.out"}) as featurize_exe:

            with FluxJobExecutor(flux_log_files=True, max_workers=n_fit_workers, flux_executor=flux_executor,
                                 block_allocation=True, init_function=init_fit,
                                 resource_dict={"cores": 1, "threads_per_core": ncores_per_fit,
                                                "gpus_per_core": 1, "num_nodes": 1,
                                                "cwd": start_path+"fits", "error_log_file": "error.out"}) as fitting_exe, \
                 FluxJobExecutor(flux_log_files=True, flux_executor=flux_executor) as exe:

                # Give block-allocated workers time to submit their Flux jobs.
                # Without this, the main thread's rapid task submissions hold the GIL,
                # starving worker threads that need GIL time to call flux_executor.submit().
                import time
                time.sleep(30)

                if entropy_mode:
                    print("Entropy jobs submission...", flush=True)
                    for i in range(nconfigurations):
                        fs = entropy_exe.submit(next_atoms_from_entropy)
                        fs.task_ = i
                        entropy_atoms_futures.append(fs)

                if labeling_mode:
                    print("LABELING jobs submission...", flush=True)
                    for i, entropy_atoms in enumerate(entropy_atoms_futures):
                        labeling_directory = f"{start_path}labeling/{i}/"
                        os.makedirs(labeling_directory, exist_ok=True)
                        fs = labeling_exe.submit(uma, start_path, entropy_atoms, i, 0, labeling_directory)
                        fs.task_ = i
                        labeling_futures.append(fs)

                    batched_labeling_futures = exe.batched(labeling_futures, n=batch_size)
                    for i, batched_labeling_future in enumerate(batched_labeling_futures):
                        fs = exe.submit(combine_b, start_path, batched_labeling_future, b_futures[-1],
                                        resource_dict={"cores": 1, "cwd": start_path+"labeling",
                                                        "error_log_file": "error.out"})
                        fs.task_ = i
                        b_futures.append(fs)

                if feature_mode:
                    print("FEATURIZATION jobs submission...", flush=True)
                    print(f"Number of cores allocated for featurization step is {ncores_per_featurization}", flush=True)
                    for i, batched_labeling_future in enumerate(batched_labeling_futures):
                        featurization_futures_temp = []
                        for j in featurizations:
                            rcuts = rcuts_list[j]
                            feature_directory = f"{start_path}features/{i}/{rcuts_to_string(rcuts, delimiter='_')}"
                            os.makedirs(feature_directory, exist_ok=True)
                            fs = featurize_exe.submit(featurize, batched_labeling_future, config, fitsnap_config, rcuts, feature_directory)
                            fs.task_ = (i,j)
                            featurization_futures_temp.append(fs)
                        featurization_futures.append(featurization_futures_temp)

                if fit_mode:
                    print("FITTING jobs submission...", flush=True)
                    for i, b_future in enumerate(b_futures[1:]):
                        fitting_futures_temp = []
                        for j in fits:
                            rcut_idx = rcuts_list.index(hyperparameters_list[j][0])
                            fit_directory = f"{start_path}fits/{i}/"
                            fit_directory += hyperparameters_to_string(mlip, hyperparameters_list[j], delimiter='_')
                            os.makedirs(fit_directory, exist_ok=True)
                            # fitting_exe is block-allocated (fixed cwd/resources): no per-submit
                            # resource_dict; fit() chdir's to fit_directory itself (like uma()).
                            fs = fitting_exe.submit(fit, f"{start_path}features/", featurization_futures[i][rcut_idx], b_future,
                                            hyperparameters_list[j], mlip, batch_ID=i,
                                            fit_directory=fit_directory, fit_device=fit_device, fit_method=fit_method)
                            fs.task_ = (i,j)
                            fitting_futures_temp.append(fs)
                        fitting_futures.append(fitting_futures_temp)

                if pareto_mode:
                    print("COST jobs submission...", flush=True)
                    atoms4cost = batched_labeling_futures[0]
                    for i in costs:
                        hyperparams = hyperparameters_list_noeweight[i]
                        rcuts = hyperparams[0]
                        costs_directory = start_path + "costs/"
                        costs_directory += hyperparameters_to_string(mlip, hyperparams, delimiter='_', w_eweight=False)
                        os.makedirs(costs_directory, exist_ok=True)
                        fs = exe.submit(featurize, atoms4cost, config, fitsnap_config, rcuts, costs_directory,
                                        only_cost=True, hyperparameters_noeweight=hyperparams,
                                        resource_dict={"cores": 1, "gpus_per_core": 0, "num_nodes": 1,
                                                        "cwd": costs_directory, "error_log_file": "error.out"})
                        fs.task_ = i
                        cost_futures.append(fs)

                    print("PARETO jobs submission...", flush=True)
                    for i, fitting_futures_per_b in enumerate(fitting_futures):
                        fs = exe.submit(pareto, start_path, i, fitting_futures_per_b, cost_futures, mlip,
                                        resource_dict={"cores": 1, "gpus_per_core": 0, "num_nodes": 1,
                                                    "cwd":start_path+"pareto-front", "error_log_file":"error.out"})
                        fs.task_ = i
                        pareto_futures.append(fs)

                if pops_mode:
                    print("UNCERTAINTY QUANTIFICATION jobs submission...", flush=True)
                    for i in fits:
                        rcut_idx = rcuts_list.index(hyperparameters_list[i][0])
                        pops_directory = f"{start_path}pops/"
                        pops_directory += hyperparameters_to_string(mlip, hyperparameters_list[i], delimiter='_')
                        os.makedirs(pops_directory, exist_ok=True)
                        fs = exe.submit(pops, start_path+"features/", featurization_futures[-1][rcut_idx], b_futures[-1],
                                        hyperparameters_list[i], mlip, batch_ID=len(b_futures)-2,
                                        resource_dict={"cores": 1, "threads_per_core": ncores_per_fit,
                                                    "gpus_per_core": 0, "num_nodes": 1, "cwd": pops_directory,
                                                    "error_log_file":"error.out"})
                        fs.task_ = i
                        pops_futures.append(fs)

                b_futures = b_futures[1:]
                num_b_futures = len(b_futures)
                entropy_exe_shutdown = False
                labeling_exe_shutdown = False
                featurize_exe_shutdown = False

                monitor.update_task_counts({
                    "entropy": len(entropy_atoms_futures),
                    "entropy_running": _count_running(entropy_atoms_futures),
                    "labeling": len(labeling_futures),
                    "labeling_running": _count_running(labeling_futures),
                    "b_collecting": len(b_futures),
                    "b_collecting_running": _count_running(b_futures),
                    "featurization": sum(len(f) for f in featurization_futures),
                    "featurization_running": _count_running(featurization_futures, list_of_lists=True),
                    "fitting": sum(len(f) for f in fitting_futures),
                    "fitting_running": _count_running(fitting_futures, list_of_lists=True),
                    "cost": len(cost_futures),
                    "cost_running": _count_running(cost_futures),
                    "pareto": len(pareto_futures),
                    "pareto_running": _count_running(pareto_futures),
                    "pops": len(pops_futures),
                    "pops_running": _count_running(pops_futures),
                })

                total_n_futures = 1  # enter loop
                while total_n_futures > 0:

                    if entropy_mode:
                        entropy_atoms_futures = check_and_print_status(entropy_atoms_futures,
                                                                        "ENTROPY", nconfigurations)
                        if not entropy_exe_shutdown and len(entropy_atoms_futures) == 0:
                            entropy_exe.shutdown(wait=False)
                            entropy_exe_shutdown = True
                            print("ENTROPY EXECUTOR SHUT DOWN - resources freed", flush=True)

                    if feature_mode:
                        featurization_futures = check_and_print_status(featurization_futures, "FEATURIZATION",
                                                                    len(featurizations), list_of_lists=True)
                        featurization_futures = [f for f in featurization_futures if len(f) > 0]
                        if not featurize_exe_shutdown and len(featurization_futures) == 0:
                            featurize_exe.shutdown(wait=False)
                            featurize_exe_shutdown = True
                            print("FEATURIZE EXECUTOR SHUT DOWN - resources freed", flush=True)

                    if labeling_mode:
                        labeling_futures = check_and_print_status(labeling_futures, "LABELING", nconfigurations)
                        b_futures = check_and_print_status(b_futures, "B_COLLECTING", num_b_futures)
                        if not labeling_exe_shutdown and len(labeling_futures) == 0:
                            labeling_exe.shutdown(wait=False)
                            labeling_exe_shutdown = True
                            print("LABELING EXECUTOR SHUT DOWN - resources freed", flush=True)

                    if fit_mode:
                        fitting_futures = check_and_print_status(fitting_futures, "FITTING", len(fits), list_of_lists=True)
                        fitting_futures = [f for f in fitting_futures if len(f) > 0]

                    if pareto_mode: cost_futures = check_and_print_status(cost_futures, "COST", len(costs))

                    if pareto_mode: pareto_futures = check_and_print_status(pareto_futures, "PARETO", num_b_futures)

                    if pops_mode: pops_futures = check_and_print_status(pops_futures, "POPS", len(fits))

                    total_n_futures = (len(entropy_atoms_futures) + len(labeling_futures) + len(b_futures) +
                                        len(featurization_futures) + len(fitting_futures) + len(cost_futures) +
                                        len(pareto_futures) + len(pops_futures))

                    monitor.update_task_counts({
                        "entropy": len(entropy_atoms_futures),
                        "entropy_running": _count_running(entropy_atoms_futures),
                        "labeling": len(labeling_futures),
                        "labeling_running": _count_running(labeling_futures),
                        "b_collecting": len(b_futures),
                        "b_collecting_running": _count_running(b_futures),
                        "featurization": sum(len(f) for f in featurization_futures),
                        "featurization_running": _count_running(featurization_futures, list_of_lists=True),
                        "fitting": sum(len(f) for f in fitting_futures),
                        "fitting_running": _count_running(fitting_futures, list_of_lists=True),
                        "cost": len(cost_futures),
                        "cost_running": _count_running(cost_futures),
                        "pareto": len(pareto_futures),
                        "pareto_running": _count_running(pareto_futures),
                        "pops": len(pops_futures),
                        "pops_running": _count_running(pops_futures),
                    })

                exe.shutdown(wait=False)

                # Workaround for executorlib exit hang: shutdown(wait=False) leaves
                # orphaned Flux jobs that flux_executor.__exit__ waits for forever.
                # shutdown(wait=True) also hangs because FluxPythonSpawner.shutdown()
                # calls future.cancel() + future.result(), but cancel doesn't actually
                # terminate the Flux MPI job. All pipeline output is complete at this
                # point so force exit is safe.
                # See: https://github.com/pyiron/executorlib/issues/XXX
                os._exit(0)


if __name__ == "__main__":
    main()
