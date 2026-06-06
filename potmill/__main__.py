import os, pickle
from potmill.tools import create_rcut_range, rcuts_to_string, nmaxes_to_string, lmaxes_to_string, twojmaxes_to_string
from potmill.tools import hyperparameters_to_string, create_eweight_range
from potmill.tools import combined_ace_hyperparameters, combined_snap_hyperparameters, parse_inputfile, configparse
from potmill.featurize import featurize, init_featurize
from potmill.vasp import vasp
from potmill.uma import uma, uma_batch, init_uma_calculator, init_uma_predictor
from potmill.lammps import lammps
from potmill.fake_vasp import fake_vasp
from potmill.fit import fit, foldfit, init_fit
from potmill.pareto import pareto
from potmill.pops import pops
from potmill.monitor import ResourceMonitor
import flux
import concurrent.futures
import flux.job
from executorlib import FluxJobExecutor


def _count_running(futures, list_of_lists=False):
    if list_of_lists:
        return sum(1 for sub in futures for f in sub if f.running())
    return sum(1 for f in futures if f.running())


def check_and_print_status(futures, name, total, list_of_lists=False, count_multiplier=1):
    """Print stage progress. count_multiplier > 1 = one future represents N items (e.g.
    labeling batched at label_batch_size=10 -- each future = 10 configs, so the printed
    REMAINING/FINISHED stay in config units while total is in config units too)."""
    if list_of_lists:
        for i in range(len(futures)):
            if len(futures[i]) != 0:
                done, futures[i] = concurrent.futures.wait(futures[i], timeout=1)
                if len(done)!=0:
                    remaining = len(futures[i]) * count_multiplier
                    print(f"{remaining} {name}S REMAINING  --- {total-remaining} {name}S FINISHED  "
                          f"--- {total} {name}S TOTAL", flush=True)
                break
    else:
        done, futures = concurrent.futures.wait(futures, timeout=1)
        if len(done)!=0:
            remaining = len(futures) * count_multiplier
            print(f"{remaining} {name}S REMAINING  --- {total-remaining} {name}S FINISHED  --- "
                  f"{total} {name}S TOTAL", flush=True)
    return futures


def combine_b(start_path, labeling_results, labeling_IDs_ready_for_fit, batch_idx):
    # If labeling was batched (label_batch_size>1), each labeling task returns a list
    # of N dicts (from uma_batch), so exe.batched yields a list-of-lists. Flatten back
    # to list-of-dicts so the existing per-config iteration below is unchanged.
    if labeling_results and isinstance(labeling_results[0], list):
        labeling_results = [item for sublist in labeling_results for item in sublist]
    labeling_IDs_finished = [labeling_result["job_ID"] for labeling_result in labeling_results]
    print("Starting b.csv file preparation for the fit...", flush=True)
    new_b_files = " ".join([f"{job_id}/b" for job_id in labeling_IDs_finished])
    new_labeling_IDs_ready_for_fit = labeling_IDs_ready_for_fit + labeling_IDs_finished
    len1, len2 = len(labeling_IDs_ready_for_fit), len(new_labeling_IDs_ready_for_fit)
    os.system(f"cat {start_path}features/b{len1}.csv {new_b_files} > {start_path}features/b{len2}.csv")
    # Per-batch b file (this batch's configs only) for the incremental foldfit: O(batch), aligned
    # row-for-row with features/{batch_idx}/<rcut>/a.npy (same config order as featurize used).
    os.makedirs(f"{start_path}features/{batch_idx}", exist_ok=True)
    os.system(f"cat {new_b_files} > {start_path}features/{batch_idx}/b_batch.csv")
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

        from potmill.entropy import max_entropy_atoms_iterator
        return {"entropy_iterator": max_entropy_atoms_iterator(worker_config)}
    return init_atoms_from_entropy


def next_atoms_from_entropy(entropy_iterator, job_id=None):
    """Yield the next entropy-generated config. When job_id is given (label_batch_size>1 path)
    returns a tagged dict so uma_batch can recover the submission index from the resolved
    batch_futs list (since the entropy_atoms_futures elements no longer carry job_id once
    update_futures_in_input replaces them with their results)."""
    result = next(entropy_iterator)
    if job_id is not None:
        return {"atoms": result, "job_id": job_id}
    return result


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
    n_fold = config["MAIN"].get("n_fold", 3)            # k for k-fold CV (test = 1/n_fold)
    fit_engine = config["MAIN"].get("fit_engine", "incremental")  # 'incremental' (R-collecting) | 'rows'
    # label_batch_size: configs per GPU forward pass. 1 = per-config uma() with ASE calculator
    # (default, no behavioral change). >1 = uma_batch() with the get_predict_unit predictor;
    # amortizes UMA's ~160 ms fixed forward overhead across N configs, so 1 lab GPU/node
    # can keep up with entropy. Must divide batch_size evenly (we don't split a combine_b batch).
    label_batch_size = config["MAIN"].get("label_batch_size", 1)
    assert 0 < fit_gpus_per_node < gpus_per_node, \
        f"fit_gpus_per_node ({fit_gpus_per_node}) must be >0 and leave GPUs for labeling " \
        f"(gpus_per_node={gpus_per_node})"
    assert label_batch_size >= 1, f"label_batch_size must be >=1, got {label_batch_size}"
    if label_batch_size > 1:
        assert config["MAIN"]["batch_size"] % label_batch_size == 0, \
            f"batch_size ({config['MAIN']['batch_size']}) must be a multiple of " \
            f"label_batch_size ({label_batch_size}) so combine_b sees whole batches"
    n_fit_workers = fit_gpus_per_node * nnodes
    n_label_workers = (gpus_per_node - fit_gpus_per_node) * nnodes
    print(f"GPU split: {n_label_workers} labeling + {n_fit_workers} fitting workers "
          f"({gpus_per_node}/node total) | fit_device={fit_device} fit_method={fit_method} "
          f"| label_batch_size={label_batch_size}", flush=True)
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

    # WARN (do NOT auto-override -- users may have custom pair_style setups) if FitSNAP.in
    # [REFERENCE] pair_style cutoff < inputfile [RCUT] max_rcut. LAMMPS compute pace aborts
    # any featurize task with rcut > pair_style cutoff:
    #   "ERROR: Compute pace cutoff is longer than pairwise cutoff (src/ML-PACE/compute_pace.cpp:129)"
    # With restart_limit=3 on the block executors those tasks fail cleanly instead of deadlocking,
    # but the affected tasks' results are still lost -- so the user should fix FitSNAP.in.
    # See CLAUDE.md "Configuration constraints".
    import re as _re
    _ps_m = _re.match(r"\s*zero\s+([0-9.]+)", fitsnap_config.get("REFERENCE", {}).get("pair_style", ""))
    if _ps_m and float(_ps_m.group(1)) < float(config["RCUT"]["max_rcut"]):
        _ps_cut = float(_ps_m.group(1))
        _max_rcut = float(config["RCUT"]["max_rcut"])
        print(f"WARNING: FitSNAP.in [REFERENCE] pair_style cutoff ({_ps_cut}) < inputfile "
              f"[RCUT] max_rcut ({_max_rcut}). LAMMPS will abort featurize tasks with "
              f"rcut > {_ps_cut} with 'compute pace cutoff > pair_style cutoff' "
              f"(src/ML-PACE/compute_pace.cpp:129). "
              f"FIX FitSNAP.in:  pair_style = zero {_max_rcut + 0.1}", flush=True)

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
                           restart_limit=3,  # auto-restart dead workers (SaddleMill pattern) so a single worker crash doesn't deadlock _drain_dead_worker
                           resource_dict={"cores": 1, "gpus_per_core": 0, "num_nodes": 1, "threads_per_core": threads_per_worker,
                                          "cwd": start_path+"entropy", "error_log_file":"error.out"}) as entropy_exe:
        
        # label_batch_size>1 uses the get_predict_unit predictor (batched .predict path); =1
        # keeps the per-config ASE calculator (default, byte-identical to prior behavior).
        _label_init = init_uma_predictor if label_batch_size > 1 else init_uma_calculator
        with FluxJobExecutor(flux_log_files=True, max_workers=n_label_workers, flux_executor=flux_executor,
                             block_allocation=True, init_function=_label_init, restart_limit=3,
                             resource_dict={"cores": 1, "gpus_per_core": 1, "num_nodes": 1,
                                            "cwd": start_path+"labeling", "error_log_file": "error.out"}) as labeling_exe:

          with FluxJobExecutor(flux_log_files=True, max_workers=n_featurize_workers, block_allocation=True, flux_executor=flux_executor,
                               init_function=init_featurize, restart_limit=3,
                               resource_dict={"cores": ncores_per_featurization, "gpus_per_core": 0, "num_nodes": 1,
                                              "cwd": start_path+"features", "error_log_file":"error.out"}) as featurize_exe:

            with FluxJobExecutor(flux_log_files=True, max_workers=n_fit_workers, flux_executor=flux_executor,
                                 block_allocation=True, init_function=init_fit, restart_limit=3,
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
                        # When batching labels, tag the entropy result with its submission index
                        # so uma_batch can recover the job_id after exe.batched (completion-ordered)
                        # collapses futures into a list.
                        if label_batch_size > 1:
                            fs = entropy_exe.submit(next_atoms_from_entropy, job_id=i)
                        else:
                            fs = entropy_exe.submit(next_atoms_from_entropy)
                        fs.task_ = i
                        entropy_atoms_futures.append(fs)

                if labeling_mode:
                    print(f"LABELING jobs submission (label_batch_size={label_batch_size})...", flush=True)
                    if label_batch_size == 1:
                        # Per-config path: one uma() submit per entropy future (unchanged).
                        for i, entropy_atoms in enumerate(entropy_atoms_futures):
                            labeling_directory = f"{start_path}labeling/{i}/"
                            os.makedirs(labeling_directory, exist_ok=True)
                            fs = labeling_exe.submit(uma, start_path, entropy_atoms, i, 0, labeling_directory)
                            fs.task_ = i
                            labeling_futures.append(fs)
                    else:
                        # Batched path: index-order chunks of label_batch_size consecutive entropy
                        # futures. Each labeling task is submitted with a LIST of L futures as an
                        # arg -- the dependency scheduler walks args, finds each future, waits for
                        # all L to resolve, then `update_futures_in_input` materializes the list of
                        # results. Each task has its OWN small future_lst (L entries) rather than
                        # the full 100k -- avoids the O(N) per-scheduler-pass per-batched-collector
                        # cost of `exe.batched(entropy_atoms_futures, n=L)` which is O(N^2) total
                        # at nconfigurations=100k / L=20 (5000 collectors x 100k entries).
                        # uma_batch sees a list of {"atoms":..., "job_id":...} dicts and extracts
                        # both. Pass 4 positional args; executorlib injects `predictor` as a kwarg
                        # from init_uma_predictor's return dict (same pattern as uma+calc).
                        for i, batch_start in enumerate(range(0, nconfigurations, label_batch_size)):
                            batch_futs = entropy_atoms_futures[batch_start:batch_start+label_batch_size]
                            fs = labeling_exe.submit(uma_batch, start_path, batch_futs, None,
                                                     f"{start_path}labeling")
                            fs.task_ = i
                            labeling_futures.append(fs)

                    # exe.batched batches by COMPLETION order (first n done), so featurize/combine_b start
                    # on the first batch_size that FINISH -- a straggler in a fixed index-chunk would idle
                    # the rest, which is exactly what this avoids. (The O(100k)-per-collector ingestion
                    # stall that made this unusable at 100k is fixed in executorlib's dependency scheduler:
                    # batched tasks now track only their skip_lst, not the full labeling-futures list.)
                    # For the batched-labeling path, each labeling future already holds label_batch_size
                    # results -- so we batch combine_b by batch_size/label_batch_size futures (each is a list).
                    combine_b_n = batch_size if label_batch_size == 1 else batch_size // label_batch_size
                    batched_labeling_futures = exe.batched(labeling_futures, n=combine_b_n)
                    for i, batched_labeling_future in enumerate(batched_labeling_futures):
                        fs = exe.submit(combine_b, start_path, batched_labeling_future, b_futures[-1], i,
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

                if fit_mode and fit_engine == "incremental":
                    # Incremental R-collecting: one sequential chain per SUBSET (rcut,nmax,lmax).
                    # foldfit(subset, batch i) folds batch i into the subset's running per-fold
                    # state (read from disk via prev_state[s]) and emits results for all eweights
                    # x folds at this checkpoint. The chain edge = prev future threaded per subset;
                    # subsets run in parallel, dynamically scheduled across the GPU fit workers.
                    print("FITTING jobs submission (incremental R-collecting)...", flush=True)
                    eweight_range = create_eweight_range(config['EWEIGHT']["middle_eweight"],
                                                         config['EWEIGHT']["num_eweights"])
                    n_subsets = len(hyperparameters_list_noeweight)
                    prev_state = [None]*n_subsets
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
                            # block-allocated fitting_exe: no per-submit resource_dict; foldfit
                            # chdir-free (writes by absolute path). b_future is a dependency barrier
                            # (ensures features/{i}/b_batch.csv exists); prev_state[s] is the chain.
                            fs = fitting_exe.submit(foldfit, f"{start_path}features/",
                                            featurization_futures[i][rcut_idx], b_future, subset_hp,
                                            eweight_range, mlip, i, prev_state[s], n_fold=n_fold,
                                            fit_dir_base=fit_dir_base, state_dir=state_dir,
                                            fit_device=fit_device, fit_method=fit_method)
                            fs.task_ = (i,s)
                            prev_state[s] = fs
                            fitting_futures_temp.append(fs)
                        fitting_futures.append(fitting_futures_temp)

                if fit_mode and fit_engine != "incremental":
                    # Row-based reference/fallback (O(N^2) cumulative reload; fixed alignment + new CV).
                    print("FITTING jobs submission (row-based)...", flush=True)
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
                                            hyperparameters_list[j], mlip, batch_ID=i, n_fold=n_fold,
                                            fit_directory=fit_directory, fit_device=fit_device, fit_method=fit_method)
                            fs.task_ = (i,j)
                            fitting_futures_temp.append(fs)
                        fitting_futures.append(fitting_futures_temp)

                if pareto_mode:
                    print("COST jobs submission...", flush=True)
                    atoms4cost = batched_labeling_futures[0]
                    cost_nstructures = 100   # cost is only a featurization-timing probe -> use a small
                                             # slice of batch 0, not all ~batch_size configs (which made
                                             # each of the 1-per-subset cost tasks ~6.5 min and starved combine_b)
                    for i in costs:
                        hyperparams = hyperparameters_list_noeweight[i]
                        rcuts = hyperparams[0]
                        costs_directory = start_path + "costs/"
                        costs_directory += hyperparameters_to_string(mlip, hyperparams, delimiter='_', w_eweight=False)
                        os.makedirs(costs_directory, exist_ok=True)
                        fs = exe.submit(featurize, atoms4cost, config, fitsnap_config, rcuts, costs_directory,
                                        only_cost=True, hyperparameters_noeweight=hyperparams,
                                        cost_nstructures=cost_nstructures,
                                        resource_dict={"cores": 1, "gpus_per_core": 0, "num_nodes": 1,
                                                        "cwd": costs_directory, "error_log_file": "error.out",
                                                        "priority": 8})
                        fs.task_ = i
                        cost_futures.append(fs)

                    print("PARETO jobs submission...", flush=True)
                    for i, fitting_futures_per_b in enumerate(fitting_futures):
                        fs = exe.submit(pareto, start_path, i, fitting_futures_per_b, cost_futures, mlip,
                                        resource_dict={"cores": 1, "gpus_per_core": 0, "num_nodes": 1,
                                                    "cwd":start_path+"pareto-front", "error_log_file":"error.out",
                                                    "priority": 8})
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
                                        hyperparameters_list[i], mlip, batch_ID=len(b_futures)-2, n_fold=n_fold,
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
                        # When batched, each labeling future is one uma_batch task covering
                        # label_batch_size configs -- pass that as count_multiplier so the
                        # printed REMAINING/FINISHED are in CONFIG units (matches total).
                        labeling_futures = check_and_print_status(labeling_futures, "LABELING",
                                                                  nconfigurations,
                                                                  count_multiplier=label_batch_size)
                        b_futures = check_and_print_status(b_futures, "B_COLLECTING", num_b_futures)
                        if not labeling_exe_shutdown and len(labeling_futures) == 0:
                            labeling_exe.shutdown(wait=False)
                            labeling_exe_shutdown = True
                            print("LABELING EXECUTOR SHUT DOWN - resources freed", flush=True)
                            # Take over the just-freed labeling GPUs for fitting (PR #589: dynamic
                            # max_workers on block-allocated executors). Roughly halves the fit tail.
                            if fit_mode:
                                fitting_exe.max_workers = n_fit_workers + n_label_workers
                                print(f"FITTING expanded to {n_fit_workers + n_label_workers} workers "
                                      f"(claimed labeling GPUs)", flush=True)

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
