import os, pickle
from autopiad.tools import create_rcut_range, rcuts_to_string, nmaxes_to_string, lmaxes_to_string, twojmaxes_to_string
from autopiad.tools import hyperparameters_to_string
from autopiad.tools import combined_ace_hyperparameters, combined_snap_hyperparameters, parse_inputfile, configparse
from autopiad.entropy import max_entropy_atoms_iterator 
from autopiad.featurize import featurize
from autopiad.vasp import vasp
from autopiad.lammps import lammps
from autopiad.fake_vasp import fake_vasp
from autopiad.fit import fit
from autopiad.pareto import pareto
from autopiad.pops import pops
import flux
import concurrent.futures
import flux.job
from executorlib import FluxJobExecutor


def check_and_print_status(futures, name, total, list_of_lists=False):
    if list_of_lists:
        for i in range(len(futures)):
            if len(futures[i]) != 0:
                done, futures[i] = concurrent.futures.wait(futures[i], timeout=0.1)
                if len(done)!=0:
                    print(f"{len(futures[i])} {name}S REMAINING  --- {total-len(futures[i])} {name}S FINISHED  "
                          f"--- {total} {name}S TOTAL")
                break
    else:
        done, futures = concurrent.futures.wait(futures, timeout=0.1)
        if len(done)!=0:
            print(f"{len(futures)} {name}S REMAINING  --- {total-len(futures)} {name}S FINISHED  --- "
                  f"{total} {name}S TOTAL")
    return futures


def combine_b(start_path, vasp_results, vasp_IDs_ready_for_fit):
    vasp_IDs_finished = vasp_results["job_ID"]
    print("Starting b.csv file preparation for the fit...")
    new_b_files = " ".join(["vasp-em_%i/b" % job_id for job_id in vasp_IDs_finished])
    new_vasp_IDs_ready_for_fit = vasp_IDs_ready_for_fit + vasp_IDs_finished
    len1, len2 = len(vasp_IDs_ready_for_fit), len(new_vasp_IDs_ready_for_fit)
    os.system(f"cat {start_path}features/b{len1}.csv {new_b_files} > {start_path}features/b{len2}.csv")
    return new_vasp_IDs_ready_for_fit


def init_atoms_from_entropy():
    return {"entropy_iterator": max_entropy_atoms_iterator()}


def next_atoms_from_entropy(entropy_iterator):
    return next(entropy_iterator)


def main():
    handle = flux.Flux()
    rs = flux.resource.status.ResourceStatusRPC(handle).get()
    rl = flux.resource.list.resource_list(handle).get()
    all_ncores = rl.all.ncores
    all_ngpus = rl.all.ngpus

    print("NODELIST:",rs.nodelist, " #CORES:",all_ncores, " #GPUS:",all_ngpus)

    start_path = os.getcwd()+'/'
    config = parse_inputfile(start_path+"inputfile")
    fitsnap_config = configparse(start_path + config['FitSNAP']['filename'])
    fitsnap_config = {section: dict(fitsnap_config.items(section)) for section in fitsnap_config.sections()}

    mlip = config["FitSNAP"]["mlip"]
    resume_mode = config["MODE"]["resume"]
    entropy_mode = config["MODE"]["entropy"]
    feature_mode = config["MODE"]["featurize"]
    vasp_mode = config["MODE"]["vasp"]
    fit_mode = config["MODE"]["fit"]
    pareto_mode = config["MODE"]["pareto"]
    pops_mode = config["MODE"]["pops"]
    fit_freq = config["MODE"]["fit_freq"]
    ncores_per_fit = config["MODE"]["ncores_per_fit"]
    auto_reduce_hps = config["MODE"]["auto_reduce_hyperparameters"]
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
    if not resume_mode and vasp_mode:
        os.system("rm -rf "+start_path+"vasp-energy")
        os.mkdir(start_path+"vasp-energy")
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

    num_configurations = 500

    if resume_mode and os.path.isfile("checkpoint.pkl"):
        with open("checkpoint.pkl", "rb") as f:
            (featurizations, fits, costs) = pickle.load(f)
            # Here I also need to have futures, because of dependency
    elif resume_mode:  # Think about new jobs deleting the old ones, make sure it creates new directories for the new jobs
        raise NotImplementedError("Resuming without checkpoint file is not implemented yet")  
    else:
        featurizations = [i for i in range(len(rcuts_list))] if feature_mode else []
        fits = [i for i in range(len(hyperparameters_list))] if fit_mode else []
        costs = [i for i in range(len(hyperparameters_list_noeweight))] if pareto_mode else []

    if entropy_mode: entropy_atoms_futures = []
    if feature_mode: featurization_futures = []
    if vasp_mode: vasp_futures = []
    if vasp_mode: b_futures = [[]]  # [[]] is not a bug it is for b_futures[-1] to work
    if fit_mode: fitting_futures = []
    if pareto_mode: cost_futures = []
    if pareto_mode: pareto_futures = []
    if pops_mode: pops_futures = []


    with FluxJobExecutor(max_workers=all_ngpus, flux_log_files=True, cache_directory=start_path+'vasp_runs') as vasp_exe:

        with FluxJobExecutor(flux_log_files=True, cache_directory=start_path+'runs') as exe:
            
            with FluxJobExecutor(init_function=init_atoms_from_entropy, block_allocation=True, max_workers=1,
                                 resource_dict={"cores": 1, "gpus_per_core": 0, "num_nodes": 1,
                                                "cwd": start_path+"entropy", "error_log_file":"error.out"}) as entropy_exe:

                if entropy_mode:
                    print("Entropy jobs submission...")
                    for i in range(num_configurations):
                        fs = entropy_exe.submit(next_atoms_from_entropy)
                        fs.task_ = i
                        entropy_atoms_futures.append(fs)
                    # entropy_atoms_futures = [entropy_exe.submit(next_atoms_from_entropy) for _ in range(num_configurations)]

                if vasp_mode:
                    print("VASP jobs submission...")
                    for i, entropy_atoms in enumerate(entropy_atoms_futures):  # Loop over atomic configuration indices
                        vasp_directory = start_path + "vasp-energy/vasp-em_%i/"%i
                        os.makedirs(vasp_directory, exist_ok=True)
                        fs = vasp_exe.submit(vasp, start_path, entropy_atoms, i, 0,
                                             resource_dict={"cores": 1, "gpus_per_core": 0, "num_nodes": 1,
                                                            "cwd": vasp_directory, "error_log_file":"error.out"})
                        fs.task_ = i
                        vasp_futures.append(fs)

                    batched_vasp_futures = exe.batched(vasp_futures, n=fit_freq)
                    for i, batched_vasp_future in enumerate(batched_vasp_futures):
                        fs = exe.submit(combine_b, start_path, batched_vasp_future, b_futures[-1],
                                        resource_dict={"cores": 1, "cwd": start_path+"vasp-energy",
                                                    "error_log_file":"error.out"})
                        fs.task_ = i
                        b_futures.append(fs)

                if feature_mode:
                    ncores_per_featurization = int((all_ncores - all_ngpus)/len(rs.nodelist)) - 3
                    print("FEATURIZATION jobs submission...")
                    print(f"Number of cores allocated for featurization step is {ncores_per_featurization}")
                    for i, batched_vasp_future in enumerate(batched_vasp_futures):
                        # entropy_atoms = [entropy_atoms_futures[vasp_future.task_] for vasp_future in batched_vasp_future]
                        featurization_futures_temp = []
                        for j in featurizations:  # Loop over rcuts_list indices
                            rcuts = rcuts_list[j]
                            feature_directory = start_path + "features/" + rcuts_to_string(rcuts, delimiter='_')
                            os.makedirs(feature_directory, exist_ok=True)
                            fs = exe.submit(featurize, batched_vasp_future, config, fitsnap_config, rcuts, batch_ID=i,
                                            resource_dict={"cores": ncores_per_featurization, "gpus_per_core": 0,
                                                        "num_nodes": 1, "cwd": feature_directory, "error_log_file":"error.out"})
                            fs.task_ = (i,j)
                            featurization_futures_temp.append(fs)
                        featurization_futures.append(featurization_futures_temp)

                if fit_mode:
                    print("FITTING jobs submission...")
                    for i, b_future in enumerate(b_futures[1:]):  # Loop over cumulative batches of finished vasp jobs
                        fitting_futures_temp = []
                        for j in fits:  # Loop over hyperparameters_list
                            rcut_idx = rcuts_list.index(hyperparameters_list[j][0])
                            fit_directory = f"{start_path}fits/{i}/"
                            fit_directory += hyperparameters_to_string(mlip, hyperparameters_list[j], delimiter='_')
                            os.makedirs(fit_directory, exist_ok=True)
                            fs = exe.submit(fit, start_path+"features/", featurization_futures[i][rcut_idx], b_future, 
                                            hyperparameters_list[j], mlip, batch_ID=i,
                                            resource_dict={"cores": 1, "threads_per_core": ncores_per_fit, 
                                                        "gpus_per_core": 0, "num_nodes": 1, "cwd": fit_directory,
                                                        "error_log_file":"error.out"})
                            fs.task_ = (i,j)
                            fitting_futures_temp.append(fs)
                        fitting_futures.append(fitting_futures_temp)  # This is a list of per batch futures lists

                if pareto_mode:
                    print("COST jobs submission...")
                    nconfigs4cost = config["MODE"]["nconfigurations_for_cost"]
                    atoms4cost = exe.batched(entropy_atoms_futures, n=nconfigs4cost)[0]
                    for i in costs:  # Loop over hyperparameters_list_noeweight
                        hyperparams = hyperparameters_list_noeweight[i]
                        rcuts = hyperparams[0]
                        costs_directory = start_path + "costs/"
                        costs_directory += hyperparameters_to_string(mlip, hyperparams, delimiter='_', w_eweight=False)
                        os.makedirs(costs_directory, exist_ok=True)
                        fs = exe.submit(featurize, atoms4cost, config, fitsnap_config, rcuts, True, hyperparams,
                                        resource_dict={"cores": 1, "gpus_per_core": 0, "num_nodes": 1, 
                                                    "cwd": costs_directory, "error_log_file":"error.out"})
                        fs.task_ = i
                        cost_futures.append(fs)
                    
                    print("PARETO jobs submission...")
                    for i, fitting_futures_per_b in enumerate(fitting_futures):  # Loop over batches of fitting jobs
                        fs = exe.submit(pareto, start_path, i, fitting_futures_per_b, cost_futures, mlip,
                                        resource_dict={"cores": 1, "gpus_per_core": 0, "num_nodes": 1,
                                                    "cwd":start_path+"pareto-front", "error_log_file":"error.out"})
                        fs.task_ = i
                        pareto_futures.append(fs)

                if pops_mode:
                    print("UNCERTAINTY QUANTIFICATION jobs submission...")
                    for i in fits:  # Loop over hyperparameters_list
                        rcut_idx = rcuts_list.index(hyperparameters_list[i][0])
                        posp_directory = f"{start_path}pops/"
                        posp_directory += hyperparameters_to_string(mlip, hyperparameters_list[i], delimiter='_')
                        os.makedirs(posp_directory, exist_ok=True)
                        fs = exe.submit(pops, start_path+"features/", featurization_futures[-1][rcut_idx], b_futures[-1], 
                                        hyperparameters_list[i], mlip,
                                        resource_dict={"cores": 1, "threads_per_core": ncores_per_fit, 
                                                    "gpus_per_core": 0, "num_nodes": 1, "cwd": posp_directory,
                                                    "error_log_file":"error.out"})
                        fs.task_ = i
                        pops_futures.append(fs)

                b_futures = b_futures[1:]
                num_b_futures = len(b_futures)
                while len(pareto_futures):

                    if entropy_mode:
                        entropy_atoms_futures = check_and_print_status(entropy_atoms_futures,
                                                                       "ENTROPY", num_configurations)

                    if feature_mode: 
                        featurization_futures = check_and_print_status(featurization_futures, "FEATURIZATION",
                                                                       len(featurizations), list_of_lists=True)

                    if vasp_mode: vasp_futures = check_and_print_status(vasp_futures, "VASP", num_configurations)

                    if vasp_mode: b_futures = check_and_print_status(b_futures, "B_COLLECTING", num_b_futures)

                    if fit_mode: fitting_futures = check_and_print_status(fitting_futures, "FITTING", len(fits), list_of_lists=True)

                    if pareto_mode: cost_futures = check_and_print_status(cost_futures, "COST", len(costs))

                    if pareto_mode: pareto_futures = check_and_print_status(pareto_futures, "PARETO", num_b_futures)

                    # with open("checkpoint.pkl", "wb") as f:
                    #     pickle.dump((vasp_futures, featurization_futures, fitting_futures, cost_futures), f)


if __name__ == "__main__":
    main()