import pandas as pd
import os, copy, time
from ase.io import write
from autopiad.tools import rcuts_to_string, twojmaxes_to_string, hyperparameters_to_string, create_rcut_range
from autopiad.tools import combined_hyperparameters, parse_inputfile, configparse
from autopiad.featurize import featurize
from autopiad.vasp import vasp
from autopiad.fake_vasp import fake_vasp
from autopiad.fit import fit
from autopiad.pareto import pareto
import flux
import concurrent.futures
import flux.job
from executorlib import Executor


def main():
    handle = flux.Flux()
    rs = flux.resource.status.ResourceStatusRPC(handle).get()
    rl = flux.resource.list.resource_list(handle).get()
    all_ncores = rl.all.ncores
    all_ngpus = rl.all.ngpus

    print("NODELIST:",rs.nodelist, " #CORES:",all_ncores, " #GPUS:",all_ngpus)

    start_path = os.getcwd()+'/'
    config = parse_inputfile(start_path+"inputfile")
    rcuts_list = create_rcut_range(config["RCUT"]["min_rcut"],config["RCUT"]["max_rcut"],config["RCUT"]["num_rcut"])
    hyperparameters_list = combined_hyperparameters(config)
    fitsnap_config = configparse(start_path + config['FitSNAP']['filename'])
    fitsnap_config = {section: dict(fitsnap_config.items(section)) for section in fitsnap_config.sections()}
    fitsnap_config["BISPECTRUM"]["twojmax"] = twojmaxes_to_string(config["TWOJMAX"]["max_twojmax"])

    vasp_mode = config["MODE"]["vasp"]
    feature_mode = config["MODE"]["feature"]
    fit_mode = config["MODE"]["fit"]
    pareto_mode = config["MODE"]["pareto"]
    fit_freq = config["MODE"]["fit_freq"]
    auto_reduce_hps = config["MODE"]["auto_reduce_hyperparameters"]

    if not os.path.isdir(start_path+"features"):
        os.mkdir(start_path+"features")
    if not os.path.isdir(start_path+"fits"):
        os.mkdir(start_path+"fits")
    if not os.path.isdir(start_path+"pareto-front"):
        os.mkdir(start_path+"pareto-front")
    if vasp_mode and not os.path.isdir(start_path+"vasp-energy"):
        os.mkdir(start_path+"vasp-energy")
    if vasp_mode and not os.path.isdir(start_path+"energy-configs"):
        os.mkdir(start_path+"energy-configs")


    # scan the available configurations and sort them by size
    df = pd.read_pickle(start_path + config["DATA"]["data_path"], compression="gzip")
    force_energy_filename = start_path + "force_energy.pkl"
    df.iloc[:,4:].to_pickle(force_energy_filename)
    index0 = 0
    index1 = df.shape[0]
    tasks = []
    first_index = [0]
    if not df.index.equals(pd.RangeIndex(0,df.shape[0],1)):
        df.reset_index(inplace=True)
    for i in range(index0,index1):
        atoms = df['ase_atoms'][i]
        n_atoms = len(atoms)
        tasks.append([i,n_atoms])
        first_index.append(first_index[-1]+1+3*n_atoms)
        if not os.path.isfile(start_path+"energy-configs/em_%i.dat"% i):
            write(start_path+"energy-configs/em_%i.dat"% i, atoms, format='vasp')
        if not os.path.isdir(start_path+"vasp-energy/vasp-em_%i"% i):
            os.makedirs(start_path+"vasp-energy/vasp-em_%i"% i)

    #large systems are at the end, small systems are at the front
    tasks.sort(key=lambda x: x[1])

    in_process_featurizations = []
    completed_featurizations = []
    remaining_featurizations = [i for i in range(len(rcuts_list))] if feature_mode else []

    in_process_tasks = []
    completed_tasks = []
    remaining_tasks = [task[0] for task in tasks] if vasp_mode else []
    failed_tasks = []

    trigger_fit = 0 if vasp_mode else 2
    wait_for_last_fit = 0
    job_ids_for_fit = []
    in_process_fits = []
    completed_fits = []
    remaining_fits = [i for i in range(len(hyperparameters_list))] if fit_mode else []

    # #look if a restart file is present
    # if os.path.isfile("restart.p"):
    #     try:
    #         completed_tasks, remaining_tasks=pickle.load( open( "restart.p", "rb" ) )
    #         print("RESTARTING")
    #     except:
    #         pass

    print(len(remaining_tasks)," TASKS REMAINING  --- ", len(in_process_tasks)," TASKS IN PROCESS  --- ", len(completed_tasks), " COMPLETED TASKS")

    if vasp_mode: vasp_futures = set()
    if feature_mode: featurization_futures = set()
    if fit_mode: fitting_futures = set()

    start_time = time.time()
    with Executor(backend="flux_allocation", flux_log_files=True) as exe:

        rl = flux.resource.list.resource_list(handle).get()
        print(rl.free.ncores, "CORES FREE ",all_ncores, "CORES TOTAL")
        print(rl.free.ngpus, "GPUS FREE ",all_ngpus, "GPUS TOTAL")
        featurize_cores = (rl.free.ncores - rl.free.ngpus)//len(rs.nodelist)
        print("Number of cores allocated for featurization step is", featurize_cores)

        print("Featurization step...")
        for i, rcuts in enumerate(rcuts_list):
            remaining_featurizations.remove(i)
            feature_directory = start_path + "features/" + rcuts_to_string(rcuts, delimiter='_')
            if not os.path.isdir(feature_directory):
                os.mkdir(feature_directory)
            fs = exe.submit(featurize, config, fitsnap_config, rcuts, start_path,
                            resource_dict={"cores": 1, "gpus_per_core": 0, "cwd": feature_directory})
            fs.task_ = i
            featurization_futures.add(fs)
            in_process_featurizations.append(i)


        while True:
            # time.sleep(1)
            #we are done
            if (len(remaining_featurizations) == 0 and len(in_process_featurizations) == 0) and \
            (len(remaining_tasks) == 0 and len(in_process_tasks) == 0) and \
            (len(remaining_fits) == 0 and len(in_process_fits) == 0) and wait_for_last_fit == 0:
                break

            rl = flux.resource.list.resource_list(handle).get()
            if len(remaining_tasks) != 0 and len(remaining_fits) != 0:
                print("It has been %.3f seconds since the last check." % (time.time() - start_time))
                start_time = time.time()
                print(rl.free.ncores, "CORES FREE ",all_ncores, "CORES TOTAL")
                print(rl.free.ngpus, "GPUS FREE ",all_ngpus, "GPUS TOTAL")


            # print("SCHEDULING VASP TASKS")
            if len(remaining_tasks)>0:
                rl = flux.resource.list.resource_list(handle).get()
                n_gpus_free = rl.free.ngpus
                n_cores_free = rl.free.ncores
            
                while n_gpus_free>=1 and len(remaining_tasks)>0 and len(in_process_tasks)<all_ngpus:
                    
                    #get one of the "big" jobs
                    task = remaining_tasks.pop(0)
                    input_file = "energy-configs/em_%i.dat"%task
                    vasp_directory = start_path + "vasp-energy/vasp-em_%i/"%task

                    print("RUNNING ", task, "on GPUs", vasp_directory, input_file)
                    fs = exe.submit(fake_vasp, force_energy_filename, task, first_index[task],
                                    resource_dict={"cores": 1, "gpus_per_core": 1, "cwd": vasp_directory})
                    # fs = exe.submit(vasp, start_path, start_path+input_file, task, first_index[task],
                    #                 resource_dict={"cores": 1, "gpus_per_core": 1, "cwd": vasp_directory})
                    fs.task_ = task
                    vasp_futures.add(fs)
                    in_process_tasks.append(task)
                    #time.sleep(0.5)
                    n_gpus_free-=1


            #wait for more resources to become available
            # print("PROCESSING VASP FUTURES")
            if vasp_mode:
                vasp_done, vasp_futures = concurrent.futures.wait(vasp_futures, timeout=0.1)
                for fut in vasp_done:
                    completed_tasks.append(fut.task_)

                    if len(completed_tasks)%fit_freq == 0 and len(in_process_fits) == 0 and len(in_process_featurizations) == 0:
                        trigger_fit = 1
                        print("Triggering fit: ",len(completed_tasks))
                        # pickle.dump( (completed_tasks, remaining_tasks) , open( "restart.p", "wb" ) )
                    if len(completed_tasks) == len(tasks) and len(in_process_fits) == 0 and len(in_process_featurizations) == 0:
                        trigger_fit = 1
                        print("Triggering last fit: ",len(completed_tasks))
                    elif len(completed_tasks) == len(tasks) and len(in_process_fits) != 0:
                        wait_for_last_fit = 1

                    try:
                        print(fut.result())
                    except:
                        pass
                    in_process_tasks.remove(fut.task_)
                    print(len(remaining_tasks)," TASKS REMAINING  --- ", len(in_process_tasks)," TASKS IN PROCESS  --- ",
                        len(completed_tasks), " COMPLETED TASKS")

            
            rl = flux.resource.list.resource_list(handle).get()
            n_cores_free = rl.free.ncores
            n_gpus_free = rl.free.ngpus
            n_excess_cores_free = n_cores_free - n_gpus_free


            # print("PREPARING B.CSV FOR THE FIT")
            if (trigger_fit == 1) and (len(in_process_featurizations)==0):
                # Filesystem is slow consider that
                print("Preparing b.csv for the fit...")
                os.chdir("vasp-energy")
                new_completed_tasks = ["vasp-em_%i/b" % job_id for job_id in completed_tasks if job_id not in job_ids_for_fit]
                print(" ".join(new_completed_tasks))
                os.system("cat " + " ".join(new_completed_tasks) + " >> " + start_path + "features/b.csv")
                os.chdir("..")
                job_ids_for_fit = copy.copy(completed_tasks)
                trigger_fit = 2


            # print("SCHEDULING FITTING TASKS")
            if (trigger_fit == 2 and len(remaining_fits) > 0) and (len(in_process_featurizations)==0):
                #save to a file the configurations that have energies already from completed tasks
                n_excess_cores_free = rl.free.ncores - rl.free.ngpus
                ncores_per_fit = config["MODE"]["ncores_per_fit"]
                while n_excess_cores_free>=ncores_per_fit and len(remaining_fits)>0 and (len(in_process_fits)<((all_ncores-all_ngpus)//ncores_per_fit)):
                    print("Starting the fits...")
                    i = remaining_fits.pop(0)
                    fit_directory = start_path + "fits/" + str(len(job_ids_for_fit)) + "_" + \
                        hyperparameters_to_string(hyperparameters_list[i], delimiter='_')
                    if not os.path.isdir(fit_directory):
                        os.mkdir(fit_directory)
                    fs = exe.submit(fit, start_path+"features/", hyperparameters_list[i], feature_names,
                                    resource_dict={"cores": 1, "gpus_per_core": 0, "cwd": fit_directory})
                    fs.task_ = i
                    fitting_futures.add(fs)
                    in_process_fits.append(i)
                    n_excess_cores_free -= ncores_per_fit
                
                if len(remaining_fits) == 0:
                    if len(remaining_tasks) != 0 or len(in_process_tasks) != 0 or wait_for_last_fit == 1:
                        trigger_fit = 0
                        remaining_fits = [i for i in range(len(hyperparameters_list))]
                    else:
                        trigger_fit = 0


            # print("PROCESSING FITSNAP FUTURES")
            if feature_mode:
                featurizations_done, featurization_futures = concurrent.futures.wait(featurization_futures, timeout=0.1)
                for fut in featurizations_done:
                    feature_names = fut.result()
                    completed_featurizations.append(fut.task_)
                    in_process_featurizations.remove(fut.task_)
                    print(len(remaining_featurizations)," FEATURIZATIONS REMAINING  --- ", len(in_process_featurizations)," FEATURIZATIONS IN PROCESS  --- ",
                        len(completed_featurizations), " COMPLETED FEATURIZATIONS")


            # print("PROCESSING FITTING FUTURES")
            if fit_mode:
                fitting_done, fitting_futures = concurrent.futures.wait(fitting_futures, timeout=0.1)
                for fut in fitting_done:
                    completed_fits.append(fut.task_)
                    in_process_fits.remove(fut.task_)
                    print(len(remaining_fits)," FITS REMAINING  --- ", len(in_process_fits)," FITS IN PROCESS  --- ",
                        len(completed_fits), " COMPLETED FITS")


            # print("PROCESSING PARETO FRONT")
            if len(completed_fits) == len(hyperparameters_list):
                completed_fits = []
                print("All fits are done!")
                if pareto_mode:
                    if pareto(tasks, rs, start_path, hyperparameters_list, feature_names, job_ids_for_fit,
                              remaining_fits, trigger_fit, auto_reduce_hps, wait_for_last_fit):
                        break
            
            
            # print("TRIGGERING LAST FIT")
            if wait_for_last_fit and len(in_process_fits) == 0:
                trigger_fit = 1
                wait_for_last_fit = 0
                print("Triggering last fit: ",len(completed_tasks))



if __name__ == "__main__":
    main()