def pareto(start_path, vasp_batch_idx, hyperparameters_list, hyperparameters_list_noeweight, mlip):
# def pareto(tasks, start_path, hyperparameters_list, hyperparameters_list_noeweight, feature_names, mlip, 
#            vasp_IDs_ready_for_fit, remaining_fits, trigger_fit, auto_reduce_hps, wait_for_last_fit):

    import pandas as pd
    import glob
    from autopiad.tools import ace_hyperparameters_to_string, snap_hyperparameters_to_string

    results_dirs = glob.glob(f"{start_path}fits/{vasp_batch_idx}/*")
    results_df = pd.DataFrame()
    for results_dir in results_dirs:
        results_ = pd.read_csv(results_dir+"/results.csv", header=None)
        columns_list = ["rcut"+str(i) for i in range(len(hyperparameters_list[0][0]))]
        if mlip == "ACE":
            columns_list.extend(["nmax"+str(i+1) for i in range(len(hyperparameters_list[0][1]))])
            columns_list.extend(["lmax"+str(i+1) for i in range(len(hyperparameters_list[0][2]))])
        elif mlip == "SNAP":
            columns_list.extend(["twojmax"+str(i) for i in range(len(hyperparameters_list[0][1]))])
        columns_list.extend(["eweight","train_e_rmse","train_f_rmse","test_e_rmse","test_f_rmse",
                             "train_e_rmse_weighted","train_f_rmse_weighted","test_e_rmse_weighted","test_f_rmse_weighted"])
        results_df = pd.concat([results_df,pd.DataFrame(results_.mean().values[1:].reshape(1,-1), columns=columns_list)])

    cost = pd.DataFrame()
    for i in range(len(hyperparameters_list_noeweight)):
        costs_directory = start_path + "costs/"
        if mlip == "ACE":
            rcuts, nmaxes, lmaxes = hyperparameters_list_noeweight[i]
            values_list = rcuts + nmaxes + lmaxes
            costs_directory += ace_hyperparameters_to_string(hyperparameters_list_noeweight[i], delimiter='_', w_eweight=False)
        if mlip == "SNAP":
            rcuts, twojmaxes = hyperparameters_list_noeweight[i]
            values_list = rcuts + twojmaxes
            costs_directory += snap_hyperparameters_to_string(hyperparameters_list_noeweight[i], delimiter='_', w_eweight=False)
        with open(costs_directory+"/flux_0.out", "r") as f:
            lines = f.readlines()
            for line in lines:
                if "process_configs" in line:
                    cost=pd.concat([cost,pd.DataFrame([values_list+[float(line.split()[2])]],
                                                      columns=columns_list[:-9]+['cost'])])

    results_df = results_df.merge(cost, how='inner', on=columns_list[:-9])

    not_minima_list = []
    for i in range(results_df.shape[0]):
        for j in range(results_df.shape[0]):
            if (results_df.iloc[i,-1] > results_df.iloc[j,-1]) and (results_df.iloc[i,-2] > results_df.iloc[j,-2]) and (results_df.iloc[i,-3] > results_df.iloc[j,-3]):
                not_minima_list.append(i)
                break
    minima_list = [i for i in range(results_df.shape[0]) if i not in not_minima_list]
    print("Number of points on Pareto Front is", len(minima_list))

    results_df["pareto_front"] = 0
    results_df.loc[minima_list, "pareto_front"] = 1
    results_df.to_csv(start_path+"pareto-front/results_%i.csv" % vasp_batch_idx, index=False)

    # if len(job_ids_for_fit) > 0.2*len(tasks) and auto_reduce_hps:
    #     file_number_count=0
    #     results_df_list = []
    #     for file_name in glob.glob(start_path+"pareto-front/results_*.csv"):
    #         # if int(file_name.split('/')[-1][8:-4]) > 0.1*len(tasks):
    #         results_df_list.append([int(file_name.split('/')[-1][8:-4]),pd.read_csv(file_name)])
    #         if int(file_name.split('/')[-1][8:-4]) < 0.2*len(tasks):
    #             file_number_count += 1
    #     results_df_list.sort(key=lambda x: x[0])
    #     results_df_list = [i[1] for i in results_df_list]
    #     results_df_list = results_df_list[-file_number_count:]
    #     if len(results_df_list) == 0:
    #         results_df_list = [results_df]
    #     pareto_count = []
    #     for hyperparameters in hyperparameters_list:
    #         pareto_count.append(0)
    #         for results_df_i in results_df_list:
    #             df_query_str = [columns_list[j]+'==%.3f'%hyperparameters[0][j] for j in range(len(hyperparameters[0]))]
    #             df_query_str.extend([columns_list[j+len(hyperparameters[0])]+'==%i'%hyperparameters[1][j] for j in range(len(hyperparameters[1]))])
    #             df_query_str.append('eweight==%.3f'%hyperparameters[2])
    #             pareto_count[-1] += results_df_i.query(" and ".join(df_query_str))['pareto_front'].iloc[0]
    #     for i in reversed(range(len(pareto_count))):
    #         if pareto_count[i] == 0:
    #             hyperparameters_list.pop(i)
        
    #     if len(remaining_fits)!=0 and trigger_fit==0:
    #         remaining_fits = [i for i in range(len(hyperparameters_list))]


    # if len(remaining_fits) == 0 and wait_for_last_fit == 0:
    #     return 1
    
    return 0