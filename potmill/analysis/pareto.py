def pareto(start_path, vasp_batch_idx, hyperparameters_list, hyperparameters_list_noeweight, mlip):

    import glob

    import pandas as pd

    from potmill.tools import ace_hyperparameters_to_string, snap_hyperparameters_to_string

    # Column dims come from hyperparameters_list_noeweight (the resolved cost_futures) -- the
    # eweight-free rcut/nmax/lmax (ACE) or rcut/twojmax (SNAP), engine-agnostic. The
    # hyperparameters_list argument is only a dependency barrier (in the incremental engine it
    # resolves to state-file paths, not hyperparameters, so it must NOT set the column structure).
    columns_list = ["rcut" + str(i) for i in range(len(hyperparameters_list_noeweight[0][0]))]
    if mlip == "ACE":
        columns_list.extend(
            ["nmax" + str(i + 1) for i in range(len(hyperparameters_list_noeweight[0][1]))]
        )
        columns_list.extend(
            ["lmax" + str(i + 1) for i in range(len(hyperparameters_list_noeweight[0][2]))]
        )
    elif mlip == "SNAP":
        columns_list.extend(
            ["twojmax" + str(i) for i in range(len(hyperparameters_list_noeweight[0][1]))]
        )
    columns_list.extend(
        [
            "eweight",
            "train_e_rmse",
            "train_f_rmse",
            "test_e_rmse",
            "test_f_rmse",
            "train_e_rmse_weighted",
            "train_f_rmse_weighted",
            "test_e_rmse_weighted",
            "test_f_rmse_weighted",
        ]
    )

    # Each fit worker appended its (fold + columns_list) rows to fits/{batch}/results_{pid}.csv (one
    # set of files per worker, not one folder per combo). Group by combo -- everything except the 8
    # RMSE columns -- and mean over the n_fold rows, reproducing the old per-combo results.csv mean.
    results_files = glob.glob(f"{start_path}fits/{vasp_batch_idx}/results_*.csv")
    all_rows = pd.concat([pd.read_csv(f, header=None) for f in results_files], ignore_index=True)
    all_rows.columns = ["fold"] + columns_list
    results_df = all_rows.groupby(columns_list[:-8], as_index=False)[columns_list[-8:]].mean()

    cost = pd.DataFrame()
    for i in range(len(hyperparameters_list_noeweight)):
        costs_directory = start_path + "costs/"
        if mlip == "ACE":
            print("hyperparameters_list_noeweight", hyperparameters_list_noeweight[i])
            rcuts, nmaxes, lmaxes = hyperparameters_list_noeweight[i]
            values_list = rcuts + nmaxes + lmaxes
            costs_directory += ace_hyperparameters_to_string(
                hyperparameters_list_noeweight[i], delimiter="_", w_eweight=False
            )
        if mlip == "SNAP":
            rcuts, twojmaxes = hyperparameters_list_noeweight[i]
            values_list = rcuts + twojmaxes
            costs_directory += snap_hyperparameters_to_string(
                hyperparameters_list_noeweight[i], delimiter="_", w_eweight=False
            )
        with open(costs_directory + "/flux_0.out") as f:
            lines = f.readlines()
            for line in lines:
                if "process_configs" in line:
                    cost = pd.concat(
                        [
                            cost,
                            pd.DataFrame(
                                [values_list + [float(line.split()[2])]],
                                columns=columns_list[:-9] + ["cost"],
                            ),
                        ]
                    )

    results_df = results_df.merge(cost, how="inner", on=columns_list[:-9])

    not_minima_list = []
    for i in range(results_df.shape[0]):
        for j in range(results_df.shape[0]):
            if (
                (results_df.iloc[i, -1] > results_df.iloc[j, -1])
                and (results_df.iloc[i, -2] > results_df.iloc[j, -2])
                and (results_df.iloc[i, -3] > results_df.iloc[j, -3])
            ):
                not_minima_list.append(i)
                break
    minima_list = [i for i in range(results_df.shape[0]) if i not in not_minima_list]
    print("Number of points on Pareto Front is", len(minima_list), flush=True)

    results_df["pareto_front"] = 0
    results_df.loc[minima_list, "pareto_front"] = 1
    results_df.to_csv(start_path + "pareto-front/results_%i.csv" % vasp_batch_idx, index=False)

    return 0
