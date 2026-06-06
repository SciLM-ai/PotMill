
def pops(features_directory, feature_names, vasp_IDs_ready_for_fit, hyperparameters,
        mlip, batch_ID=None, train_fraction = 0.7, n_fold = 3):

    import numpy as np
    from POPSRegression import POPSRegression
    from potmill.tools import rcuts_to_string, nmaxes_to_string, lmaxes_to_string, twojmaxes_to_string
    from potmill.fit import config_fold, _feature_indices
    from potmill.bfile import read_b

    if mlip == "ACE":
        rcuts, nmaxes, lmaxes, eweight = hyperparameters
        hp_noeweight = [rcuts, nmaxes, lmaxes]
    elif mlip == "SNAP":
        rcuts, twojmaxes, eweight = hyperparameters
        hp_noeweight = [rcuts, twojmaxes]
    feature_indices = _feature_indices(mlip, feature_names, hp_noeweight)

    b_size = len(vasp_IDs_ready_for_fit)
    local_idx, job_id_col, b_values = read_b(f"{features_directory}b{b_size}.csv")
    is_energy = (local_idx == 0)                             # energy row = local index 0 per config
    rcuts_str = rcuts_to_string(rcuts, delimiter='_')
    if batch_ID is None:
        a_matr_map = np.load(f"{features_directory}{rcuts_str}/a.npy", mmap_mode='r')
        a_matr = np.ascontiguousarray(a_matr_map[:, feature_indices])
    else:
        parts = [np.load(f"{features_directory}{bid}/{rcuts_str}/a.npy", mmap_mode='r')[:, feature_indices]
                 for bid in range(batch_ID+1)]
        a_matr = np.concatenate(parts)
    assert a_matr.shape[0] == len(b_values), (a_matr.shape[0], len(b_values))
    # fixed config->fold partition (per row), proper k-fold (test = 1/n_fold)
    fold_of_job = {int(j): config_fold(j, n_fold) for j in np.unique(job_id_col)}
    part = np.array([fold_of_job[int(j)] for j in job_id_col])

    for fold in range(n_fold):

        print("===================== FOLD ",fold," OF ",n_fold,"=====================", flush=True)
        if mlip == "ACE":
            print("Hyperparameters rcut, nmax, lmax and eweight are " + rcuts_to_string(rcuts) + ", " +
                nmaxes_to_string(nmaxes) + ", " + lmaxes_to_string(lmaxes) + " and %.3f"%eweight, flush=True)
        if mlip == "SNAP":
            print("Hyperparameters rcut, 2Jmax and eweight are " + rcuts_to_string(rcuts) + ", " +
                twojmaxes_to_string(twojmaxes) + " and %.3f"%eweight, flush=True)

        train_index = np.where(part != fold)[0]
        test_index = np.where(part == fold)[0]

        a_train = a_matr[train_index]
        a_test = a_matr[test_index]
        b_train = b_values[train_index]
        b_test = b_values[test_index]

        energy_selector_train = is_energy[train_index]
        energy_selector_test = is_energy[test_index]
        force_selector_train = ~energy_selector_train
        force_selector_test = ~energy_selector_test
        
        eweights_train = np.exp(-b_train[energy_selector_train]/5)
        eweights_train /= np.sum(eweights_train)
        eweights_train *= eweight

        fweights_train = 1./np.maximum(3.,np.fabs(b_train[force_selector_train]))
        fweights_train /= np.sum(fweights_train)
        fweights_train *= 1.

        eweights_test = np.exp(-b_test[energy_selector_test]/5)
        eweights_test /= np.sum(eweights_test)
        eweights_test *= 1.

        fweights_test = 1./np.maximum(3.,np.fabs(b_test[force_selector_test]))
        fweights_test /= np.sum(fweights_test)
        fweights_test *= 1.

        a_e_train_w = np.multiply(eweights_train[:,None],a_train[energy_selector_train])
        a_f_train_w = np.multiply(fweights_train[:,None],a_train[force_selector_train])
        b_e_train_w = np.multiply(eweights_train,b_train[energy_selector_train])
        b_f_train_w = np.multiply(fweights_train,b_train[force_selector_train])
        print(a_e_train_w.shape, a_f_train_w.shape, flush=True)

        a_stack = np.concatenate([a_e_train_w,a_f_train_w])
        b_stack = np.concatenate([b_e_train_w,b_f_train_w])
        print(a_stack.shape, b_stack.shape, flush=True)

        model = POPSRegression(resampling_method='sobol',resample_density=1.)
        model.fit(a_stack,b_stack)

        b_train_pred, b_train_std = model.predict(a_train, return_std=True)
        train_residual = np.square(b_train_pred - b_train)
        train_e_std_mean = np.mean(b_train_std[energy_selector_train])
        train_f_std_mean = np.mean(b_train_std[force_selector_train])
        train_e_rmse = np.sqrt(np.mean(train_residual[energy_selector_train]))
        train_f_rmse = np.sqrt(np.mean(train_residual[force_selector_train]))
        train_e_rmse_weighted = np.sqrt(np.sum(np.multiply(eweights_train,train_residual[energy_selector_train])))
        train_f_rmse_weighted = np.sqrt(np.sum(np.multiply(fweights_train,train_residual[force_selector_train])))
        print("Energy training RMSE is", np.sqrt(np.mean(train_residual[energy_selector_train])), flush=True)
        print("Force training RMSE is", np.sqrt(np.mean(train_residual[force_selector_train])), flush=True)
        b_test_pred, b_test_std = model.predict(a_test, return_std=True)
        test_residual = np.square(b_test_pred - b_test)
        test_e_std_mean = np.mean(b_test_std[energy_selector_test])
        test_f_std_mean = np.mean(b_test_std[force_selector_test])
        test_e_rmse = np.sqrt(np.mean(test_residual[energy_selector_test]))
        test_f_rmse = np.sqrt(np.mean(test_residual[force_selector_test]))
        test_e_rmse_weighted = np.sqrt(np.sum(np.multiply(eweights_test,test_residual[energy_selector_test])))
        test_f_rmse_weighted = np.sqrt(np.sum(np.multiply(fweights_test,test_residual[force_selector_test])))
        print("Energy testing RMSE is", np.sqrt(np.mean(test_residual[energy_selector_test])), flush=True)
        print("Force testing RMSE is", np.sqrt(np.mean(test_residual[force_selector_test])), flush=True)
        

        with open("results.csv","a") as file:
            if mlip == "ACE":
                results_line = "%i,"%fold + rcuts_to_string(rcuts,delimiter=",") + "," + \
                    nmaxes_to_string(nmaxes,delimiter=",") + "," + lmaxes_to_string(lmaxes,delimiter=",")
            elif mlip == "SNAP":
                results_line = "%i,"%fold + rcuts_to_string(rcuts,delimiter=",") + "," + \
                    twojmaxes_to_string(twojmaxes,delimiter=",")
            results_line += ",%.3f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n" % \
                    (eweight,train_e_rmse,train_f_rmse,test_e_rmse,test_f_rmse,train_e_rmse_weighted,
                     train_f_rmse_weighted,test_e_rmse_weighted,test_f_rmse_weighted,
                     train_e_std_mean,train_f_std_mean,test_e_std_mean,test_f_std_mean)
            file.write(results_line)

    return hyperparameters