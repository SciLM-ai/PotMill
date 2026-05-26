
def init_fit():
    """executorlib init_function: warm torch + CUDA + subdatapy once per GPU fit worker.

    Block-allocated workers persist, so creating the CUDA context here (instead of on the
    first fit) amortizes the ~seconds-long context init across every fit the worker handles.
    Returns {} -- its only job is to warm the persistent process; nothing is injected.
    """
    import os
    import torch
    from subdatapy import linalg  # noqa: F401  (warm the import cache)
    device_count = torch.cuda.device_count()
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")  # create the CUDA context now, not on first fit
    print(f"init_fit: torch {torch.__version__} | cuda_devices={device_count} | "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    return {}


def _gpu_solve(A, B, fit_method, fit_device, rcond):
    """Solve the weighted least-squares (A, B) on GPU. A:(n,p), B:(n,1) torch tensors.
    Returns the 1-D coefficient tensor on `fit_device` (kept on GPU for the RMSE step)."""
    import torch
    if fit_method == "svd":
        # SVD pseudo-inverse with rcond truncation: matches np.linalg.lstsq(rcond) exactly,
        # but on GPU. Robust to the rank-deficient / ill-conditioned ACE design matrices that
        # appear for early (low-config) batches, where plain QR (no regularization) blows up.
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        S_inv = torch.where(S > rcond * S[0], 1.0 / S, torch.zeros_like(S))
        return (Vh.mT @ (S_inv.unsqueeze(-1) * (U.mT @ B))).reshape(-1)
    elif fit_method == "sd_svd":
        from subdatapy import linalg
        XTX_inv, _, _, _ = linalg.xtx_inv_from_svd(A, device=fit_device)
        return (XTX_inv @ (A.mT @ B)).reshape(-1)
    else:  # 'qr' (TSQR) or 'lstsq' -- SubDataPy solve_wls; assumes full column rank
        from subdatapy import linalg
        return linalg.solve_wls(A, B, method=fit_method, device=fit_device, n_chunks=None,
                                partitioned=False, local_devices=None, dtype=torch.float64).reshape(-1)


def fit(features_directory, feature_names, vasp_IDs_ready_for_fit, hyperparameters,
        mlip, batch_ID=None, train_fraction = 0.7, n_fold = 3, rcond = 1e-13,
        fit_directory=None, fit_device="cuda", fit_method="svd"):

    import os
    import numpy as np
    import pandas as pd
    import random
    from autopiad.tools import rcuts_to_string, nmaxes_to_string, lmaxes_to_string, twojmaxes_to_string

    # Block-allocated fitting workers have a fixed CWD; chdir to this fit's directory so
    # results.csv and the beta files land in the right per-fit folder (mirrors uma()).
    if fit_directory is not None:
        os.chdir(fit_directory)

    if isinstance(feature_names[0][0],list): feature_names = feature_names[0]

    if mlip == "ACE":
        rcuts, nmaxes, lmaxes, eweight = hyperparameters
        nindcs_to_bodyorder = {5:2, 8:3, 12:4, 16:5, 20:6, 24:7}
        feature_indices = []
        for i, lst in enumerate(feature_names):
            if len(lst)==1:
                feature_indices.append(i)
            else:
                nu = nindcs_to_bodyorder[len(lst)]
                if all(lst[nu+1+k]<=nmaxes[nu-2] and lst[2*nu+k]<=lmaxes[nu-2] for k in range(nu-1)):
                    feature_indices.append(i)
    elif mlip == "SNAP":
        rcuts, twojmaxes, eweight = hyperparameters
        feature_indices = [i for i, lst in enumerate(feature_names) if len(lst)==1 or all(value <= twojmaxes[0] for value in lst[1:])]

    print(len(feature_indices), len(feature_names), flush=True)
    b_size = len(vasp_IDs_ready_for_fit)
    b_vect = pd.read_csv(f"{features_directory}b{b_size}.csv", index_col=0, header=None).sort_index()
    rcuts_str = rcuts_to_string(rcuts, delimiter='_')

    use_gpu = (fit_method != "numpy")
    if use_gpu:
        import torch

    # ---- Load the (cumulative) design matrix, column-selected to feature_indices ----
    # numpy path keeps it on CPU; GPU path concatenates and lives on `fit_device`.
    if batch_ID is None:
        b_vect_index = b_vect.index.to_numpy()
        a_map = np.load(f"{features_directory}{rcuts_str}/a.npy", mmap_mode='r')
        a_sel = np.ascontiguousarray(a_map[b_vect_index[:, None], feature_indices])
        a_matr = torch.as_tensor(a_sel, dtype=torch.float64, device=fit_device) if use_gpu else a_sel
    else:
        if use_gpu:
            parts = []
            for bid in range(batch_ID+1):
                a_map = np.load(f"{features_directory}{bid}/{rcuts_str}/a.npy", mmap_mode='r')
                parts.append(torch.from_numpy(np.ascontiguousarray(a_map[:, feature_indices])).to(
                    device=fit_device, dtype=torch.float64))
            a_matr = torch.cat(parts)   # concat on GPU
        else:
            parts = []
            for bid in range(batch_ID+1):
                a_map = np.load(f"{features_directory}{bid}/{rcuts_str}/a.npy", mmap_mode='r')
                parts.append(a_map[:, feature_indices])
            a_matr = np.concatenate(parts)

    b_vect.reset_index(inplace=True)
    b_vect_no_dupl = b_vect.drop_duplicates(subset=1)
    job_ids = b_vect_no_dupl[1].to_list()
    energy_selector = b_vect_no_dupl.index.to_list()
    force_selector = b_vect.drop(index=energy_selector).index.to_list()
    assert len(force_selector)+len(energy_selector) == b_vect.shape[0]

    b_values = b_vect[2].values                       # 1-D numpy (tiny)
    if use_gpu:
        b_all_t = torch.as_tensor(b_values, dtype=torch.float64, device=fit_device)

    for fold in range(n_fold):

        print("===================== FOLD ",fold," OF ",n_fold,"=====================", flush=True)
        if mlip == "ACE":
            print("Hyperparameters rcut, nmax, lmax and eweight are " + rcuts_to_string(rcuts) + ", " +
                nmaxes_to_string(nmaxes) + ", " + lmaxes_to_string(lmaxes) + " and %.3f"%eweight, flush=True)
        if mlip == "SNAP":
            print("Hyperparameters rcut, 2Jmax and eweight are " + rcuts_to_string(rcuts) + ", " +
                twojmaxes_to_string(twojmaxes) + " and %.3f"%eweight, flush=True)

        # ---- train/test split by config (CPU; b is tiny). Identical for both paths. ----
        random.seed(fold)
        random.shuffle(job_ids)
        train_ids = job_ids[:int(train_fraction*len(job_ids))]
        test_ids = job_ids[int(train_fraction*len(job_ids)):]
        train_index = b_vect[b_vect[1].isin(train_ids)].index.to_list()
        test_index = b_vect[b_vect[1].isin(test_ids)].index.to_list()
        energy_selector_train = np.in1d(train_index, energy_selector)
        energy_selector_test  = np.in1d(test_index,  energy_selector)
        force_selector_train  = np.in1d(train_index, force_selector)
        force_selector_test   = np.in1d(test_index,  force_selector)

        if not use_gpu:
            # -------------------- original all-CPU (numpy) path --------------------
            a_train = a_matr[train_index]
            a_test = a_matr[test_index]
            b_train = b_values[train_index]
            b_test = b_values[test_index]

            eweights_train = np.exp(-b_train[energy_selector_train]/5)
            eweights_train /= np.sum(eweights_train); eweights_train *= eweight
            fweights_train = 1./np.maximum(3.,np.fabs(b_train[force_selector_train]))
            fweights_train /= np.sum(fweights_train); fweights_train *= 1.
            eweights_test = np.exp(-b_test[energy_selector_test]/5)
            eweights_test /= np.sum(eweights_test); eweights_test *= 1.
            fweights_test = 1./np.maximum(3.,np.fabs(b_test[force_selector_test]))
            fweights_test /= np.sum(fweights_test); fweights_test *= 1.

            a_e_train_w = np.multiply(eweights_train[:,None],a_train[energy_selector_train])
            a_f_train_w = np.multiply(fweights_train[:,None],a_train[force_selector_train])
            b_e_train_w = np.multiply(eweights_train,b_train[energy_selector_train])
            b_f_train_w = np.multiply(fweights_train,b_train[force_selector_train])
            a_stack = np.concatenate([a_e_train_w,a_f_train_w])
            b_stack = np.concatenate([b_e_train_w,b_f_train_w])
            print(a_stack.shape, b_stack.shape, flush=True)

            beta, *_ = np.linalg.lstsq(a_stack, b_stack, rcond)

            train_residual = np.square(np.dot(a_train,beta) - b_train)
            test_residual  = np.square(np.dot(a_test, beta) - b_test)
            train_e_rmse = float(np.sqrt(np.mean(train_residual[energy_selector_train])))
            train_f_rmse = float(np.sqrt(np.mean(train_residual[force_selector_train])))
            train_e_rmse_weighted = float(np.sqrt(np.sum(np.multiply(eweights_train,train_residual[energy_selector_train]))))
            train_f_rmse_weighted = float(np.sqrt(np.sum(np.multiply(fweights_train,train_residual[force_selector_train]))))
            test_e_rmse = float(np.sqrt(np.mean(test_residual[energy_selector_test])))
            test_f_rmse = float(np.sqrt(np.mean(test_residual[force_selector_test])))
            test_e_rmse_weighted = float(np.sqrt(np.sum(np.multiply(eweights_test,test_residual[energy_selector_test]))))
            test_f_rmse_weighted = float(np.sqrt(np.sum(np.multiply(fweights_test,test_residual[force_selector_test]))))
        else:
            # -------------------- GPU path: everything in torch on fit_device --------------------
            ti = torch.as_tensor(train_index, dtype=torch.long, device=fit_device)
            tj = torch.as_tensor(test_index,  dtype=torch.long, device=fit_device)
            e_tr = torch.as_tensor(energy_selector_train, dtype=torch.bool, device=fit_device)
            e_te = torch.as_tensor(energy_selector_test,  dtype=torch.bool, device=fit_device)
            f_tr = torch.as_tensor(force_selector_train,  dtype=torch.bool, device=fit_device)
            f_te = torch.as_tensor(force_selector_test,   dtype=torch.bool, device=fit_device)
            a_train = a_matr[ti]; a_test = a_matr[tj]
            b_train = b_all_t[ti]; b_test = b_all_t[tj]

            eweights_train = torch.exp(-b_train[e_tr]/5)
            eweights_train = eweights_train/eweights_train.sum()*eweight
            fweights_train = 1./torch.clamp(torch.abs(b_train[f_tr]), min=3.)
            fweights_train = fweights_train/fweights_train.sum()
            eweights_test = torch.exp(-b_test[e_te]/5)
            eweights_test = eweights_test/eweights_test.sum()
            fweights_test = 1./torch.clamp(torch.abs(b_test[f_te]), min=3.)
            fweights_test = fweights_test/fweights_test.sum()

            a_e_train_w = eweights_train.unsqueeze(1)*a_train[e_tr]
            a_f_train_w = fweights_train.unsqueeze(1)*a_train[f_tr]
            b_e_train_w = eweights_train*b_train[e_tr]
            b_f_train_w = fweights_train*b_train[f_tr]
            a_stack = torch.cat([a_e_train_w,a_f_train_w])
            b_stack = torch.cat([b_e_train_w,b_f_train_w])
            print(tuple(a_stack.shape), tuple(b_stack.shape), flush=True)

            beta_t = _gpu_solve(a_stack, b_stack.reshape(-1,1), fit_method, fit_device, rcond)
            if not torch.all(torch.isfinite(beta_t)):
                print(f"WARNING: non-finite beta from '{fit_method}' solve (rank-deficient "
                      f"design?) for hyperparameters {hyperparameters}", flush=True)

            train_residual = torch.square(a_train @ beta_t - b_train)
            test_residual  = torch.square(a_test  @ beta_t - b_test)
            train_e_rmse = torch.sqrt(torch.mean(train_residual[e_tr])).item()
            train_f_rmse = torch.sqrt(torch.mean(train_residual[f_tr])).item()
            train_e_rmse_weighted = torch.sqrt(torch.sum(eweights_train*train_residual[e_tr])).item()
            train_f_rmse_weighted = torch.sqrt(torch.sum(fweights_train*train_residual[f_tr])).item()
            test_e_rmse = torch.sqrt(torch.mean(test_residual[e_te])).item()
            test_f_rmse = torch.sqrt(torch.mean(test_residual[f_te])).item()
            test_e_rmse_weighted = torch.sqrt(torch.sum(eweights_test*test_residual[e_te])).item()
            test_f_rmse_weighted = torch.sqrt(torch.sum(fweights_test*test_residual[f_te])).item()
            beta = beta_t.detach().cpu().numpy()       # only the (p,) coeffs come back to CPU

        print("Energy training RMSE is", train_e_rmse, flush=True)
        print("Force training RMSE is", train_f_rmse, flush=True)
        print("Energy testing RMSE is", test_e_rmse, flush=True)
        print("Force testing RMSE is", test_f_rmse, flush=True)

        # ---- write results.csv row + beta file (shared; identical format for both paths) ----
        with open("results.csv","a") as file:
            if mlip == "ACE":
                results_line = "%i,"%fold + rcuts_to_string(rcuts,delimiter=",") + "," + \
                    nmaxes_to_string(nmaxes,delimiter=",") + "," + lmaxes_to_string(lmaxes,delimiter=",")
            elif mlip == "SNAP":
                results_line = "%i,"%fold + rcuts_to_string(rcuts,delimiter=",") + "," + \
                    twojmaxes_to_string(twojmaxes,delimiter=",")
            results_line += ",%.3f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n" % \
                    (eweight,train_e_rmse,train_f_rmse,test_e_rmse,test_f_rmse,train_e_rmse_weighted,
                     train_f_rmse_weighted,test_e_rmse_weighted,test_f_rmse_weighted)
            file.write(results_line)

        beta_filename = "pot__rcut_" + rcuts_to_string(rcuts,delimiter="_")
        if mlip == "ACE":
            beta_filename += "__nmax_" + nmaxes_to_string(nmaxes,delimiter="_") + \
                "__lmax_" + nmaxes_to_string(nmaxes,delimiter="_") + "__eweight_%.3f__fold_%i.csv"%(eweight,fold)
        elif mlip == "SNAP":
            beta_filename += "__2jmax_" + twojmaxes_to_string(twojmaxes,delimiter="_") + \
                "__eweight_%.3f__fold_%i.csv"%(eweight,fold)
        np.savetxt(beta_filename, beta)

    return hyperparameters
