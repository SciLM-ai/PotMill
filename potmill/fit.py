
def init_fit():
    """executorlib init_function: warm torch + CUDA + subdatapy once per GPU fit worker.

    Block-allocated workers persist, so creating the CUDA context here (instead of on the
    first fit) amortizes the ~seconds-long context init across every fit the worker handles.
    Returns {} -- its only job is to warm the persistent process; nothing is injected.
    """
    import os
    import torch
    try:
        from subdatapy import linalg  # noqa: F401  (warm the import cache if present)
    except Exception:
        pass
    device_count = torch.cuda.device_count()
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")  # create the CUDA context now, not on first fit
    print(f"init_fit: torch {torch.__version__} | cuda_devices={device_count} | "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    return {}


def config_fold(job_id, n_fold):
    """Deterministic, fixed config->fold assignment for k-fold CV.

    A configuration's fold is decided once from its job_id and NEVER changes as more
    batches arrive -- this is what makes incremental accumulation valid (a config's
    energy + all its forces always stay together on the same train/test side).
    md5 hash gives a balanced, order-independent partition.
    """
    import hashlib
    return int(hashlib.md5(str(int(job_id)).encode()).hexdigest(), 16) % n_fold


def _feature_indices(mlip, feature_names, hp_noeweight):
    """Column subset (feature_indices) selected by a hyperparameter's (rcut,)nmax,lmax / twojmax.
    hp_noeweight is the eweight-free hyperparameter: [rcut, nmax, lmax] (ACE) or [rcut, twojmax] (SNAP)."""
    if isinstance(feature_names[0][0], list):
        feature_names = feature_names[0]
    if mlip == "ACE":
        rcuts, nmaxes, lmaxes = hp_noeweight
        nindcs_to_bodyorder = {5: 2, 8: 3, 12: 4, 16: 5, 20: 6, 24: 7}
        feature_indices = []
        for i, lst in enumerate(feature_names):
            if len(lst) == 1:
                feature_indices.append(i)
            else:
                nu = nindcs_to_bodyorder[len(lst)]
                if all(lst[nu+1+k] <= nmaxes[nu-2] and lst[2*nu+k] <= lmaxes[nu-2] for k in range(nu-1)):
                    feature_indices.append(i)
    elif mlip == "SNAP":
        rcuts, twojmaxes = hp_noeweight
        feature_indices = [i for i, lst in enumerate(feature_names)
                           if len(lst) == 1 or all(value <= twojmaxes[0] for value in lst[1:])]
    return feature_indices


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


# ============================================================================================
# Row-based fit (reference / fallback, fit_engine='rows').
#   - a/b paired in RAW (config-major) order -- NO sort_index (the misalignment fix).
#   - fixed config->fold k-fold CV (test = 1/n_fold), shared with the incremental path.
#   - loads the cumulative design matrix each checkpoint (O(N^2) overall); fine for small N
#     and as the exact reference the incremental foldfit() is validated against.
# ============================================================================================
def fit(features_directory, feature_names, vasp_IDs_ready_for_fit, hyperparameters,
        mlip, batch_ID=None, n_fold=3, rcond=1e-13,
        fit_directory=None, fit_device="cuda", fit_method="svd"):

    import os
    import numpy as np
    import pandas as pd
    from potmill.tools import rcuts_to_string, nmaxes_to_string, lmaxes_to_string, twojmaxes_to_string

    # Block-allocated fitting workers have a fixed CWD; chdir to this fit's directory so
    # results.csv and the beta files land in the right per-fit folder (mirrors uma()).
    if fit_directory is not None:
        os.chdir(fit_directory)

    if mlip == "ACE":
        rcuts, nmaxes, lmaxes, eweight = hyperparameters
        hp_noeweight = [rcuts, nmaxes, lmaxes]
    elif mlip == "SNAP":
        rcuts, twojmaxes, eweight = hyperparameters
        hp_noeweight = [rcuts, twojmaxes]
    feature_indices = _feature_indices(mlip, feature_names, hp_noeweight)

    print(len(feature_indices), len(feature_names if not isinstance(feature_names[0][0], list)
          else feature_names[0]), flush=True)
    b_size = len(vasp_IDs_ready_for_fit)
    # NO sort_index: rows stay in raw (config-major) order, aligned with a.npy.
    b_vect = pd.read_csv(f"{features_directory}b{b_size}.csv", header=None)
    rcuts_str = rcuts_to_string(rcuts, delimiter='_')

    use_gpu = (fit_method != "numpy")
    if use_gpu:
        import torch

    # ---- Load the (cumulative) design matrix, column-selected, config-major (aligned with b) ----
    if batch_ID is None:
        a_map = np.load(f"{features_directory}{rcuts_str}/a.npy", mmap_mode='r')
        a_sel = np.ascontiguousarray(a_map[:, feature_indices])
        a_matr = torch.as_tensor(a_sel, dtype=torch.float64, device=fit_device) if use_gpu else a_sel
    else:
        parts = []
        for bid in range(batch_ID+1):
            a_map = np.load(f"{features_directory}{bid}/{rcuts_str}/a.npy", mmap_mode='r')
            sub = np.ascontiguousarray(a_map[:, feature_indices])
            parts.append(torch.from_numpy(sub).to(device=fit_device, dtype=torch.float64) if use_gpu else sub)
        a_matr = torch.cat(parts) if use_gpu else np.concatenate(parts)

    job_id_col = b_vect[1].values
    b_values = b_vect[2].values                              # 1-D numpy (tiny)
    is_energy = (b_vect[0].values == 0)                      # energy row = local index 0 per config
    assert a_matr.shape[0] == len(b_values), (a_matr.shape[0], len(b_values))
    if use_gpu:
        b_all_t = torch.as_tensor(b_values, dtype=torch.float64, device=fit_device)

    # fixed config->fold partition (per row)
    fold_of_job = {int(j): config_fold(j, n_fold) for j in np.unique(job_id_col)}
    part = np.array([fold_of_job[int(j)] for j in job_id_col])

    for fold in range(n_fold):

        print("===================== FOLD ", fold, " OF ", n_fold, "=====================", flush=True)
        if mlip == "ACE":
            print("Hyperparameters rcut, nmax, lmax and eweight are " + rcuts_to_string(rcuts) + ", " +
                nmaxes_to_string(nmaxes) + ", " + lmaxes_to_string(lmaxes) + " and %.3f" % eweight, flush=True)
        if mlip == "SNAP":
            print("Hyperparameters rcut, 2Jmax and eweight are " + rcuts_to_string(rcuts) + ", " +
                twojmaxes_to_string(twojmaxes) + " and %.3f" % eweight, flush=True)

        # ---- k-fold split BY CONFIG: test = partition==fold (1/n_fold), train = the rest ----
        train_index = np.where(part != fold)[0]
        test_index = np.where(part == fold)[0]
        energy_selector_train = is_energy[train_index]
        energy_selector_test  = is_energy[test_index]
        force_selector_train  = ~energy_selector_train
        force_selector_test   = ~energy_selector_test

        if not use_gpu:
            # -------------------- original all-CPU (numpy) path --------------------
            a_train = a_matr[train_index]; a_test = a_matr[test_index]
            b_train = b_values[train_index]; b_test = b_values[test_index]

            eweights_train = np.exp(-b_train[energy_selector_train]/5)
            eweights_train /= np.sum(eweights_train); eweights_train *= eweight
            fweights_train = 1./np.maximum(3., np.fabs(b_train[force_selector_train]))
            fweights_train /= np.sum(fweights_train); fweights_train *= 1.
            eweights_test = np.exp(-b_test[energy_selector_test]/5)
            eweights_test /= np.sum(eweights_test); eweights_test *= 1.
            fweights_test = 1./np.maximum(3., np.fabs(b_test[force_selector_test]))
            fweights_test /= np.sum(fweights_test); fweights_test *= 1.

            a_e_train_w = np.multiply(eweights_train[:, None], a_train[energy_selector_train])
            a_f_train_w = np.multiply(fweights_train[:, None], a_train[force_selector_train])
            b_e_train_w = np.multiply(eweights_train, b_train[energy_selector_train])
            b_f_train_w = np.multiply(fweights_train, b_train[force_selector_train])
            a_stack = np.concatenate([a_e_train_w, a_f_train_w])
            b_stack = np.concatenate([b_e_train_w, b_f_train_w])
            print(a_stack.shape, b_stack.shape, flush=True)

            beta, *_ = np.linalg.lstsq(a_stack, b_stack, rcond)

            train_residual = np.square(np.dot(a_train, beta) - b_train)
            test_residual  = np.square(np.dot(a_test, beta) - b_test)
            train_e_rmse = float(np.sqrt(np.mean(train_residual[energy_selector_train])))
            train_f_rmse = float(np.sqrt(np.mean(train_residual[force_selector_train])))
            train_e_rmse_weighted = float(np.sqrt(np.sum(np.multiply(eweights_train, train_residual[energy_selector_train]))))
            train_f_rmse_weighted = float(np.sqrt(np.sum(np.multiply(fweights_train, train_residual[force_selector_train]))))
            test_e_rmse = float(np.sqrt(np.mean(test_residual[energy_selector_test])))
            test_f_rmse = float(np.sqrt(np.mean(test_residual[force_selector_test])))
            test_e_rmse_weighted = float(np.sqrt(np.sum(np.multiply(eweights_test, test_residual[energy_selector_test]))))
            test_f_rmse_weighted = float(np.sqrt(np.sum(np.multiply(fweights_test, test_residual[force_selector_test]))))
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
            a_stack = torch.cat([a_e_train_w, a_f_train_w])
            b_stack = torch.cat([b_e_train_w, b_f_train_w])
            print(tuple(a_stack.shape), tuple(b_stack.shape), flush=True)

            beta_t = _gpu_solve(a_stack, b_stack.reshape(-1, 1), fit_method, fit_device, rcond)
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

        _write_results(".", mlip, hyperparameters, fold, dict(
            tr_E=train_e_rmse, tr_F=train_f_rmse, te_E=test_e_rmse, te_F=test_f_rmse,
            tr_E_w=train_e_rmse_weighted, tr_F_w=train_f_rmse_weighted,
            te_E_w=test_e_rmse_weighted, te_F_w=test_f_rmse_weighted, beta=beta))

    return hyperparameters


# ============================================================================================
# Incremental R-collecting fit (production, fit_engine='incremental').
#   Per (subset, fold) we keep only augmented-QR R-factors (O(p^2), constant size).  Each new
#   batch is FOLDED into the running R's (TSQR merge); the cumulative design matrix is never
#   re-read.  Every residual is computed as ||R[x;-1]||^2 (a sum of squares -> no catastrophic
#   cancellation, even for tiny energy residuals).  Validated vs the row-based fit() to ~1e-9.
#
#   Channels per fold (each an augmented (p+1)x(p+1) R of [a | b] folded with a per-row weight):
#     solve['E'] (train E, weight w=exp(-E/5)), solve['F'] (train F, weight v=1/max(3,|f|))
#     rmse[(side,type,'0')]  unweighted  -> unweighted RMSE = sqrt(||R[x;-1]||^2 / n)
#     rmse[(side,type,'sq')] sqrt(weight) -> weighted RMSE   = sqrt(scale * ||R[x;-1]||^2)
#   Weight normalization is FACTORED: per-row weights are applied at fold time; the global
#   normalization sums (Sw) and eweight enter only at solve time as the scalars alpha/beta.
# ============================================================================================
class _FoldState:
    SIDES = ("tr", "te"); TYPES = ("E", "F")

    def __init__(self, p, device, dtype):
        import torch
        self.p = p; self.device = device; self.dtype = dtype
        z = lambda: torch.zeros((0, p+1), device=device, dtype=dtype)
        self.solve = {"E": z(), "F": z()}
        self.rmse = {(s, t, v): z() for s in self.SIDES for t in self.TYPES for v in ("0", "sq")}
        self.n  = {(s, t): 0   for s in self.SIDES for t in self.TYPES}
        self.Sw = {(s, t): 0.0 for s in self.SIDES for t in self.TYPES}

    @staticmethod
    def _aug(a, b, w=None):
        import torch
        if w is None:
            return torch.cat([a, b.reshape(-1, 1)], dim=1)
        return torch.cat([w.reshape(-1, 1)*a, (w*b).reshape(-1, 1)], dim=1)

    @staticmethod
    def _merge(R, M):
        import torch
        return torch.linalg.qr(torch.cat([R, M], dim=0), mode='r').R

    def fold_batch(self, a, b, e_mask, part_rows, fold):
        import torch
        train = part_rows != fold; test = part_rows == fold
        for side, smask in (("tr", train), ("te", test)):
            for typ, tmask in (("E", e_mask), ("F", ~e_mask)):
                m = smask & tmask
                if not bool(m.any()):
                    continue
                aa = a[m]; bb = b[m]
                w = torch.exp(-bb/5) if typ == "E" else 1.0/torch.clamp(torch.abs(bb), min=3.0)
                self.n[(side, typ)]  += int(m.sum().item())
                self.Sw[(side, typ)] += float(w.sum().item())
                self.rmse[(side, typ, "0")]  = self._merge(self.rmse[(side, typ, "0")],  self._aug(aa, bb))
                self.rmse[(side, typ, "sq")] = self._merge(self.rmse[(side, typ, "sq")], self._aug(aa, bb, torch.sqrt(w)))
                if side == "tr":
                    self.solve[typ] = self._merge(self.solve[typ], self._aug(aa, bb, w))

    def solve_and_rmse(self, eweight, rcond):
        import torch
        alpha = eweight/self.Sw[("tr", "E")]; beta = 1.0/self.Sw[("tr", "F")]
        R_solve = torch.linalg.qr(torch.cat([alpha*self.solve["E"], beta*self.solve["F"]], dim=0), mode='r').R
        R = R_solve[:self.p, :self.p]; d = R_solve[:self.p, self.p]
        U, S, Vh = torch.linalg.svd(R, full_matrices=False)
        S_inv = torch.where(S > rcond * S[0], 1.0 / S, torch.zeros_like(S))
        x = Vh.mT @ (S_inv * (U.mT @ d))
        xx = torch.cat([x, torch.tensor([-1.0], device=x.device, dtype=x.dtype)])
        out = {"beta": x.detach().cpu().numpy()}
        if not bool(torch.all(torch.isfinite(x))):
            print(f"WARNING: non-finite beta from incremental solve (rank-deficient design?) eweight={eweight}", flush=True)
        for s in self.SIDES:
            for t in self.TYPES:
                esc = eweight if (s == "tr" and t == "E") else 1.0
                sse0 = float(torch.sum(torch.square(self.rmse[(s, t, "0")]  @ xx)).item())
                ssew = float(torch.sum(torch.square(self.rmse[(s, t, "sq")] @ xx)).item())
                out[f"{s}_{t}"]   = (max(sse0, 0.0) / self.n[(s, t)]) ** 0.5
                out[f"{s}_{t}_w"] = (max(esc/self.Sw[(s, t)] * ssew, 0.0)) ** 0.5
        return out

    def to_blob(self):
        return {"n": self.n, "Sw": self.Sw,
                "solve": {k: v.detach().cpu() for k, v in self.solve.items()},
                "rmse":  {k: v.detach().cpu() for k, v in self.rmse.items()}}

    @classmethod
    def from_blob(cls, blob, p, device, dtype):
        st = cls(p, device, dtype)
        st.n = blob["n"]; st.Sw = blob["Sw"]
        st.solve = {k: v.to(device=device, dtype=dtype) for k, v in blob["solve"].items()}
        st.rmse  = {k: v.to(device=device, dtype=dtype) for k, v in blob["rmse"].items()}
        return st


def _save_states(states, path):
    import os
    import torch
    tmp = path + ".tmp"
    torch.save([st.to_blob() for st in states], tmp)
    os.replace(tmp, path)   # atomic: a reader on the chain never sees a half-written file


def _load_states(path, p, device, dtype):
    import torch
    blob = torch.load(path, map_location=device, weights_only=False)
    return [_FoldState.from_blob(b, p, device, dtype) for b in blob]


def _write_results(fit_dir, mlip, hyperparameters, fold, res):
    """Append one fold's row to results.csv and write its beta file. Identical format/columns
    to the original fit() so pareto.py's contract is preserved."""
    import numpy as np
    from potmill.tools import rcuts_to_string, nmaxes_to_string, twojmaxes_to_string, lmaxes_to_string
    if mlip == "ACE":
        rcuts, nmaxes, lmaxes, eweight = hyperparameters
    elif mlip == "SNAP":
        rcuts, twojmaxes, eweight = hyperparameters
    with open(f"{fit_dir}/results.csv", "a") as file:
        if mlip == "ACE":
            results_line = "%i," % fold + rcuts_to_string(rcuts, delimiter=",") + "," + \
                nmaxes_to_string(nmaxes, delimiter=",") + "," + lmaxes_to_string(lmaxes, delimiter=",")
        elif mlip == "SNAP":
            results_line = "%i," % fold + rcuts_to_string(rcuts, delimiter=",") + "," + \
                twojmaxes_to_string(twojmaxes, delimiter=",")
        results_line += ",%.3f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n" % \
            (eweight, res["tr_E"], res["tr_F"], res["te_E"], res["te_F"],
             res["tr_E_w"], res["tr_F_w"], res["te_E_w"], res["te_F_w"])
        file.write(results_line)

    beta_filename = "pot__rcut_" + rcuts_to_string(rcuts, delimiter="_")
    if mlip == "ACE":
        beta_filename += "__nmax_" + nmaxes_to_string(nmaxes, delimiter="_") + \
            "__lmax_" + nmaxes_to_string(nmaxes, delimiter="_") + "__eweight_%.3f__fold_%i.csv" % (eweight, fold)
    elif mlip == "SNAP":
        beta_filename += "__2jmax_" + twojmaxes_to_string(twojmaxes, delimiter="_") + \
            "__eweight_%.3f__fold_%i.csv" % (eweight, fold)
    np.savetxt(f"{fit_dir}/{beta_filename}", res["beta"])


def foldfit(features_directory, feature_names, b_dependency, subset_hp, eweight_list, mlip,
            batch_ID, prev_state_path, n_fold=3, rcond=1e-13,
            fit_dir_base=None, state_dir=None, fit_device="cuda", fit_method="svd"):
    """One link in a subset's chain: fold batch `batch_ID` into the running per-fold state, then
    solve + RMSE for every (eweight, fold) of this subset at this checkpoint.

    Dependencies (resolved by executorlib before this runs):
      feature_names  -- featurization future for (batch_ID, rcut): the descriptor labels (bnames)
      b_dependency   -- combine_b future for batch_ID: ensures features/{batch_ID}/b_batch.csv exists
      prev_state_path-- previous link's returned state-file path (None at batch 0): the chain edge
    Returns the state-file path (constant per subset; the chain just enforces ordering)."""
    import os
    import numpy as np
    import pandas as pd
    import torch
    from potmill.tools import rcuts_to_string, hyperparameters_to_string

    dtype = torch.float64
    rcut = subset_hp[0]
    rcuts_str = rcuts_to_string(rcut, delimiter='_')
    feature_indices = _feature_indices(mlip, feature_names, subset_hp)
    p = len(feature_indices)

    # ---- batch design matrix (this batch only), column-selected, on device ----
    a_map = np.load(f"{features_directory}{batch_ID}/{rcuts_str}/a.npy", mmap_mode='r')
    a_sel = np.ascontiguousarray(a_map[:, feature_indices])
    a_t = torch.as_tensor(a_sel, dtype=dtype, device=fit_device)

    # ---- batch targets (per-batch b file written by combine_b), aligned row-for-row with a ----
    bb = pd.read_csv(f"{features_directory}{batch_ID}/b_batch.csv", header=None)
    assert bb.shape[0] == a_sel.shape[0], (bb.shape[0], a_sel.shape[0], batch_ID)
    local_idx = bb[0].values; job_id = bb[1].values; bval = bb[2].values
    b_t = torch.as_tensor(bval, dtype=dtype, device=fit_device)
    e_mask = torch.as_tensor(local_idx == 0, device=fit_device)                 # energy = local idx 0
    part = torch.as_tensor([config_fold(j, n_fold) for j in job_id], device=fit_device)

    # ---- load running state (k folds) or start fresh ----
    if prev_state_path and os.path.exists(prev_state_path):
        states = _load_states(prev_state_path, p, fit_device, dtype)
    else:
        states = [_FoldState(p, fit_device, dtype) for _ in range(n_fold)]

    # ---- fold this batch into every fold's state ----
    for f in range(n_fold):
        states[f].fold_batch(a_t, b_t, e_mask, part, f)

    # ---- persist updated state (the chain edge) ----
    os.makedirs(state_dir, exist_ok=True)
    new_path = f"{state_dir}/state.pt"
    _save_states(states, new_path)

    # ---- solve + RMSE for every (eweight, fold), write results.csv + beta ----
    for eweight in eweight_list:
        hp = (subset_hp + [eweight]) if mlip == "ACE" else (subset_hp + [eweight])
        fit_dir = fit_dir_base + hyperparameters_to_string(mlip, hp, delimiter='_')
        os.makedirs(fit_dir, exist_ok=True)
        for f in range(n_fold):
            res = states[f].solve_and_rmse(eweight, rcond)
            _write_results(fit_dir, mlip, hp, f, res)
    return new_path
