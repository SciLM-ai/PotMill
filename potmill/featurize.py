
def init_featurize():
    """executorlib init_function: pre-import all dependencies once per worker."""
    import os
    import numpy as np
    from mpi4py import MPI
    from fitsnap3lib.fitsnap import FitSnap
    from fitsnap3lib.scrapers.ase_funcs import ase_scraper
    from potmill.tools import rcuts_to_string
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return {"comm": comm, "rank": rank, "size": size}


def featurize(atoms_traj, config, fitsnap_config, rcuts, feature_directory,
              only_cost=False, hyperparameters_noeweight=None, cost_nstructures=None,
              comm=None, rank=0, size=1):

    if comm is None:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    import os
    import numpy as np
    from fitsnap3lib.fitsnap import FitSnap
    from fitsnap3lib.scrapers.ase_funcs import ase_scraper
    from potmill.tools import rcuts_to_string

    os.chdir(feature_directory)

    if isinstance(atoms_traj, dict):
        atoms_traj = atoms_traj["atoms"]
    elif isinstance(atoms_traj, list):
        # If labeling was batched (label_batch_size>1), each labeling task returns a list
        # of N dicts (from uma_batch), so exe.batched yields a list-of-lists. Flatten to
        # the list-of-dicts shape this function expects -- otherwise len(atoms_traj) counts
        # outer lists (=combine_b_n) instead of configs (=batch_size), and cost_nstructures
        # slices outer lists instead of capping atoms (which causes cost tasks to process
        # 20x more atoms than intended and stall the dynamic-exe queue).
        if atoms_traj and isinstance(atoms_traj[0], list):
            atoms_traj = [item for sublist in atoms_traj for item in sublist]
        if isinstance(atoms_traj[0], dict):
            atoms_traj = [atoms["atoms"] for atoms in atoms_traj]

    # cost is only a featurization-TIMING probe for the Pareto cost axis -- it does not need the whole
    # batch. Cap it at cost_nstructures (set in __main__) so the one-per-subset cost tasks finish in
    # seconds instead of ~6.5 min each on the full ~1000-config batch (which otherwise hogs the
    # dynamic-exe cores and starves combine_b). Real featurization (only_cost=False) is untouched.
    if only_cost and cost_nstructures is not None and isinstance(atoms_traj, list):
        atoms_traj = atoms_traj[:cost_nstructures]

    configs_num = len(atoms_traj)
    ratio = configs_num//size
    rem = configs_num%size
    a1 = rank*ratio + min(rank,rem)
    a2 = (rank+1)*ratio + min(rank,rem-1) + 1

    print("rcuts = " + rcuts_to_string(rcuts), flush=True)
    if config['FitSNAP']['mlip'] == "ACE":
        if len(rcuts) == 1:
            fitsnap_config["ACE"]["rcutfac"] = rcuts_to_string(rcuts*(int(fitsnap_config["ACE"]["numTypes"])**2))
        else:
            fitsnap_config["ACE"]["rcutfac"] = rcuts_to_string(rcuts)
    elif config['FitSNAP']['mlip'] == "SNAP":
        fitsnap_config["BISPECTRUM"]["radelem"] = rcuts_to_string(rcuts)

    fs = FitSnap(fitsnap_config, comm=comm, arglist=["--overwrite"])
    fs.data = ase_scraper(atoms_traj[a1:a2])
    fs.process_configs(allgather=True)

    comm.Barrier()

    if rank == 0:
        os.system("rm -rf coupling_coefficients.yace *.pickle")
        if not only_cost:
            a_arr = fs.pt.shared_arrays["a"].array
            n_bad = int(np.sum(~np.isfinite(a_arr)))
            if n_bad:
                print(f"WARNING: featurize {os.getcwd()} a.npy has {n_bad} non-finite "
                      f"descriptor values (degenerate config?)", flush=True)
            np.save("a.npy", a_arr)

            bnames = []
            if config['FitSNAP']['mlip'] == "ACE":
                numtypes = fs.config.sections["ACE"].numtypes
                ncoeff = len(fs.config.sections["ACE"].blist)//numtypes
                for ielem in range(numtypes):
                    bstart = ielem * ncoeff
                    bstop = bstart + ncoeff
                    bnames += [[0]] + fs.config.sections["ACE"].blist[bstart:bstop]
            elif config['FitSNAP']['mlip'] == "SNAP":
                numtypes = fs.config.sections["BISPECTRUM"].numtypes
                ncoeff = fs.config.sections["BISPECTRUM"].ncoeff
                for ielem in range(numtypes):
                    bstart = ielem * ncoeff
                    bstop = bstart + ncoeff
                    bnames += [[0]] + fs.config.sections["BISPECTRUM"].blist[bstart:bstop]
            return bnames
        else:
            return hyperparameters_noeweight