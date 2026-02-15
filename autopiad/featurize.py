
def featurize(atoms_traj, config, fitsnap_config, rcuts, only_cost=False, hyperparameters_noeweight=None):

    import os
    import numpy as np
    from mpi4py import MPI
    from fitsnap3lib.fitsnap import FitSnap
    from fitsnap3lib.scrapers.ase_funcs import ase_scraper
    from autopiad.tools import rcuts_to_string

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if isinstance(atoms_traj, dict):
        atoms_traj = atoms_traj["atoms"]
    elif isinstance(atoms_traj, list):
        if isinstance(atoms_traj[0], dict):
            atoms_traj = [atoms["atoms"] for atoms in atoms_traj] 

    configs_num = len(atoms_traj)
    ratio = configs_num//size
    rem = configs_num%size
    a1 = rank*ratio + min(rank,rem)
    a2 = (rank+1)*ratio + min(rank,rem-1) + 1

    print("rcuts = " + rcuts_to_string(rcuts))
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
            np.save("a.npy", fs.pt.shared_arrays["a"].array)

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