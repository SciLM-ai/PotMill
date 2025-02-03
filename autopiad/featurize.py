import numpy as np
from mpi4py import MPI
from fitsnap3lib.fitsnap import FitSnap
from fitsnap3lib.scrapers.ase_funcs import ase_scraper
import pandas as pd
from autopiad.tools import rcuts_to_string


def featurize(config, fitsnap_config, rcuts, start_path):
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # atoms_traj = pd.read_hdf(start_path + config['DATA']['data_path'])['ase_atoms'].to_list()
    atoms_traj = pd.read_pickle(start_path + config["DATA"]["data_path"], compression="gzip")['ase_atoms'].to_list()
    configs_num = len(atoms_traj)
    ratio = configs_num//size
    rem = configs_num%size
    a1 = rank*ratio + min(rank,rem)
    a2 = (rank+1)*ratio + min(rank,rem-1) + 1

    print("rcuts = " + rcuts_to_string(rcuts))
    fitsnap_config["BISPECTRUM"]["radelem"] = rcuts_to_string([rcut/2 for rcut in rcuts])
    fs = FitSnap(fitsnap_config, comm=comm, arglist=["--overwrite"])
    fs.data = ase_scraper(atoms_traj[a1:a2])
    fs.process_configs(allgather=True)
    # fs.output.write_lammps(np.ones((fs.config.sections["BISPECTRUM"].numtypes,
    #                                 fs.config.sections["BISPECTRUM"].ncoeff+1)))

    comm.Barrier()

    if rank == 0:
        np.save("a.npy", fs.pt.shared_arrays["a"].array)
    
    bnames = []
    numtypes = fs.config.sections["BISPECTRUM"].numtypes
    ncoeff = fs.config.sections["BISPECTRUM"].ncoeff
    for ielem in range(numtypes):
        bstart = ielem * ncoeff
        bstop = bstart + ncoeff
        bnames += [[0]] + fs.config.sections["BISPECTRUM"].blist[bstart:bstop]
    
    return bnames