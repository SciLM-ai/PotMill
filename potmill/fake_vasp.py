import numpy as np
import pandas as pd


def fake_vasp(force_energy_filename, job_id, first_index):
    
    forces,ener = pd.read_pickle(force_energy_filename).iloc[job_id,[0,2]]

    b = np.vstack([np.arange(first_index,first_index+1+forces.size),
                    np.full(1+forces.size,job_id),
                    np.concatenate([np.array([ener])/forces.size,forces.ravel()])]).T
    
    np.savetxt("b", b, delimiter=',', fmt=['%i','%i','%.10f'])

    return job_id