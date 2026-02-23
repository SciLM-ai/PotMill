import numpy as np
import os
from ase import Atoms
from ase.io import read, write
from fairchem.core import FAIRChemCalculator


def uma(start_path, input_file, job_id, first_index, dirpath):

    os.chdir(dirpath)

    calc = FAIRChemCalculator.from_model_checkpoint("uma-s-1p1", task_name="omat", device = "cuda")

    # #check whether this task has been executed already. If so, skip it
    # if os.path.isfile("b"):
    #     sys.exit(0)

    if isinstance(input_file, Atoms):
        atoms = input_file
    else:
        atoms = read(start_path+input_file, index=0, format='vasp')
    atoms.pbc = True
    atoms.calc = calc

    print("RUN DIRECTORY: ", os.getcwd(), " INPUT FILE: ", input_file, flush=True)

    ener = atoms.get_potential_energy()
    forces = atoms.get_forces().ravel()

    n_atoms = len(atoms)
    b = np.vstack([np.arange(first_index,first_index+1+3*n_atoms),
                    np.full(1+3*n_atoms,job_id),
                    np.concatenate([np.array([ener])/n_atoms,forces])]).T
    np.savetxt("b", b, delimiter=',', fmt=['%i','%i','%.10f'])

    #write the output in ASE traj format
    write(f"atoms_{job_id}.traj", images=atoms, format='traj')

    atoms.calc = None

    return {"job_ID":job_id, "atoms":atoms}