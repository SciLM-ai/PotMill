import numpy as np
import os, glob, sys, json, traceback
import xml.etree.ElementTree as ET
from ase.io import read, write
from shutil import make_archive
from ase.calculators.lammpsrun import LAMMPS


def lammps(start_path, input_file, job_id, first_index):

    os.environ['ASE_LAMMPSRUN_COMMAND'] = start_path+"run_lammps_ase.sh"
    # os.environ['ASE_LAMMPSRUN_COMMAND']="flux run -n 1 -c 1 -g 0 /users/baghishov/codes/lammps/build-fitsnap/lmp"
    # os.environ['ASE_LAMMPSRUN_COMMAND']="mpirun -np 1 /users/baghishov/codes/lammps/build-fitsnap/lmp"
    # os.environ['ASE_LAMMPSRUN_COMMAND']="/users/baghishov/codes/lammps/build-fitsnap/lmp"

    # #check whether this task has been executed already. If so, skip it
    # if os.path.isfile("b"):
    #     sys.exit(0)

    #have to set this up accordingly
    atom_type_mapping = ["Be"]
    ace_file = start_path + "pot.yace"
    pair_coeff = ['* * ' + ace_file + ' ' + ' '.join(atom_type_mapping)]
    files = [ace_file]
    parameters = {'pair_style': 'pace', 'pair_coeff': pair_coeff}
    calc = LAMMPS(files=files, **parameters, keep_tmp_files=True, tmp_dir="lammps_temp", log_file="log.lammps")

    atoms = read(input_file, index=0, format='vasp')
    atoms.pbc = True
    atoms.calc = calc

    print("RUN DIRECTORY: ", os.getcwd(), " INPUT FILE: ", input_file, flush=True)

    #execute the calculation
    try:
        ener = atoms.get_potential_energy()
        forces = atoms.get_forces().ravel()

        n_atoms = len(atoms)
        b = np.vstack([np.arange(first_index,first_index+1+3*n_atoms),
                        np.full(1+3*n_atoms,job_id),
                        np.concatenate([np.array([ener])/n_atoms,forces])]).T
        np.savetxt("b", b, delimiter=',', fmt=['%i','%i','%.10f'])

        #write the output in ASE traj format
        write("atoms_%i.traj" % job_id,images=atoms,format='traj')

        # #look into using Custodian here to do error detection/validation
    except Exception:
        print("Error while running LAMMPS or writing the output files", flush=True)
        traceback.print_exc()

    return job_id