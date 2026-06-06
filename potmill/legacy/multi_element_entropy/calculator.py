import ase
import ase.build
import ase.calculators.lammpslib
import lammps
import lammps.mliap
import scipy
import scipy.linalg
import numpy as np
import pandas as pd
from mpi4py import MPI

import jax.numpy as jaxnp
from jax import grad, jit, vmap
from jax import random
from functools import partial

def compute_descriptors(atoms):
    atoms.get_potential_energy()
    return atoms.calc.entropy_model.last_bispectrum


class EntropyCalculator(ase.calculators.lammpslib.LAMMPSlib):

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.entropy_model=kwargs['model']

    def initialise_lammps(self, atoms):
        #print("initialise_lammps",flush=True)
        import lammps
        import lammps.mliap
        import numpy as np
        from ase.data import (atomic_numbers as ase_atomic_numbers,
                    chemical_symbols as ase_chemical_symbols,
                    atomic_masses as ase_atomic_masses)
        from ase.calculators.lammps import convert

        # Initialising commands
        if self.parameters.boundary:
            # if the boundary command is in the supplied commands use that
            # otherwise use atoms pbc
            for cmd in self.parameters.lmpcmds:
                if 'boundary' in cmd:
                    break
            else:
                self.lmp.command('boundary ' + self.lammpsbc(atoms))

        # Initialize cell
        self.set_cell(atoms, change=not self.parameters.create_box)

        if self.parameters.atom_types is None:
            # if None is given, create from atoms object in order of appearance
            s = atoms.get_chemical_symbols()
            _, idx = np.unique(s, return_index=True)
            s_red = np.array(s)[np.sort(idx)].tolist()
            self.parameters.atom_types = {j: i + 1 for i, j in enumerate(s_red)}

        # Initialize box
        if self.parameters.create_box:
            # count number of known types
            n_types = len(self.parameters.atom_types)
            create_box_command = 'create_box {} cell'.format(n_types)
            self.lmp.command(create_box_command)

        # Initialize the atoms with their types
        # positions do not matter here
        if self.parameters.create_atoms:
            self.lmp.command('echo none')  # don't echo the atom positions
            self.rebuild(atoms)
            self.lmp.command('echo log')  # turn back on
        else:
            self.previous_atoms_numbers = atoms.numbers.copy()

        lammps.mliap.activate_mliappy(self.lmp)
        # execute the user commands
        for cmd in self.parameters.lmpcmds:
            #print("self.parameters.lmpcmds: ",cmd)
            self.lmp.command(cmd)
        lammps.mliap.load_model(self.entropy_model)
        # Set masses after user commands, e.g. to override
        # EAM-provided masses
        for sym in self.parameters.atom_types:
            if self.parameters.atom_type_masses is None:
                mass = ase_atomic_masses[ase_atomic_numbers[sym]]
            else:
                mass = self.parameters.atom_type_masses[sym]
            self.lmp.command('mass %d %.30f' % (
                self.parameters.atom_types[sym],
                convert(mass, "mass", "ASE", self.units)))

        # Define force & energy variables for extraction
        self.lmp.command('variable pxx equal pxx')
        self.lmp.command('variable pyy equal pyy')
        self.lmp.command('variable pzz equal pzz')
        self.lmp.command('variable pxy equal pxy')
        self.lmp.command('variable pxz equal pxz')
        self.lmp.command('variable pyz equal pyz')

        # I am not sure why we need this next line but LAMMPS will
        # raise an error if it is not there. Perhaps it is needed to
        # ensure the cell stresses are calculated
        self.lmp.command('thermo_style custom pe pxx emol ecoul')

        self.lmp.command('variable fx atom fx')
        self.lmp.command('variable fy atom fy')
        self.lmp.command('variable fz atom fz')

        # do we need this if we extract from a global ?
        self.lmp.command('variable pe equal pe')

        self.lmp.command("neigh_modify delay 0 every 1 check yes")

        self.initialized = True


#generate a random cell with a given volume per atom and number of atoms
def generate_random_cell(atom_numbers, target_volume, shape=[1,1,1], ratio_of_covalent_radii=0.5):
    from ase_ga.utilities import closest_distances_generator
    from ase_ga.utilities import get_all_atom_types
    from ase_ga.startgenerator import StartGenerator
    from ase.data import atomic_numbers
    n_atoms=len(atom_numbers)
    #generate a random box
    a=(np.array(shape)+np.random.rand(3))
    angles=np.array([60,60,60])+np.random.rand(3)*30
    #angles=[90,90,90]
    cell = ase.cell.Cell.fromcellpar([a[0],a[1],a[2], angles[0], angles[1], angles[2]])
    #scale the box to reach the target density with the target number of atoms
    #overshoot the volume at first to make things easier
    current_volume = cell.volume/n_atoms
    cell = ase.cell.Cell(cell*((1.3*target_volume/current_volume)**0.33333333))
    current_volume = cell.volume/n_atoms
    #print(target_volume,current_volume,flush=True)

    #fill box with atoms
    slab = ase.Atoms()
    slab.cell = cell
    slab.set_pbc([True, True, True])
    unique_atom_types = list(set([ x if isinstance(x,int) else  atomic_numbers[x] for x in atom_numbers ]))
    blmin = closest_distances_generator(atom_numbers=unique_atom_types,ratio_of_covalent_radii=ratio_of_covalent_radii)
    print(blmin, flush=True)
    sg = StartGenerator(slab, atom_numbers, blmin)
    atoms = sg.get_new_candidate(maxiter=1000)
    #use pbc by default
    atoms.set_pbc([True, True, True])
    current_volume = atoms.get_volume()/n_atoms
    #fix the volume
    atoms.set_cell(atoms.get_cell()*(target_volume/current_volume)**0.33333333,scale_atoms=True)

    return atoms

