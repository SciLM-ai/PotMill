import ase
import ase.build
import ase.calculators.lammpslib
import numpy as np
from ase.calculators.calculator import Calculator, all_changes


def compute_descriptors(atoms):
    """Compute SNAP bispectrum descriptors by triggering a LAMMPS evaluation."""
    atoms.get_potential_energy()
    return atoms.calc.entropy_model.last_bispectrum


class SoftRepulsionCalculator(Calculator):
    """Pure Python implementation of LAMMPS soft pair potential.

    V(r) = A * [1 + cos(pi * r / r_c)] for r < r_c, 0 otherwise.

    Much faster than LAMMPS for small systems (e.g., 12 atoms) because it
    avoids LAMMPS process creation overhead entirely. Used for the initial
    soft relaxation step before the entropy model relaxation.

    Args:
        core_radii: List of core radius for each atom (in atom order).
            Pair cutoff for atoms i,j = core_radii[i] + core_radii[j].
        A: Amplitude of the soft potential (default 10.0).
    """
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, core_radii, A=10.0, **kwargs):
        super().__init__(**kwargs)
        self.core_radii = np.asarray(core_radii, dtype=float)
        self.A = A

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        n = len(self.atoms)
        positions = self.atoms.get_positions()
        cell = self.atoms.get_cell()
        pbc = self.atoms.get_pbc()

        from ase.geometry import get_distances
        D, d = get_distances(positions, cell=cell, pbc=pbc)

        energy = 0.0
        forces = np.zeros((n, 3))

        for i in range(n):
            ri = self.core_radii[i]
            for j in range(i + 1, n):
                r = d[i, j]
                if r < 1e-10:
                    continue
                rc = ri + self.core_radii[j]
                if r < rc:
                    x = np.pi * r / rc
                    energy += self.A * (1.0 + np.cos(x))
                    f_mag = self.A * np.pi / rc * np.sin(x) / r
                    f_vec = f_mag * D[i, j]
                    forces[i] -= f_vec
                    forces[j] += f_vec

        self.results['energy'] = energy
        self.results['forces'] = forces
        self.results['stress'] = np.zeros(6)


class EntropyCalculator(ase.calculators.lammpslib.LAMMPSlib):
    """LAMMPS calculator that loads an MLIAP entropy model.

    Extends LAMMPSlib to activate the MLIAP Python interface and load
    a CNModel instance that computes entropy-based energy and forces
    from SNAP bispectrum descriptors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entropy_model = kwargs['model']

    def initialise_lammps(self, atoms):
        import lammps
        import lammps.mliap
        import numpy as np
        from ase.data import (atomic_numbers as ase_atomic_numbers,
                              chemical_symbols as ase_chemical_symbols,
                              atomic_masses as ase_atomic_masses)
        from ase.calculators.lammps import convert

        if self.parameters.boundary:
            for cmd in self.parameters.lmpcmds:
                if 'boundary' in cmd:
                    break
            else:
                self.lmp.command('boundary ' + self.lammpsbc(atoms))

        self.set_cell(atoms, change=not self.parameters.create_box)

        if self.parameters.atom_types is None:
            s = atoms.get_chemical_symbols()
            _, idx = np.unique(s, return_index=True)
            s_red = np.array(s)[np.sort(idx)].tolist()
            self.parameters.atom_types = {j: i + 1 for i, j in enumerate(s_red)}

        if self.parameters.create_box:
            n_types = len(self.parameters.atom_types)
            create_box_command = 'create_box {} cell'.format(n_types)
            self.lmp.command(create_box_command)

        if self.parameters.create_atoms:
            self.lmp.command('echo none')
            self.rebuild(atoms)
            self.lmp.command('echo log')
        else:
            self.previous_atoms_numbers = atoms.numbers.copy()

        lammps.mliap.activate_mliappy(self.lmp)
        for cmd in self.parameters.lmpcmds:
            self.lmp.command(cmd)
        lammps.mliap.load_model(self.entropy_model)

        for sym in self.parameters.atom_types:
            if self.parameters.atom_type_masses is None:
                mass = ase_atomic_masses[ase_atomic_numbers[sym]]
            else:
                mass = self.parameters.atom_type_masses[sym]
            self.lmp.command('mass %d %.30f' % (
                self.parameters.atom_types[sym],
                convert(mass, "mass", "ASE", self.units)))

        self.lmp.command('variable pxx equal pxx')
        self.lmp.command('variable pyy equal pyy')
        self.lmp.command('variable pzz equal pzz')
        self.lmp.command('variable pxy equal pxy')
        self.lmp.command('variable pxz equal pxz')
        self.lmp.command('variable pyz equal pyz')
        self.lmp.command('thermo_style custom pe pxx emol ecoul')
        self.lmp.command('variable fx atom fx')
        self.lmp.command('variable fy atom fy')
        self.lmp.command('variable fz atom fz')
        self.lmp.command('variable pe equal pe')
        self.lmp.command("neigh_modify delay 0 every 1 check yes")
        self.initialized = True


def generate_random_cell_binary(atom_numbers, target_volume, shape=None,
                                ratio_of_covalent_radii=0.5):
    """Generate a random cell for binary systems using ASE's StartGenerator.

    Uses closest_distances_generator with covalent radii to determine
    minimum interatomic distances.

    Args:
        atom_numbers: List of chemical symbols (e.g., ["Re","Re","W","W"]).
        target_volume: Target volume per atom.
        shape: Cell aspect ratios [a, b, c]. Defaults to [1, 1, 1].
        ratio_of_covalent_radii: Scaling factor for covalent radii distances.
    """
    from ase_ga.utilities import closest_distances_generator
    from ase_ga.startgenerator import StartGenerator
    from ase.data import atomic_numbers

    if shape is None:
        shape = [1, 1, 1]

    n_atoms = len(atom_numbers)
    a = np.array(shape) + np.random.rand(3)
    angles = np.array([60, 60, 60]) + np.random.rand(3) * 30
    cell = ase.cell.Cell.fromcellpar([a[0], a[1], a[2], angles[0], angles[1], angles[2]])
    current_volume = cell.volume / n_atoms
    cell = ase.cell.Cell(cell * ((1.3 * target_volume / current_volume) ** 0.33333333))

    slab = ase.Atoms()
    slab.cell = cell
    slab.set_pbc([True, True, True])
    unique_atom_types = list(set(
        [x if isinstance(x, int) else atomic_numbers[x] for x in atom_numbers]))
    blmin = closest_distances_generator(
        atom_numbers=unique_atom_types,
        ratio_of_covalent_radii=ratio_of_covalent_radii)
    sg = StartGenerator(slab, atom_numbers, blmin)
    atoms = sg.get_new_candidate(maxiter=1000)
    atoms.set_pbc([True, True, True])
    current_volume = atoms.get_volume() / n_atoms
    atoms.set_cell(
        atoms.get_cell() * (target_volume / current_volume) ** 0.33333333,
        scale_atoms=True)
    return atoms


def generate_random_cell(radii, species, target_volume, shape=None):
    """Generate a random cell for multi-element systems using radii dict.

    Uses per-species core radii to determine minimum interatomic distances.
    Suitable for arbitrary numbers of (pseudo-)species.

    Args:
        radii: Dict mapping type_id -> {'symbol': str, 'r_core': float, ...}
        species: List of species symbols for each atom.
        target_volume: Target volume per atom.
        shape: Cell aspect ratios [a, b, c]. Defaults to [1, 1, 1].
    """
    from ase_ga.startgenerator import StartGenerator

    if shape is None:
        shape = [1, 1, 1]

    species_index_map = {v['symbol']: k for k, v in radii.items()}
    n_atoms = len(species)

    a = np.array(shape) + np.random.rand(3)
    angles = np.array([60, 60, 60]) + np.random.rand(3) * 30
    cell = ase.cell.Cell.fromcellpar([a[0], a[1], a[2], angles[0], angles[1], angles[2]])
    current_volume = cell.volume / n_atoms
    cell = ase.cell.Cell(cell * ((1.3 * target_volume / current_volume) ** 0.33333333))

    slab = ase.Atoms()
    slab.cell = cell
    slab.set_pbc([True, True, True])

    ratio = 0.75
    blmin = {}
    for s_i in species:
        ii = species_index_map[s_i]
        blmin[(ii, ii)] = radii[ii]['r_core'] * ratio
        for s_j in species:
            jj = species_index_map[s_j]
            if ii == jj:
                continue
            if (ii, jj) in blmin:
                continue
            blmin[(ii, jj)] = blmin[(jj, ii)] = (
                radii[ii]['r_core'] + radii[jj]['r_core']) * ratio

    sg = StartGenerator(slab, species, blmin, test_too_far=False)
    atoms = sg.get_new_candidate(maxiter=100)
    atoms.set_pbc([True, True, True])
    current_volume = atoms.get_volume() / n_atoms
    atoms.set_cell(
        atoms.get_cell() * (target_volume / current_volume) ** 0.33333333,
        scale_atoms=True)
    return atoms
