import os
import random
import pickle
import numpy as np
import ase.io.lammpsdata
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.calculators.lammpslib import LAMMPSlib

from autopiad.structuregen.model import CNModel, CNManager
from autopiad.structuregen.calculator import (
    EntropyCalculator, compute_descriptors,
    generate_random_cell_binary, generate_random_cell)
from autopiad.structuregen.lammps_utils import (
    compute_n_descriptors, write_mliap_descriptor,
    generate_lammps_scripts, write_mliap_descriptor_multi,
    generate_binary_lammps_scripts)
from autopiad.structuregen.samplers import BinaryRadiusSampler, MendeleevUniformRadiusSampler


class RandomEntropyInitializer:
    """Phase 1: Generate random configurations to build normalization matrices.

    Creates random atomic configurations with varying compositions and cell
    shapes, computes their SNAP bispectrum descriptors, and accumulates
    statistics to build the renormalization matrix used in Phase 2.

    Supports two methods:
    - 'binary': Fixed pair of elements with NN-distance-based radii and
      chemically-aware SNAP descriptors (chemflag=1).
    - 'multi_element': Arbitrary elements with Mendeleev-based radius
      sampling and pseudo-species SNAP descriptors.
    """

    def __init__(self, config):
        self.method = config.get('method', 'binary')
        self.elements = config['elements']
        self.twojmax = config.get('twojmax', 4 if self.method == 'binary' else 8)
        self.chemflag = config.get('chemflag', 1 if self.method == 'binary' else 0)
        self.bzeroflag = config.get('bzeroflag', 0 if self.method == 'binary' else 1)
        self.energy_mode = bool(config.get('energy_mode',
                                           0 if self.method == 'binary' else 1))
        self.epsilon = config.get('epsilon', 1e-4)
        self.n_renorm_configs = config.get('n_renorm_configs',
                                           10 if self.method == 'binary' else 100)

        if self.method == 'binary':
            self._init_binary(config)
        else:
            self._init_multi_element(config)

    def _init_binary(self, config):
        n_elements = len(self.elements)
        self.n_descriptors_tot = compute_n_descriptors(
            self.twojmax, n_elements, self.chemflag, self.bzeroflag)

        self.sampler = BinaryRadiusSampler(self.elements)
        self.atom_types = {e: i + 1 for i, e in enumerate(self.elements)}
        self.N_atoms = range(
            config.get('n_atoms_min', 2),
            config.get('n_atoms_max', 25) + 1)
        self.shapes = [[4, 1, 1], [1, 1, 1], [3, 3, 1]]

        # Write SNAP descriptor file (fixed for all binary configurations)
        nn_dists = self.sampler.nn_dists
        rcuts = {e: nn_dists[e] * 2 for e in self.elements}
        rcut_max = max(rcuts.values())
        radelems_ref = 0.5
        radelems = [np.round((rcuts[e] * radelems_ref) / rcut_max, 4)
                    for e in self.elements]

        self.descriptor_filename = "entropy.mliap.descriptor"
        write_mliap_descriptor(
            self.descriptor_filename, self.elements, rcut_max, self.twojmax,
            radelems, self.chemflag, self.bzeroflag)

    def _init_multi_element(self, config):
        self.n_atoms = config.get('n_atoms', config.get('n_atoms_max', 12))
        self.n_descriptors_tot = compute_n_descriptors(
            self.twojmax, self.n_atoms, self.chemflag, self.bzeroflag)
        self.N_atoms = [self.n_atoms]
        self.shapes = [[2, 1, 1], [1, 1, 1], [2, 2, 1]]

        width = config.get('radius_width', 0.3)
        a_beta = config.get('radius_beta_a', 1.25)
        b_beta = config.get('radius_beta_b', 1.25)
        self.sampler = MendeleevUniformRadiusSampler(
            self.elements, width, a_beta, b_beta)

        self.volume_scaling = [
            config.get('volume_scaling_min', 1.0),
            config.get('volume_scaling_max', 3.5),
        ]

        self.descriptor_filename = "entropy.mliap.descriptor"

    def looping(self):
        """Run Phase 1: generate random configs and build renormalization matrix."""
        self.manager = CNManager(self.n_descriptors_tot)
        n_elems = len(self.elements) if self.method == 'binary' else self.n_atoms
        self.model = CNModel(
            n_elems, self.n_descriptors_tot,
            energy_mode=self.energy_mode, populations=None, mask=None,
            cross_=None, renorm_=None, mean_=None, count_=0,
            epsilon_=self.epsilon)

        i = 0
        while i < self.n_renorm_configs:
            i = self._create_configuration(i)

        mean = self.manager.sum / self.manager.count
        covariance = self.manager.cross / self.manager.count - np.outer(mean, mean)
        var = np.sqrt(np.diagonal(covariance))
        renorm = np.outer(var, var)

        pickle.dump(renorm, open("renormalization_matrix.pckl", "wb"))
        pickle.dump(self.manager.data, open("random-ref-data.p", "wb"))
        pickle.dump(self.manager, open("random-manager.p", "wb"))

    def _create_configuration(self, i):
        if self.method == 'binary':
            return self._create_binary_config(i)
        else:
            return self._create_multi_element_config(i)

    def _create_binary_config(self, i):
        n_atoms = random.choice(self.N_atoms)
        shape = random.choice(self.shapes)
        n_first = random.choice(range(1, n_atoms))

        (core_radius_0, core_radius_1, core_radius_cross,
         atom_types, symbols) = self.sampler.sample_radii(n_atoms, n_first)

        min_dist_0 = core_radius_0 * 0.9
        min_dist_1 = core_radius_1 * 0.9
        min_dist_cross = core_radius_cross * 0.9

        # Compute target volume (matching original binary_entropy logic)
        volume_0 = ((np.sqrt(2) * core_radius_0) ** 3) / 4.0
        volume_1 = ((np.sqrt(2) * core_radius_1) ** 3) / 4.0
        target_volume = ((n_first * volume_0 + (n_atoms - n_first) * volume_1)
                         / n_atoms) * random.uniform(0.9, 1.9)

        # Generate LAMMPS scripts using binary-specific templates
        mliap_script, zero_script = generate_binary_lammps_scripts(
            self.elements, self.descriptor_filename,
            core_radius_0, core_radius_1, core_radius_cross,
            min_dist_0, min_dist_1, min_dist_cross)

        calculator_relax = LAMMPSlib(
            lmpcmds=zero_script.split("\n"), log_file="lammpslog",
            keep_alive=True, atom_types=atom_types)
        calculator_min = EntropyCalculator(
            lmpcmds=mliap_script.split("\n"), log_file=None,
            model=self.model, keep_alive=True, atom_types=atom_types)

        try:
            print("Generating atoms:", n_atoms, n_first, shape, target_volume)
            atoms = generate_random_cell_binary(
                symbols, target_volume=target_volume, shape=shape,
                ratio_of_covalent_radii=0.5)

            print("Relaxing with core repulsion")
            atoms.calc = calculator_relax
            opt = BFGSLineSearch(atoms, logfile="log_relax")
            opt.run(fmax=0.05, steps=50)

            atoms.calc = calculator_min
            d = compute_descriptors(atoms)

            if _check_distances_binary(atoms, self.elements, atom_types,
                                       min_dist_0, min_dist_1, min_dist_cross):
                print("Compute descriptors and update")
                self.manager.update(d)
                ase.io.lammpsdata.write_lammps_data(
                    "renorm_configs/renorm_config_{}.dat".format(i), atoms)
                i += 1
        except Exception as e:
            print(e)

        return i

    def _create_multi_element_config(self, i):
        n_atoms = random.choice(self.N_atoms)
        shape = random.choice(self.shapes)

        radii, radii_by_symbol = self.sampler(n_atoms)
        atom_types = {v['symbol']: k for k, v in radii_by_symbol.items()
                      if k in [radii[kk]['symbol'] for kk in radii]}
        # Simpler: build atom_types from radii directly
        atom_types = {v['symbol']: v['species_id'] for v in radii.values()}

        species_list = [radii[k]['symbol'] for k in sorted(radii.keys())]

        # Write descriptor file for this configuration's pseudo-species
        write_mliap_descriptor_multi(
            self.descriptor_filename, radii, self.twojmax, self.bzeroflag)

        mliap_script, zero_script = generate_lammps_scripts(
            radii, self.descriptor_filename)

        calculator_relax = LAMMPSlib(
            lmpcmds=zero_script.split("\n"), log_file=None,
            keep_alive=True, atom_types=atom_types)
        calculator_min = EntropyCalculator(
            lmpcmds=mliap_script.split("\n"), log_file="lammps.log",
            model=self.model, keep_alive=True, atom_types=atom_types)

        # Target volume from per-atom exclusion volumes
        target_volume = 0.0
        for s in species_list:
            target_volume += radii_by_symbol[s]['volume'] / len(species_list)
        target_volume *= np.random.uniform(
            low=self.volume_scaling[0], high=self.volume_scaling[1])

        try:
            atoms = generate_random_cell(
                radii, species_list, target_volume, shape=shape)
            print(i, atoms)

            atoms.calc = calculator_relax
            atoms.get_potential_energy()
            opt = BFGSLineSearch(atoms, logfile="min.log")
            opt.run(fmax=0.05, steps=30)

            atoms.calc = calculator_min
            d = compute_descriptors(atoms)

            if _check_distances_multi(atoms, radii, species_list):
                self.manager.update(d)
                i += 1
        except Exception as e:
            print(e)

        return i


def _check_distances_binary(atoms, elements, atom_types,
                            min_dist_0, min_dist_1, min_dist_cross):
    """Check pairwise distances for binary systems.

    Matches the original binary_entropy get_AB_distances + per-pair check
    logic. Each element pair (0-0, 1-1, cross) has its own independent
    minimum distance threshold (single radius, NOT sum-of-radii).

    Args:
        atoms: ASE Atoms object.
        elements: List of two element symbols [elem0, elem1].
        atom_types: Dict mapping symbol -> LAMMPS type id.
        min_dist_0: Minimum allowed distance for elem0-elem0 pairs.
        min_dist_1: Minimum allowed distance for elem1-elem1 pairs.
        min_dist_cross: Minimum allowed distance for elem0-elem1 pairs.
    """
    dists = atoms.get_all_distances(mic=True)
    dists += 1000 * np.identity(len(atoms))
    cell_lengths = atoms.cell.lengths()
    symbols = atoms.get_chemical_symbols()

    indices_0 = [j for j, s in enumerate(symbols) if s == elements[0]]
    indices_1 = [j for j, s in enumerate(symbols) if s == elements[1]]

    # Handle edge cases matching original get_AB_distances logic
    if len(atoms) == 2:
        dists_cross = [dists[i][j] for i in indices_0 for j in indices_1]
        dists_0 = np.min(cell_lengths)
        dists_1 = np.min(cell_lengths)
    else:
        if len(indices_0) == 1:
            dists_0 = np.min(cell_lengths)
            dists_1 = [dists[i][j] for i in indices_1 for j in indices_1
                       if i != j]
            dists_cross = [dists[i][j] for i in indices_0 for j in indices_1]
        elif len(indices_1) == 1:
            dists_1 = np.min(cell_lengths)
            dists_0 = [dists[i][j] for i in indices_0 for j in indices_0
                       if i != j]
            dists_cross = [dists[i][j] for i in indices_0 for j in indices_1]
        else:
            dists_0 = [dists[i][j] for i in indices_0 for j in indices_0
                       if i != j]
            dists_1 = [dists[i][j] for i in indices_1 for j in indices_1
                       if i != j]
            dists_cross = [dists[i][j] for i in indices_0 for j in indices_1]

    if (np.min(dists_0) > min_dist_0 and
            np.min(dists_1) > min_dist_1 and
            np.min(dists_cross) > min_dist_cross):
        return True
    return False


def _check_distances_multi(atoms, radii, species_list):
    """Check pairwise distances for multi-element systems.

    Verifies that for each pair (i, j), the distance exceeds r_min_i + r_min_j.
    """
    n_atoms = len(atoms)
    dists = atoms.get_all_distances(mic=True)
    dists += 1000 * np.identity(n_atoms)

    species_index_map = {v['symbol']: k for k, v in radii.items()}
    for a1 in range(n_atoms):
        for a2 in range(a1 + 1, n_atoms):
            r_min_sum = (radii[species_index_map[species_list[a1]]]['r_min'] +
                         radii[species_index_map[species_list[a2]]]['r_min'])
            if dists[a1, a2] < r_min_sum:
                return False

    return True
