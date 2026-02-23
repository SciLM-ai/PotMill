import os
import random
import pickle
import numpy as np
import ase.build
from ase.io import write
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
from autopiad.structuregen.renorm import _check_distances_binary, _check_distances_multi


class EntropyMaximizer:
    """Phase 2: Monte Carlo entropy maximization.

    Generates candidate atomic configurations and accepts those that decrease
    the negative log-determinant of the normalized information matrix (i.e.,
    increase information entropy in the descriptor space).

    Uses adaptive K scaling: K increases when many candidates are rejected
    (to explore more aggressively) and decreases when distance constraints
    are frequently violated.

    Supports two methods:
    - 'binary': Fixed element pair with NN-distance-based radii.
    - 'multi_element': Arbitrary elements with pseudo-species remapping.
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
        self.n_optimizer_iterations = config.get('n_optimizer_iterations', 5000)
        self.strict_entropy_decrease = bool(config.get('strict_entropy_decrease',
                                                       1 if self.method == 'binary' else 0))

        # Adaptive K parameters
        self.K = config.get('K_init', 1.0)
        self.current_det = 0
        self.i_reject_dist = 0
        self.i_reject_improve = 0
        self.n_reject_improve = 0
        self.n_reject_dist = 0
        self.i_accept = 0
        self.n_accept = 0
        self.n_det_all = []
        self.n_det_acc = []
        self.n_cond_all = []
        self.n_cond_acc = []

        # Load renormalization data from Phase 1
        random_manager = pickle.load(open("random-manager.p", "rb"))
        self.mean = random_manager.sum / random_manager.count
        self.renorm = pickle.load(open("renormalization_matrix.pckl", "rb"))

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

        # Write SNAP descriptor file
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

        # Get fixed radii for optimizer (uses grid index 10 with step 0.18)
        (self.core_radius_0, self.core_radius_1, self.core_radius_cross,
         _, _) = self.sampler.sample_radii_fixed(
            2, 1, grid_index=10, scale_step=0.18)

        self.manager = CNManager(
            self.n_descriptors_tot, energy_mode=self.energy_mode,
            mean=self.mean, renorm=self.renorm, epsilon=self.epsilon)

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

        self.manager = CNManager(
            self.n_descriptors_tot, energy_mode=self.energy_mode,
            mean=self.mean, renorm=self.renorm, epsilon=self.epsilon)

    def looping(self):
        """Run Phase 2: entropy-maximizing Monte Carlo search.

        Yields ASE Atoms objects for accepted configurations.
        """
        for i in range(self.n_optimizer_iterations):
            yield from self._create_configuration(i)

        pickle.dump(self.manager.data, open("d-opti.p", "wb"))

    def _create_configuration(self, i):
        if self.method == 'binary':
            yield from self._create_binary_config(i)
        else:
            yield from self._create_multi_element_config(i)

    def _create_binary_config(self, i):
        n_atoms = random.choice(self.N_atoms)
        shape = random.choice(self.shapes)
        n_first = random.choice(range(1, n_atoms))

        # Use fixed radii stored at init time
        atom_types = {e: idx + 1 for idx, e in enumerate(self.elements)}
        symbols = (int(n_first) * [self.elements[0]] +
                   int(n_atoms - n_first) * [self.elements[1]])

        min_dist_0 = self.core_radius_0 * 0.9
        min_dist_1 = self.core_radius_1 * 0.9
        min_dist_cross = self.core_radius_cross * 0.9

        # Compute target volume (matching original binary_entropy logic)
        volume_0 = ((np.sqrt(2) * self.core_radius_0) ** 3) / 4.0
        volume_1 = ((np.sqrt(2) * self.core_radius_1) ** 3) / 4.0
        target_volume = ((n_first * volume_0 + (n_atoms - n_first) * volume_1)
                         / n_atoms) * random.uniform(1.0, 2.0)

        print(n_atoms, n_first, shape, target_volume)

        # Create model with current state
        model = CNModel(
            len(self.elements), self.n_descriptors_tot,
            energy_mode=self.energy_mode, populations=None, mask=None,
            cross_=self.manager.cross, renorm_=self.renorm,
            mean_=self.mean, count_=self.manager.count,
            epsilon_=self.epsilon)

        if self.i_accept < 10:
            model.active = False
            model.K = 0.0
        else:
            model.active = True
            model.K = self.K

        # Generate LAMMPS scripts using binary-specific templates
        mliap_script, zero_script = generate_binary_lammps_scripts(
            self.elements, self.descriptor_filename,
            self.core_radius_0, self.core_radius_1, self.core_radius_cross,
            min_dist_0, min_dist_1, min_dist_cross)

        calculator_relax = LAMMPSlib(
            lmpcmds=zero_script.split("\n"), log_file=None,
            keep_alive=True, atom_types=atom_types)
        calculator_min = EntropyCalculator(
            lmpcmds=mliap_script.split("\n"), log_file=None,
            model=model, keep_alive=True, atom_types=atom_types)

        try:
            print("Generating atoms")
            atoms = generate_random_cell_binary(
                symbols, target_volume=target_volume, shape=shape,
                ratio_of_covalent_radii=0.5)

            print("Relaxing with core repulsion")
            atoms.calc = calculator_relax
            opt = BFGSLineSearch(atoms, logfile=None)
            opt.run(fmax=0.05, steps=50)

            print("Relaxing with entropy model")
            atoms.calc = calculator_min
            opt = BFGSLineSearch(atoms, logfile=None)
            opt.run(fmax=0.05, steps=50)

            print("Compute descriptors and evaluate det")
            d = compute_descriptors(atoms)
            cand_cond, cand_det = self.manager.evaluate(d)

            if self.i_accept > 0:
                print("CANDIDATE:", cand_cond, cand_det,
                      "CURRENT:", self.current_cond, self.current_det)

            dists_cond = _check_distances_binary(
                atoms, self.elements, atom_types,
                min_dist_0, min_dist_1, min_dist_cross)

            file_name = "configs/POSCAR_{}_{}".format(n_atoms, self.i_accept)
            accepted = False

            if self.i_accept <= 10 and dists_cond:
                accepted = True
            elif dists_cond and ((self.strict_entropy_decrease and
                                  cand_det < self.current_det) or
                                 not self.strict_entropy_decrease):
                self.n_reject_dist = 0
                self.n_reject_improve = 0
                self.n_accept += 1
                accepted = True
            else:
                if dists_cond:
                    self.n_reject_improve += 1
                    self.i_reject_improve += 1
                    self.n_det_all.append(cand_det)
                    self.n_cond_all.append(cand_cond)
                else:
                    self.n_reject_dist += 1
                    self.i_reject_dist += 1

            if accepted:
                self.manager.update(d)
                self.current_cond, self.current_det = self.manager.evaluate()
                print("***ACCEPTED:", self.current_cond, self.current_det)
                write(file_name, atoms)
                atoms.calc = None
                yield atoms
                self.i_accept += 1
                self.n_det_acc.append(self.current_det)
                if i > 1:
                    self.n_cond_acc.append(self.current_cond)

            self._adapt_K()
            self._save_state(i, n_atoms, cand_det)

        except Exception as e:
            print(e)

    def _create_multi_element_config(self, i):
        n_atoms = random.choice(self.N_atoms)
        shape = random.choice(self.shapes)

        radii, radii_by_symbol = self.sampler(n_atoms)
        atom_types = {v['symbol']: v['species_id'] for v in radii.values()}

        species_list = [radii[k]['symbol'] for k in sorted(radii.keys())]

        # Write descriptor file for this configuration
        write_mliap_descriptor_multi(
            self.descriptor_filename, radii, self.twojmax, self.bzeroflag)

        mliap_script, zero_script = generate_lammps_scripts(
            radii, self.descriptor_filename)

        # Create model with current state
        model = CNModel(
            n_atoms, self.n_descriptors_tot,
            energy_mode=self.energy_mode, populations=None, mask=None,
            cross_=self.manager.cross, renorm_=self.renorm,
            mean_=self.mean, count_=self.manager.count,
            epsilon_=self.epsilon)

        if self.i_accept < 10:
            model.active = False
            model.K = 0.0
        else:
            model.active = True
            model.K = self.K

        calculator_relax = LAMMPSlib(
            lmpcmds=zero_script.split("\n"), log_file=None,
            keep_alive=True, atom_types=atom_types)
        calculator_min = EntropyCalculator(
            lmpcmds=mliap_script.split("\n"), log_file=None,
            model=model, keep_alive=True, atom_types=atom_types)

        # Target volume from per-atom exclusion volumes
        target_volume = 0.0
        for s in species_list:
            target_volume += radii_by_symbol[s]['volume'] / len(species_list)
        target_volume *= np.random.uniform(
            low=self.volume_scaling[0], high=self.volume_scaling[1])

        ntry = 0
        atoms = None
        while ntry < 10:
            try:
                atoms = generate_random_cell(
                    radii, species_list, target_volume=target_volume,
                    shape=shape)
                break
            except Exception:
                ntry += 1

        if atoms is None:
            return

        try:
            atoms.calc = calculator_relax
            opt = BFGSLineSearch(atoms, logfile=None)
            opt.run(fmax=0.05, steps=30)

            atoms.calc = calculator_min
            opt = BFGSLineSearch(atoms, logfile=None)
            opt.run(fmax=0.05, steps=100)

            d = compute_descriptors(atoms)
            cand_cond, cand_det = self.manager.evaluate(d)

            if self.i_accept > 0:
                print("CANDIDATE:", cand_det, "CURRENT:", self.current_det)

            dists_ok = _check_distances_multi(atoms, radii, species_list)

            accepted = False
            if (self.i_accept <= 10) and dists_ok:
                accepted = True
            elif dists_ok and ((self.strict_entropy_decrease and
                                cand_det < self.current_det) or
                               not self.strict_entropy_decrease):
                self.n_reject_dist = 0
                self.n_reject_improve = 0
                self.n_accept += 1
                accepted = True
            else:
                if dists_ok:
                    self.n_reject_improve += 1
                    self.i_reject_improve += 1
                    self.n_det_all.append(cand_det)
                    self.n_cond_all.append(cand_cond)
                else:
                    self.n_reject_dist += 1
                    self.i_reject_dist += 1

            if accepted:
                self.manager.update(d)
                self.current_cond, self.current_det = self.manager.evaluate()

                # Remap pseudo-species back to original species
                mapping = {v['symbol']: v['original_symbol']
                           for v in radii.values()}
                original_species = atoms.get_chemical_symbols()
                remapped_species = [mapping[k] for k in original_species]
                atoms.set_chemical_symbols(remapped_species)
                atoms = ase.build.sort(atoms)

                file_name = "configs/POSCAR_{}_{}".format(n_atoms, self.i_accept)
                write(file_name, atoms, format='vasp')

                atoms.calc = None
                yield atoms
                self.i_accept += 1
                self.n_det_acc.append(self.current_det)
                if i > 1:
                    self.n_cond_acc.append(self.current_cond)

            self._adapt_K()
            self._save_state(i, n_atoms, cand_det)

        except Exception as e:
            print(e)

    def _adapt_K(self):
        """Adapt the entropy strength parameter K based on rejection statistics."""
        if self.n_reject_improve > 10:
            self.K *= 1.05
            self.n_reject_improve = 0
            self.n_reject_dist = 0
        if self.n_reject_dist > 10:
            self.K *= 0.9
            self.n_reject_improve = 0
            self.n_reject_dist = 0
        if self.n_accept > 10:
            self.K *= 1.005
            self.n_accept = 0

        print("K=", self.K, "n_reject_improve=", self.n_reject_improve,
              "n_reject_dist=", self.n_reject_dist)

    def _save_state(self, i, n_atoms, cand_det):
        """Save current optimization state to files."""
        if i % 10 == 0:
            self.manager.print_status()
            if self.energy_mode:
                pickle.dump(self.manager.data, open("d-opti-energy.p", "wb"))
            else:
                pickle.dump(self.manager.data, open("d-opti-forces.p", "wb"))

        to_save = (
            "i = {}\n"
            "K = {}\n"
            "N_atoms = {}\n"
            "i_accept = {}\n"
            "rejected configs due to distance = {}\n"
            "rejected configs due to determinant = {}\n"
            "current determinant = {}\n"
            "candidate determinant = {}\n"
            "current count = {}"
        ).format(i, self.K, n_atoms, self.i_accept, self.i_reject_dist,
                 self.i_reject_improve, self.current_det, cand_det,
                 self.manager.count)
        with open("current_i_k_n.txt", "w") as f:
            f.write(to_save)

        pickle.dump(self.n_det_all, open("det_all.pckl", "wb"))
        pickle.dump(self.n_det_acc, open("det_acc.pckl", "wb"))
        pickle.dump(self.n_cond_all, open("cond_all.pckl", "wb"))
        pickle.dump(self.n_cond_acc, open("cond_acc.pckl", "wb"))
