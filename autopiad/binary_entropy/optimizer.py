import numpy as np
import random
import pickle

from ase.io import write
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.calculators.lammpslib import LAMMPSlib
from ase.calculators.lammpsrun import LAMMPS
from ase.data import covalent_radii, atomic_numbers

import autopiad.binary_entropy.calculator as entropy
from autopiad.binary_entropy.model import CNModel, CNManager
# from pybispectrum import calc_bispectrum_names


covalent = {"H": covalent_radii[atomic_numbers["H"]],
            "Be": covalent_radii[atomic_numbers["Be"]],
            "C": covalent_radii[atomic_numbers["C"]],
            "Al": covalent_radii[atomic_numbers["Al"]],
            "W": covalent_radii[atomic_numbers["W"]],
            "Re": covalent_radii[atomic_numbers["Re"]],
            "Os": covalent_radii[atomic_numbers["Os"]],
            "Sb": covalent_radii[atomic_numbers["Sb"]],
            "Te": covalent_radii[atomic_numbers["Te"]],
            "Cs": covalent_radii[atomic_numbers["Cs"]],
            "Pu": covalent_radii[atomic_numbers["Pu"]],
            "U": covalent_radii[atomic_numbers["U"]],
            "O": covalent_radii[atomic_numbers["O"]]}

#from https://www.knowledgedoor.com/2/elements_handbook/nearest_neighbor_distance.html =>
# --> p. 21 in Charles Kittel. Introduction to Solid State Physics, 8th edition. Hoboken, NJ: John Wiley & Sons, Inc, 2005.
NN_dists = {"H":0.75,
            "Be": 2.22,
            "C": 1.54,
            "Al": 2.86,
            "W": 2.74,
            "Re": 2.74,
            "Os": 2.68,
            "Pu": 2.5, #3.1,
            "U": 2.75,
            "O": 1.2,
            "Sb": 2.91,
            "Te": 2.86,
            "Cs": 5.24}

#https://www.wiredchemist.com/chemistry/data/metallic-radii
metallic = {"Be": 1.12,
           "Al": 1.43,
           "W": 1.41,
           "Re": 1.35,
           "Os": 1.35,
           "Sb": 1.61,
           "Cs": 2.72,
           "U": 1.56}


def get_AB_distances(atoms):
    
    dists=atoms.get_all_distances(mic=True)
    dists+=1000*np.identity(len(atoms))
    cell_lengths = atoms.cell.lengths()
    
    symbols = atoms.get_chemical_symbols()
    indices_Be = [ind for ind in range(len(symbols)) if symbols[ind]=="Re"]
    indices_W  = [ind for ind in range(len(symbols)) if symbols[ind]=="W"]
    
    if len(atoms) == 2:
        dists_WBe = [dists[i][j] for i in indices_Be for j in indices_W]
        dists_W = dists_Be = np.min(cell_lengths)
        
    else:
        if len(indices_Be)==1:
            dists_Be = np.min(cell_lengths)
            dists_W = [dists[i][j] for i in indices_W for j in indices_W]
            dists_WBe = [dists[i][j] for i in indices_Be for j in indices_W]
        elif len(indices_W)==1:
            dists_W = np.min(cell_lengths)
            dists_Be = [dists[i][j] for i in indices_Be for j in indices_Be]
            dists_WBe = [dists[i][j] for i in indices_Be for j in indices_W]
        else:
            dists_Be = [dists[i][j] for i in indices_Be for j in indices_Be]
            dists_W = [dists[i][j] for i in indices_W for j in indices_W]
            dists_WBe = [dists[i][j] for i in indices_Be for j in indices_W]
        
    return dists_Be, dists_W, dists_WBe



def write_mliap_descriptor(rcutfac=4.67637, twojmax=6, radelems="0.5 0.5"):
    with open("WBe.mliap.descriptor", "w") as f:
        f.write("# DATE: 2014-09-05 UNITS: metal CONTRIBUTOR: Aidan Thompson athomps@sandia.gov CITATION: Thompson, Swiler, Trott, Foiles and Tucker, arxiv.org, 1409.3880 (2014)\n")
        f.write("# LAMMPS SNAP parameters for Ta_Cand06A\n")
        f.write("# required\n")
        f.write("rcutfac {} \n".format(rcutfac))
        f.write("twojmax {} \n".format(twojmax))
        f.write("# elements\n")
        f.write("nelems 2\n")
        f.write("elems Re W \n")
        #f.write("type Be W \n")
        f.write("radelems {} \n".format(radelems))
        f.write("welems 1 1\n")
        f.write("chemflag 1 \n")
        f.write("# optional\n")
        f.write("rfac0 0.99363\n")
        f.write("rmin0 0\n")
        f.write("bzeroflag 0\n")


class EntropyMaximizer:

    def __init__(self):

        rcut_W = NN_dists["W"]*2
        rcut_Be = NN_dists["Re"]*2
        radelems_W  = 0.5
        radelems_Be = np.round((rcut_Be*radelems_W)/rcut_W, 4)
        radelems = str(radelems_Be) + " " + str(radelems_W)
        print(radelems, rcut_Be, rcut_W, radelems_Be*rcut_W*2)

        write_mliap_descriptor(rcutfac=rcut_W, twojmax=4, radelems=radelems)

        # bispec_names = calc_bispectrum_names(twojmax=4)
        # n_bispec = len(bispec_names)
        n_bispec = 14
        self.n_descriptors_tot = ((2**3)*n_bispec)
        # base_mask = list(range(self.n_descriptors_tot))
        # n_keep = self.n_descriptors_tot #work on n_keep descriptors at the time

        self.epsilon = 1e-4
        self.K = 1.

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

        self.energy_mode = False
        self.strict_entropy_decrease = True

        self.generate_zero_t =\
        """
        pair_style soft 5.0
        pair_coeff 1 1 10 %f
        pair_coeff 1 2 8 %f
        pair_coeff 2 2 5 %f
        """

        self.generate_min_t =\
        """
        pair_style hybrid/overlay soft 5 mliap model mliappy LATER descriptor sna WBe.mliap.descriptor
        pair_coeff 1 1 soft 10 %f
        pair_coeff 1 2 soft 10 %f
        pair_coeff 2 2 soft 10 %f
        pair_coeff * * mliap Re W
        compute pe_peratom all pe/atom
        """
        
        # path_renorm = "/vast/home/apasubramanyam/Work/Entropy/binaries_new/W-Re/renorm_configs_matrix/"
        random_data = pickle.load(open("random-ref-data.p", "rb"))
        random_manager = pickle.load(open("random-manager.p", "rb" ))
        self.mean = random_manager.sum/random_manager.count
        self.renorm = pickle.load(open("renormalization_matrix.pckl", "rb"))

        #aggregates the mean and covariances from multiple configurations
        self.manager = CNManager(self.n_descriptors_tot,energy_mode=self.energy_mode,mean=self.mean,renorm=self.renorm,epsilon=self.epsilon)

        self.n_elems = 2
        self.atom_types = {"Re":1, "W":2}
        self.shapes = [[4, 1, 1], [1, 1, 1], [3, 3, 1]]
        self.N_atoms = range(2, 26)

        self.core_radius_W  = NN_dists["W"]
        core_radii_Be  = NN_dists["Re"]*np.arange(0.7, 1.8, 0.18)
        NN_dists_WBe = NN_dists["W"]/2. + NN_dists["Re"]/2.
        core_radii_WBe = NN_dists_WBe*np.arange(0.7, 1.8, 0.18)
        radii_to_sample = [[c_Be, c_WBe] for c_Be in core_radii_Be for c_WBe in core_radii_WBe]

        print(len(radii_to_sample), 
            "\nW:", self.core_radius_W, 
            "\nRe:", core_radii_Be, 
            "\nWRe:", core_radii_WBe)

        sl = 10
        # self.rad = radii_to_sample[sl]
        # core_radius_Be  = self.rad[0]
        # core_radius_WBe = self.rad[1]
        self.core_radius_Be, self.core_radius_WBe = radii_to_sample[sl]

        # min_distance_W   = self.core_radius_W*0.9
        # min_distance_Be  = self.core_radius_Be*0.9
        # min_distance_WBe = self.core_radius_WBe*0.9

        # Volume_Be      = ((np.sqrt(2)*self.core_radius_Be)**3)/4.
        # Volume_WBe     = (1*Volume_Be + 1*Volume_W)/2
        # Volume_WBe2    = ((np.sqrt(2)*self.core_radius_WBe)**3)/4.
        # target_volume  = Volume_WBe*random.uniform(1.0, 2.0)

        # print("core radius: W, Re, WRe", self.core_radius_W, self.core_radius_Be, self.core_radius_WBe)
        # print("Min dist: W, Re, WRe", min_distance_W, min_distance_Be, min_distance_WBe)
        # print("Volumes: W, Re, WRe: ", Volume_W, Volume_Be, Volume_WBe, Volume_WBe2)
        # print("target volume:", Volume_WBe, Volume_WBe2, Volume_WBe*1.0, Volume_WBe*2.0)

    
    def looping(self):
        for i in range(5000):
            yield from self.create_configuration(i)

        pickle.dump(self.manager.data, open("d-opti.p", "wb"))


    def create_configuration(self, i):
        n_atoms    = random.choice(self.N_atoms)
        shape      = random.choice(self.shapes)
        allowed_Be = range(1, n_atoms)
        n_Be       = random.choice(allowed_Be)
        Be_conc    = float(n_Be/n_atoms)

        min_distance_W = self.core_radius_W*0.9
        min_distance_Be = self.core_radius_Be*0.9
        min_distance_WBe = self.core_radius_WBe*0.9

        Volume_W = ((np.sqrt(2)*self.core_radius_W)**3)/4.
        Volume_Be = ((np.sqrt(2)*self.core_radius_Be)**3)/4.
        Volume_WBe = (n_Be*Volume_Be + (n_atoms-n_Be)*Volume_W)/n_atoms
        target_volume = Volume_WBe*random.uniform(1.0, 2.0)
        #target_volume = Volume_WBe*random.uniform(np.round(0.9**3, 2), 1.8)

        print(n_atoms, n_Be, shape, target_volume)

        symbols = int(n_Be)*["Re"] + int(n_atoms-n_Be)*["W"]

        # TODO: Shouldn't it be manager.mean instead of mean which I presume comes from random distribution?
        model = CNModel(self.n_elems, self.n_descriptors_tot, energy_mode=self.energy_mode, populations=None, mask=None,
                        cross_=self.manager.cross, renorm_=self.renorm, mean_=self.mean, count_=self.manager.count, epsilon_=self.epsilon)

        if self.i_accept < 10:
            model.active = False
            model.K = 0.0
        else:
            model.active = True
            model.K = self.K

        generate_zero = self.generate_zero_t % (min_distance_Be, min_distance_WBe, min_distance_W)
        calculator_relax = LAMMPSlib(lmpcmds=generate_zero.split("\n"),
                                     log_file=None,
                                     keep_alive=True,
                                     atom_types=self.atom_types)

        generate_min = self.generate_min_t % (self.core_radius_Be, self.core_radius_WBe, self.core_radius_W)
        calculator_min = entropy.EntropyCalculator(lmpcmds=generate_min.split("\n"),
                                                log_file=None,
                                                model=model,
                                                keep_alive=True,
                                                atom_types=self.atom_types)

        try:
            print("Generating atoms")
            ratio_of_covalent_radii = 0.5
            atoms = entropy.generate_random_cell(symbols, target_volume=target_volume, shape=shape, ratio_of_covalent_radii=ratio_of_covalent_radii)

            #relax with the core repulsion alone
            print("Relaxing with core repulsion")
            atoms.calc = calculator_relax
            opt = BFGSLineSearch(atoms, logfile=None)#logfile="log_relax")
            opt.run(fmax=0.05, steps=50)

            #relax with the entropy model overlapped with the repulsion
            print("Relaxing with entropy model")
            atoms.calc = calculator_min
            opt = BFGSLineSearch(atoms, logfile=None)#logfile="log_entropy_model")
            opt.run(fmax=0.05, steps=50)

            print("Compute descriptors and evaluate det")
            d = entropy.compute_descriptors(atoms)
            cand_cond, cand_det = self.manager.evaluate(d)

            if self.i_accept > 0:
                print("CANDIDATE:", cand_cond, cand_det, "CURRENT:", self.current_cond, self.current_det, "\n")

            dists_Be, dists_W, dists_WBe = get_AB_distances(atoms)
            if (np.min(dists_Be) > min_distance_Be) and (np.min(dists_W) > min_distance_W) and (np.min(dists_WBe) > min_distance_WBe):
                dists_cond = True
            else:
                dists_cond = False

            print("n_atoms, n_Be, Be_conc, target volume: ", n_atoms, n_Be, Be_conc, target_volume)
            print("Candidate distances (W, Be, WBe): ", np.min(dists_W), np.min(dists_Be), np.min(dists_WBe))
            print("Results:i, diff: ", i, cand_det-self.current_det)

            file_name = "configs/POSCAR_"+str(n_atoms)+"_"+str(self.i_accept)
            if self.i_accept<=10 and dists_cond:
                self.manager.update(d)
                self.current_cond, self.current_det = self.manager.evaluate()
                print("***CANDIDATE:", cand_cond, cand_det, "CURRENT:", self.current_cond, self.current_det)
                if self.energy_mode:
                    write(file_name, atoms)
                else:
                    write(file_name, atoms)
                atoms.calc = None
                yield atoms  # yield "entropy/"+file_name
                self.i_accept += 1
                self.n_det_acc.append(self.current_det)
                if i>1:
                    self.n_cond_acc.append(self.current_cond)
            else:
                if dists_cond and ((self.strict_entropy_decrease and cand_det < self.current_det) or not self.strict_entropy_decrease):
                    self.n_reject_dist = 0
                    self.n_reject_improve = 0
                    self.manager.update(d)
                    self.current_cond, self.current_det = self.manager.evaluate()
                    print("***CANDIDATE:", cand_cond, cand_det, "CURRENT:", self.current_cond, self.current_det)
                    if self.energy_mode:
                        write(file_name, atoms)
                    else:
                        write(file_name, atoms)
                    atoms.calc = None
                    yield atoms  # yield "entropy/"+file_name 
                    self.i_accept += 1
                    self.n_accept += 1
                    self.n_det_acc.append(self.current_det)
                    self.n_cond_acc.append(self.current_cond)
                else:
                    if dists_cond:
                        self.n_reject_improve+=1
                        self.i_reject_improve+=1
                        self.n_det_all.append(cand_det)
                        self.n_cond_all.append(cand_cond)
                    else:
                        self.n_reject_dist+=1
                        self.i_reject_dist+=1

            print("K=", self.K, "n_reject_improve=", self.n_reject_improve, "n_reject_dist=", self.n_reject_dist)

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

            print("K=", self.K, "n_reject_improve=", self.n_reject_improve, "n_reject_dist=", self.n_reject_dist, "\n")

            if i%10 == 0:
                self.manager.print_status()
                if self.energy_mode:
                    pickle.dump(self.manager.data, open("d-opti-energy.p", "wb"))
                else:
                    pickle.dump(self.manager.data, open("d-opti-forces.p", "wb"))
            
            # TODO: Figure out this breaking condition (should be in the __main__ file)
            # if model.count>100000:
            #     break

            to_save = "i = "+str(i)+ \
                    "\nK = "+str(self.K)+ \
                    "\nN_atoms = "+str(n_atoms)+ \
                    "\ni_accept = "+str(self.i_accept)+ \
                    "\nrejected configs due to distance = "+str(self.i_reject_dist)+ \
                    "\nrejected configs due to determinant = "+str(self.i_reject_improve)+ \
                    "\ncurrent determinant = "+str(self.current_det)+" #to consider"+ \
                    "\ncandidate determinant = "+str(cand_det)+ \
                    "\ncurrent count = "+str(self.manager.count)
            with open("current_i_k_n.txt", "w") as f:
                f.write(to_save)

            pickle.dump(self.n_det_all, open("det_all.pckl", "wb"))
            pickle.dump(self.n_det_acc, open("det_acc.pckl", "wb"))
            pickle.dump(self.n_cond_all, open("cond_all.pckl", "wb"))
            pickle.dump(self.n_cond_acc, open("cond_acc.pckl", "wb"))

        except Exception as e:
            print(e)
