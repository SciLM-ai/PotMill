import os
import glob
import numpy as np
import random
import copy
import pickle
import spglib
import pandas as pd
import ase.io.lammpsdata
import matplotlib.pyplot as plt


import jax
import jax.numpy as jaxnp
from jax import grad, jit, vmap
from functools import partial
from ase.io import write
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.calculators.lammpslib import LAMMPSlib
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
            "Pu": 3.1,
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


def write_mliap_descriptor(rcutfac=4.67637, twojmax=6, radelems="0.5 0.5"): #4.812302818
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


class RandomEntropyInitializer:

    def __init__(self):

        rcut_W = NN_dists["W"]*2 #4.67637
        rcut_Be = NN_dists["Re"]*2
        radelems_W  = 0.5
        radelems_Be = np.round((rcut_Be*radelems_W)/rcut_W, 4)
        radelems = str(radelems_Be) + " " + str(radelems_W)

        write_mliap_descriptor(rcutfac=rcut_W, twojmax=4, radelems=radelems)

        #total number of available descriptors
        # bispec_names = calc_bispectrum_names(twojmax=4)
        # n_bispec = len(bispec_names)
        n_bispec = 14
        self.n_descriptors_tot = ((2**3)*n_bispec)
        # base_mask = list(range(self.n_descriptors_tot))
        # n_keep = self.n_descriptors_tot #work on n_keep descriptors at the time

        self.epsilon = 1e-4
        #initial strength of the entropy term
        # K_init=3.
        self.energy_mode = True
        # self.strict_entropy_decrease = True

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

        self.core_radius_W  = NN_dists["W"]
        self.core_radii_Be  = NN_dists["Re"]*np.arange(0.7, 1.8, 0.15)
        NN_dists_WBe = NN_dists["W"]/2. + NN_dists["Re"]/2.
        self.core_radii_WBe = NN_dists_WBe*np.arange(0.7, 1.8, 0.15)
        self.radii_to_sample = [[c_Be, c_WBe] for c_Be in self.core_radii_Be for c_WBe in self.core_radii_WBe]

        self.n_elems=2
        self.atom_types={"Re":1, "W":2}

        self.N_atoms = range(2, 26)
        self.shapes=[[4, 1, 1], [1, 1, 1], [3, 3, 1]]

    
    def looping(self):
        self.manager_random = CNManager(self.n_descriptors_tot)

        self.model = CNModel(self.n_elems, self.n_descriptors_tot, 
                             energy_mode=self.energy_mode, populations=None, 
                             mask=None, cross_=None, 
                             renorm_=None, mean_=None, 
                             count_=0, epsilon_=self.epsilon)

        self.target_V = []
        self.target_D = []
        i = 0
        while i<10:  #TODO: Change this back to 10
            i = self.create_configuration(i)

        mean = self.manager_random.sum/self.manager_random.count
        covariance = self.manager_random.cross/self.manager_random.count-np.outer(mean,mean)
        var = np.sqrt(np.diagonal(covariance))
        renorm = np.outer(var,var)

        pickle.dump(renorm, open("renormalization_matrix.pckl", "wb"))

        pickle.dump(self.manager_random.data, open("random-ref-data.p", "wb"))
        pickle.dump(self.manager_random, open("random-manager.p", "wb"))


    def create_configuration(self, i):
        n_atoms    = random.choice(self.N_atoms)
        shape      = random.choice(self.shapes)
        allowed_Be = range(1, n_atoms)
        n_Be       = random.choice(allowed_Be)
        Be_conc    = float(n_Be/n_atoms)

        rad = random.choice(self.radii_to_sample)
        core_radius_Be  = rad[0]
        core_radius_WBe = rad[1]
        min_distance_W = self.core_radius_W*0.9
        min_distance_Be = core_radius_Be*0.9
        min_distance_WBe = core_radius_WBe*0.9

        Volume_W = ((np.sqrt(2)*self.core_radius_W)**3)/4.
        Volume_Be = ((np.sqrt(2)*core_radius_Be)**3)/4.
        Volume_WBe = (n_Be*Volume_Be + (n_atoms-n_Be)*Volume_W)/n_atoms
        target_volume = Volume_WBe*random.uniform(0.9, 1.9)
        self.target_V.append(target_volume)
        
        print(n_atoms, n_Be, shape, target_volume)
        print(min_distance_W, min_distance_Be, min_distance_WBe)

        symbols = int(n_Be)*["Re"] + int(n_atoms-n_Be)*["W"]

        generate_zero = self.generate_zero_t % (min_distance_Be, min_distance_WBe, min_distance_W)
        calculator_relax = LAMMPSlib(lmpcmds=generate_zero.split("\n"),
                                    log_file="lammpslog",
                                    keep_alive=True,
                                    atom_types=self.atom_types)

        generate_min = self.generate_min_t % (core_radius_Be, core_radius_WBe, self.core_radius_W)
        calculator_min = entropy.EntropyCalculator(lmpcmds=generate_min.split("\n"),
                                                log_file=None,
                                                model=self.model,
                                                keep_alive=True,
                                                atom_types=self.atom_types)

        try:
            print("Generating atoms")
            ratio_of_covalent_radii = 0.5
            atoms = entropy.generate_random_cell(symbols, target_volume=target_volume, shape=shape, ratio_of_covalent_radii=ratio_of_covalent_radii)

            dists_Be, dists_W, dists_WBe = get_AB_distances(atoms)
            print(np.min(dists_W), np.min(dists_Be), np.min(dists_WBe))
            
            #relax with the core repulsion alone
            print("Relaxing with core repulsion")
            atoms.calc = calculator_relax
            opt = BFGSLineSearch(atoms, logfile="log_relax")
            opt.run(fmax=0.05, steps=50)
            
            #No optimizing with the entropy model
            atoms.calc = calculator_min
            d = entropy.compute_descriptors(atoms)
                    
            dists_Be, dists_W, dists_WBe = get_AB_distances(atoms)
            print(np.min(dists_W), np.min(dists_Be), np.min(dists_WBe))

            if (np.min(dists_Be) > min_distance_Be) and (np.min(dists_W) > min_distance_W) and (np.min(dists_WBe) > min_distance_WBe):
                dists_cond = True
            else:
                dists_cond = False

            if dists_cond:
                print("Compute descriptors and update")
                print(np.min(dists_W), np.min(dists_Be), np.min(dists_WBe))
                self.target_D.append([np.min(dists_W), np.min(dists_Be), np.min(dists_WBe)])
                self.manager_random.update(d)
                ase.io.lammpsdata.write_lammps_data("renorm_configs/renorm_config_" + str(i) + ".dat", atoms)
                i += 1
        except Exception as e:
            print(e)
        
        print("\n")

        return i
