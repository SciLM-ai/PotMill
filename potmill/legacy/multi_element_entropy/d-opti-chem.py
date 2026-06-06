# %%
import traceback
import entropy_utils
import numpy as np
import random
import pickle
import copy

import jax
from jax.config import config
config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')


import jax.numpy as jaxnp
from jax import grad, jit, vmap
from functools import partial
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd 
import scipy
import sys

# %%
# unnormalized probability for having key element

# weights need to be discussed
# 1 H 
# 6 noble gas elements
# 35 sp block elements, first 14 most common
# about 30 sd block elements
# what to do with all the other elements ?


# currently: took out magnetic 3d elements and f elements
"""
species_prob = {
'H': 10.0,
'He': 1.0,
'Li': 5.0,
'Be': 5.0,
'B': 5.0,
'C': 10.0,
'N': 5.0,
'O': 5.0,
'F': 5.0,
'Ne': 1.0,
'Na': 5.0,
'Mg': 5.0,
'Al': 5.0,
'Si': 5.0,
'P': 5.0,
'S': 5.0,
'Cl': 5.0,
'Ar': 1.0,
'K': 1.0,
'Ca': 1.0,
'Sc': 1.0,
'Ti': 1.0,
'V': 1.0,
'Cr': 0.0,
'Mn': 0.0,
'Fe': 0.0,
'Co': 0.0,
'Ni': 0.0,
'Cu': 1.0,
'Zn': 2.0,
'Ga': 2.0,
'Ge': 2.0,
'As': 2.0,
'Se': 2.0,
'Br': 2.0,
'Kr': 1.0,
'Rb': 1.0,
'Sr': 1.0,
'Y': 1.0,
'Zr': 1.0,
'Nb': 1.0,
'Mo': 1.0,
'Tc': 1.0,
'Ru': 1.0,
'Rh': 1.0,
'Pd': 1.0,
'Ag': 1.0,
'Cd': 1.0,
'In': 1.0,
'Sn': 1.0,
'Sb': 1.0,
'Te': 1.0,
'I': 1.0,
'Xe': 1.0,
'Cs': 0.0,
'Ba': 0.0,
'La': 0.0,
'Ce': 0.0,
'Pr': 0.0,
'Nd': 0.0,
'Pm': 0.0,
'Sm': 0.0,
'Eu': 0.0,
'Gd': 0.0,
'Tb': 0.0,
'Dy': 0.0,
'Ho': 0.0,
'Er': 0.0,
'Tm': 0.0,
'Yb': 0.0,
'Lu': 0.0,
'Hf': 1.0,
'Ta': 1.0,
'W': 1.0,
'Re': 1.0,
'Os': 1.0,
'Ir': 1.0,
'Pt': 1.0,
'Au': 1.0,
'Hg': 1.0,
'Tl': 1.0,
'Pb': 1.0,
'Bi': 1.0,
'Po': 1.0,
'At': 1.0,
'Rn': 1.0,
'Fr': 0.0,
'Ra': 0.0,
'Ac': 0.0,
'Th': 0.0,
'Pa': 0.0,
'U': 0.0,
'Np': 0.0,
'Pu': 0.0,
'Am': 0.0,
'Cm': 0.0,
'Bk': 0.0,
'Cf': 0.0,
'Es': 0.0,
'Fm': 0.0,
'Md': 0.0,
'No': 0.0,
'Lr': 0.0,
'Rf': 0.0,
'Db': 0.0,
'Sg': 0.0,
'Bh': 0.0,
'Hs': 0.0,
'Mt': 0.0,
'Ds': 0.0,
'Rg': 0.0,
'Cn': 0.0,
'Nh': 0.0,
'Fl': 0.0,
'Mc': 0.0,
'Lv': 0.0,
'Ts': 0.0,
'Og': 0.0
}
"""
# unnormalized probability for having key element

# weights need to be discussed
# 1 H 
# 6 noble gas elements
# 35 sp block elements, first 14 most common
# about 30 sd block elements
# what to do with all the other elements ?

#GRACE STANDARD
"""
species_prob = {
'H': 10.0,
'He': 1.0,
'Li': 5.0,
'Be': 5.0,
'B': 5.0,
'C': 10.0,
'N': 5.0,
'O': 5.0,
'F': 5.0,
'Ne': 1.0,
'Na': 5.0,
'Mg': 5.0,
'Al': 5.0,
'Si': 5.0,
'P': 5.0,
'S': 5.0,
'Cl': 5.0,
'Ar': 1.0,
'K': 1.0,
'Ca': 1.0,
'Sc': 1.0,
'Ti': 1.0,
'V': 1.0,
'Cr': 1.0,
'Mn': 1.0,
'Fe': 1.0,
'Co': 1.0,
'Ni': 1.0,
'Cu': 1.0,
'Zn': 2.0,
'Ga': 2.0,
'Ge': 2.0,
'As': 2.0,
'Se': 2.0,
'Br': 2.0,
'Kr': 1.0,
'Rb': 1.0,
'Sr': 1.0,
'Y': 1.0,
'Zr': 1.0,
'Nb': 1.0,
'Mo': 1.0,
'Tc': 1.0,
'Ru': 1.0,
'Rh': 1.0,
'Pd': 1.0,
'Ag': 1.0,
'Cd': 1.0,
'In': 1.0,
'Sn': 1.0,
'Sb': 1.0,
'Te': 1.0,
'I': 1.0,
'Xe': 1.0,
'Cs': 0.5,
'Ba': 0.5,
'La': 0.5,
'Ce': 0.5,
'Pr': 0.5,
'Nd': 0.5,
'Pm': 0.5,
'Sm': 0.5,
'Eu': 0.5,
'Gd': 0.5,
'Tb': 0.5,
'Dy': 0.5,
'Ho': 0.5,
'Er': 0.5,
'Tm': 0.5,
'Yb': 0.5,
'Lu': 0.5,
'Hf': 1.0,
'Ta': 1.0,
'W': 1.0,
'Re': 1.0,
'Os': 1.0,
'Ir': 1.0,
'Pt': 1.0,
'Au': 1.0,
'Hg': 1.0,
'Tl': 1.0,
'Pb': 1.0,
'Bi': 1.0,
'Po': 1.0,
'At': 0.5,
'Rn': 0.5,
'Fr': 0.5,
'Ra': 0.5,
'Ac': 0.5,
'Th': 0.5,
'Pa': 0.5,
'U': 0.5,
'Np': 0.5,
'Pu': 0.5,
'Am': 0.5,
'Cm': 0.5,
'Bk': 0.0,
'Cf': 0.0,
'Es': 0.0,
'Fm': 0.0,
'Md': 0.0,
'No': 0.0,
'Lr': 0.0,
'Rf': 0.0,
'Db': 0.0,
'Sg': 0.0,
'Bh': 0.0,
'Hs': 0.0,
'Mt': 0.0,
'Ds': 0.0,
'Rg': 0.0,
'Cn': 0.0,
'Nh': 0.0,
'Fl': 0.0,
'Mc': 0.0,
'Lv': 0.0,
'Ts': 0.0,
'Og': 0.0
}

"""

#LANL URANIUM SET

w_actinides=1.0
w_lanthanides=1.0
w_fission=1.0
w_gas=1.0
w_other=1.0
w_uranium=1000.

species_prob = {
'H': w_gas,
'He': w_gas,
'Li': w_fission,
'Be': w_other,
'B': w_other,
'C': w_other,
'N': w_other,
'O': w_other,
'F': w_other,
'Ne': w_gas,
'Na': w_other,
'Mg': w_other,
'Al': w_other,
'Si': w_other,
'P': w_other,
'S': w_other,
'Cl': w_other,
'Ar': w_gas,
'K': w_fission,
'Ca': w_fission,
'Sc': w_other,
'Ti': w_other,
'V': w_other,
'Cr': w_other,
'Mn': w_other,
'Fe': w_other,
'Co': w_other,
'Ni': w_other,
'Cu': w_other,
'Zn': w_other,
'Ga': w_other,
'Ge': w_other,
'As': w_other,
'Se': w_fission,
'Br': w_fission,
'Kr': w_gas,
'Rb': w_fission,
'Sr': w_fission,
'Y': w_fission,
'Zr': w_fission,
'Nb': w_other,
'Mo': w_fission,
'Tc': w_fission,
'Ru': w_fission,
'Rh': w_fission,
'Pd': w_fission,
'Ag': w_fission,
'Cd': w_other,
'In': w_other,
'Sn': w_fission,
'Sb': w_fission,
'Te': w_fission,
'I': w_fission,
'Xe': w_gas,
'Cs': w_fission,
'Ba': w_fission,
'La': w_lanthanides,
'Ce': w_lanthanides,
'Pr': w_lanthanides,
'Nd': w_lanthanides,
'Pm': w_lanthanides,
'Sm': w_lanthanides,
'Eu': w_lanthanides,
'Gd': w_lanthanides,
'Tb': w_lanthanides,
'Dy': w_lanthanides,
'Ho': w_lanthanides,
'Er': w_lanthanides,
'Tm': w_lanthanides,
'Yb': w_lanthanides,
'Lu': w_lanthanides,
'Hf': w_other,
'Ta': w_other,
'W': w_other,
'Re': w_other,
'Os': w_other,
'Ir': w_other,
'Pt': w_other,
'Au': w_other,
'Hg': w_other,
'Tl': w_other,
'Pb': w_other,
'Bi': w_other,
'Po': w_other,
'At': w_other,
'Rn': w_other,
'Fr': w_other,
'Ra': w_gas,
'Ac': w_actinides,
'Th': w_actinides,
'Pa': w_actinides,
'U': w_uranium,
'Np': w_actinides,
'Pu': w_actinides,
'Am': w_actinides,
'Cm': w_actinides,
'Bk': 0.0,
'Cf': 0.0,
'Es': 0.0,
'Fm': 0.0,
'Md': 0.0,
'No': 0.0,
'Lr': 0.0,
'Rf': 0.0,
'Db': 0.0,
'Sg': 0.0,
'Bh': 0.0,
'Hs': 0.0,
'Mt': 0.0,
'Ds': 0.0,
'Rg': 0.0,
'Cn': 0.0,
'Nh': 0.0,
'Fl': 0.0,
'Mc': 0.0,
'Lv': 0.0,
'Ts': 0.0,
'Og': 0.0
}

"""
#LANL MISSION SET

w_actinides=3.0
w_lanthanides=3.0
w_fission=2.0
w_gas=2.0
w_other=1.0

species_prob = {
'H': w_gas,
'He': w_gas,
'Li': w_fission,
'Be': w_other,
'B': w_other,
'C': w_other,
'N': w_other,
'O': w_other,
'F': w_other,
'Ne': w_gas,
'Na': w_other,
'Mg': w_other,
'Al': w_other,
'Si': w_other,
'P': w_other,
'S': w_other,
'Cl': w_other,
'Ar': w_gas,
'K': w_fission,
'Ca': w_fission,
'Sc': w_other,
'Ti': w_other,
'V': w_other,
'Cr': w_other,
'Mn': w_other,
'Fe': w_other,
'Co': w_other,
'Ni': w_other,
'Cu': w_other,
'Zn': w_other,
'Ga': w_other,
'Ge': w_other,
'As': w_other,
'Se': w_fission,
'Br': w_fission,
'Kr': w_gas,
'Rb': w_fission,
'Sr': w_fission,
'Y': w_fission,
'Zr': w_fission,
'Nb': w_other,
'Mo': w_fission,
'Tc': w_fission,
'Ru': w_fission,
'Rh': w_fission,
'Pd': w_fission,
'Ag': w_fission,
'Cd': w_other,
'In': w_other,
'Sn': w_fission,
'Sb': w_fission,
'Te': w_fission,
'I': w_fission,
'Xe': w_gas,
'Cs': w_fission,
'Ba': w_fission,
'La': w_lanthanides,
'Ce': w_lanthanides,
'Pr': w_lanthanides,
'Nd': w_lanthanides,
'Pm': w_lanthanides,
'Sm': w_lanthanides,
'Eu': w_lanthanides,
'Gd': w_lanthanides,
'Tb': w_lanthanides,
'Dy': w_lanthanides,
'Ho': w_lanthanides,
'Er': w_lanthanides,
'Tm': w_lanthanides,
'Yb': w_lanthanides,
'Lu': w_lanthanides,
'Hf': w_other,
'Ta': w_other,
'W': w_other,
'Re': w_other,
'Os': w_other,
'Ir': w_other,
'Pt': w_other,
'Au': w_other,
'Hg': w_other,
'Tl': w_other,
'Pb': w_other,
'Bi': w_other,
'Po': w_other,
'At': w_other,
'Rn': w_other,
'Fr': w_other,
'Ra': w_gas,
'Ac': w_actinides,
'Th': w_actinides,
'Pa': w_actinides,
'U': w_actinides,
'Np': w_actinides,
'Pu': w_actinides,
'Am': w_actinides,
'Cm': w_actinides,
'Bk': 0.0,
'Cf': 0.0,
'Es': 0.0,
'Fm': 0.0,
'Md': 0.0,
'No': 0.0,
'Lr': 0.0,
'Rf': 0.0,
'Db': 0.0,
'Sg': 0.0,
'Bh': 0.0,
'Hs': 0.0,
'Mt': 0.0,
'Ds': 0.0,
'Rg': 0.0,
'Cn': 0.0,
'Nh': 0.0,
'Fl': 0.0,
'Mc': 0.0,
'Lv': 0.0,
'Ts': 0.0,
'Og': 0.0
}
"""

"""
#REFRACTORY SET
species_prob = {
'H': 1.0,
'He': 1.0,
'Li': 0.1,
'Be': 0.1,
'B': 0.1,
'C': 0.1,
'N': 0.1,
'O': 0.1,
'F': 0.1,
'Ne': 0.1,
'Na': 0.1,
'Mg': 0.1,
'Al': 0.1,
'Si': 0.1,
'P': 0.1,
'S': 0.1,
'Cl': 0.1,
'Ar': 0.1,
'K': 0.1,
'Ca': 0.1,
'Sc': 0.1,
'Ti': 1.0,
'V': 1.0,
'Cr': 1.0,
'Mn': 0.1,
'Fe': 0.1,
'Co': 0.1,
'Ni': 0.1,
'Cu': 0.1,
'Zn': 0.1,
'Ga': 0.1,
'Ge': 0.1,
'As': 0.1,
'Se': 0.1,
'Br': 0.1,
'Kr': 0.1,
'Rb': 0.1,
'Sr': 0.1,
'Y': 0.1,
'Zr': 1.0,
'Nb': 1.0,
'Mo': 1.0,
'Tc': 1.0,
'Ru': 1.0,
'Rh': 1.0,
'Pd': 0.1,
'Ag': 0.1,
'Cd': 0.1,
'In': 0.1,
'Sn': 0.1,
'Sb': 0.1,
'Te': 0.1,
'I': 0.1,
'Xe': 0.1,
'Cs': 0.1,
'Ba': 0.1,
'La': 0.1,
'Ce': 0.1,
'Pr': 0.1,
'Nd': 0.1,
'Pm': 0.1,
'Sm': 0.1,
'Eu': 0.1,
'Gd': 0.1,
'Tb': 0.1,
'Dy': 0.1,
'Ho': 0.1,
'Er': 0.1,
'Tm': 0.1,
'Yb': 0.1,
'Lu': 0.1,
'Hf': 1.0,
'Ta': 1.0,
'W': 1.0,
'Re': 1.0,
'Os': 1.0,
'Ir': 1.0,
'Pt': 0.1,
'Au': 0.1,
'Hg': 0.1,
'Tl': 0.1,
'Pb': 0.1,
'Bi': 0.1,
'Po': 0.1,
'At': 0.1,
'Rn': 0.1,
'Fr': 0.1,
'Ra': 0.1,
'Ac': 0.1,
'Th': 0.1,
'Pa': 0.1,
'U': 0.1,
'Np': 0.1,
'Pu': 0.1,
'Am': 0.1,
'Cm': 0.1,
'Bk': 0.0,
'Cf': 0.0,
'Es': 0.0,
'Fm': 0.0,
'Md': 0.0,
'No': 0.0,
'Lr': 0.0,
'Rf': 0.0,
'Db': 0.0,
'Sg': 0.0,
'Bh': 0.0,
'Hs': 0.0,
'Mt': 0.0,
'Ds': 0.0,
'Rg': 0.0,
'Cn': 0.0,
'Nh': 0.0,
'Fl': 0.0,
'Mc': 0.0,
'Lv': 0.0,
'Ts': 0.0,
'Og': 0.0
}
"""

# unnormalized probability for having key different species
num_species_dict = {
    1: 0.5,
    2: 1.0,
    3: 2.0,
    4: 2.0,
    5: 2.0,
    6: 2.0,
    7: 1.5,
    8: 1.0,
    9: 1.0,
   10: 0.5,
   11: 0.2,
   12: 0.1,
   13: 0.09,
   14: 0.08,
   15: 0.07,
   16: 0.06,
   17: 0.05,
   18: 0.04,
   19: 0.03,    
   20: 0.02,
   21: 0.01,
   22: 0.01,
   23: 0.01,
   24: 0.01,
   25: 0.01,
   26: 0.01,
   27: 0.01,
   28: 0.01,
   29: 0.01,
   30: 0.01,
   31: 0.01,
   32: 0.01,
   33: 0.01,
   34: 0.01,    
   35: 0.01,
   36: 0.01,
   37: 0.01,
   38: 0.01,
   39: 0.01,
   40: 0.01,
   41: 0.01,
   42: 0.01,
   43: 0.01,
   44: 0.01,
   45: 0.01,
   46: 0.01,
   47: 0.01,
   48: 0.01,
   49: 0.01,    
   50: 0.01,
   51: 0.01,
   52: 0.01,
   53: 0.01,
   54: 0.01,
   55: 0.01,
   56: 0.01,
   57: 0.01,
   58: 0.01,
   59: 0.01,
   60: 0.01,
   61: 0.01,
   62: 0.01,
   63: 0.01,
   64: 0.01,    
   65: 0.01,
   66: 0.01,
   67: 0.01,
   68: 0.01,
   69: 0.01,
   70: 0.01,
   71: 0.01,
   72: 0.01,
   73: 0.01,
   74: 0.01,
   75: 0.01,
   76: 0.01,
   77: 0.01,
   78: 0.01,
   79: 0.01,    
   80: 0.01
}    









# unnormalized probability for having key number of atoms in cell
num_atoms_dict = {
    1: 1.0,
    2: 2.0,
    3: 3.0,
    4: 4.0,
    5: 5.0,
    6: 6.0,
    7: 7.0,
    8: 8.0,
    9: 9.0,
   10: 10.0,
   11: 10.0,
   12: 10.0,
   13: 10.0,
   14: 10.0,
   15: 10.0,
   16: 10.0,
   17: 10.0,
   18: 10.0,
   19: 10.0,    
   20: 10.0,
   21: 10.0,
   22: 10.0,
   23: 10.0,
   24: 10.0,
   25: 10.0,
   26: 9.0,
   27: 8.0,
   28: 7.0,
   29: 6.0,
   30: 5.0,
   31: 4.0,
   32: 3.0,
   33: 2.0,
   34: 1.0,    
   35: 1.0,
   36: 1.0,
   37: 1.0,
   38: 1.0,
   39: 1.0,
   40: 1.0,
   41: 0.5,
   42: 0.5,
   43: 0.5,
   44: 0.5,
   45: 0.5,
   46: 0.5,
   47: 0.5,
   48: 0.5,
   49: 0.5,    
   50: 0.5,
   51: 0.2,
   52: 0.2,
   53: 0.2,
   54: 0.2,
   55: 0.2,
   56: 0.2,
   57: 0.2,
   58: 0.2,
   59: 0.2,
   60: 0.1,
   61: 0.1,
   62: 0.1,
   63: 0.1,
   64: 0.1,    
   65: 0.1,
   66: 0.1,
   67: 0.1,
   68: 0.1,
   69: 0.1,
   70: 0.1,
   71: 0.1,
   72: 0.1,
   73: 0.1,
   74: 0.1,
   75: 0.1,
   76: 0.1,
   77: 0.1,
   78: 0.1,
   79: 0.1,    
   80: 0.1,
   81: 0.1,
   82: 0.1,
   83: 0.1,
   84: 0.1,    
   85: 0.1,
   86: 0.1,
   87: 0.1,
   88: 0.1,
   89: 0.1,
   90: 0.1,
   91: 0.1,
   92: 0.1,
   93: 0.1,
   94: 0.1,
   95: 0.1,
   96: 0.1,
   97: 0.1,
   98: 0.1,
   99: 0.1,    
  100: 0.1
}  

# %%
#n_species=12

#use energy mode or force mode
energy_mode=True
#volume_scaling=1 corresponds to a total volume that is the sum of the per-atom exclusion volumes.
volume_scaling_min=1.0
volume_scaling_max=2.5
volume_scaling_max=3.5

output_prefix="./configs/" 
#n_atoms_min=2
#n_atoms_max=24

n_atoms=12
#n_species=5

#relative width of the radius distribution
width=0.25
width=0.3
#parameters of a beta distribution the radius is sampled from
a_beta=1.25
b_beta=1.25

species_prob = {
'Li': 1.0,
'Cl': 1.0,
}

num_species_dict = {
    2: 1.0,
}

#target_species=["Cs","Te"]
#species_probabilities=[1,1]



"""
output_prefix="./configs-U3Cl12/" 
n_atoms=15
target_species=["U","Cl","Cl","Cl","Cl"]
fixed_stoichiometry=True
"""




# %%

target_species=list(species_prob.keys())
species_probabilities=list(species_prob.values())
target_number_of_species=list(num_species_dict.keys())
number_of_species_probabilities=list(num_species_dict.values())
fixed_stoichiometry=False


# %%
#this code is now hard-wired so that the number of pseudo-species is the total number of atoms
#print(species_probabilities)
species_probabilities=np.array(species_probabilities)
species_probabilities/=np.sum(species_probabilities)

target_number_of_species=target_number_of_species[:n_atoms]
number_of_species_probabilities=np.array(number_of_species_probabilities[:n_atoms])
number_of_species_probabilities/=np.sum(number_of_species_probabilities)


sampler = entropy_utils.MendeleevUniformRadiusSampler(target_species,width,a_beta,b_beta,fixed_stoichiometry)

#sample_radii = lambda x : sampler(x, n_species, probabilities=species_probabilities)

sample_radii = lambda x : sampler(x, np.random.choice(target_number_of_species,p=number_of_species_probabilities), probabilities=species_probabilities)
volume_scaling=[volume_scaling_min,volume_scaling_max]

#if n_species>n_atoms:
#    n_species=n_atoms



# %%
target_number_of_species,number_of_species_probabilities

# %%
#n_atoms=list(range(n_atoms_min,n_atoms_max))


# %%
#rr,rr_by_symbol=entropy_utils.sample_radii(n_species)
#atom_types={ k:v['species_id'] for k,v in rr_by_symbol.items() }


#n_elems=n_species
n_descriptors_tot=55
n_keep=n_descriptors_tot
base_mask=list(range(n_descriptors_tot))
n_descriptors_sample=n_descriptors_tot


#number of atoms in the cell

epsilon=1e-6

#initial strenght of the entropy term
K_init=1.

#allowed cell shapes
shapes=[ [2, 1, 1], [1, 1, 1], [2, 2, 1] ]


#enforce entropy decrease before accepting configuration
strict_entropy_decrease=False


min_accept_per_cycle=1

trials_per_cycle=10000

#have to fix this
#volume_scaling=[1,1.5]
#volume_scaling=[ x/0.74 for x in volume_scaling ]

# %%

#setup a reference for the scale matrix by generating random configs with no atoms approching by less than the min_distance

rr,rr_by_symbol=sample_radii(n_atoms)
atom_types={ k:v['species_id'] for k,v in rr_by_symbol.items() }
print(atom_types, flush=True)

manager_random=entropy_utils.CNManager(n_descriptors_tot)
model=entropy_utils.CNModel(n_atoms, n_descriptors_tot, energy_mode=energy_mode, populations=None, mask=None, cross_=None, renorm_=None, mean_=None, count_=0, epsilon_=epsilon)


from ase.optimize.bfgslinesearch import BFGSLineSearch
from  ase.calculators.lammpslib import LAMMPSlib


for i in range(100):
    #choose a number of atoms at random
    #n_at=random.choice(n_atoms)
    n_at=n_atoms
    if i%1==0:
        #print(i,end=" ")
        print(i, flush=True)



    rr,rr_by_symbol=sample_radii(n_at)

    #atom_types={ k:v['species_id'] for k,v in rr_by_symbol.items() }

    print(rr, flush=True)

    generate_min,generate_zero,snap_descriptors=entropy_utils.generate_lammps_scripts(rr,"./test")
    calculator_relax=LAMMPSlib(lmpcmds=generate_zero.split("\n"), log_file=None,keep_alive=True)

    shape=random.choice(shapes)
    
    species_list=[rr[k]['symbol'] for k in rr.keys() ]
    sampled_species=species_list
    #sampled_species=np.random.choice(species_list, n_at, replace=True)
    #sampled_species.sort()
    #print(n_at,"sampled_species",sampled_species)
    calculator_min = entropy_utils.EntropyCalculator(lmpcmds=generate_min.split("\n"), log_file="lammps.log", model=model, keep_alive=True, atom_types=atom_types)
    
    target_volume=0.
    for s in sampled_species:
        target_volume+=rr_by_symbol[s]['volume']/len(sampled_species)
    target_volume=np.random.uniform(low=volume_scaling[0],high=volume_scaling[1])*target_volume

    #print(target_volume, end=" ")
    
    mask=np.random.choice(base_mask,size=(n_keep,),replace=False).tolist()
    mask.sort()
    indices=list(range(n_at))
    populations={}
    populations[1]=indices
    
    calculator_relax=LAMMPSlib(lmpcmds=generate_zero.split("\n"), log_file="relax.log",keep_alive=True,atom_types=atom_types)

    ntry=0
    while ntry<10:
        try:
            atoms = entropy_utils.generate_random_cell(rr, sampled_species, target_volume, shape=shape)
            print(atoms, flush=True)
            ntry+=1
            break
        except Exception:
            ntry+=1
            traceback.print_exc()
    print(ntry, flush=True)
    
    if not ntry==10:
        atoms.calc=calculator_relax 
        atoms.get_potential_energy()
        #relax with the core repulsion alone for reference
        opt = BFGSLineSearch(atoms, force_consistent=True,logfile="min.log")
        opt.run(fmax=0.05, steps=30)
            
        atoms.calc=calculator_min
        d=entropy_utils.compute_descriptors(atoms)
        #print(d.shape)
        #print(np.max(np.fabs(d),axis=0))
        #print(d[:,0])
        #print(d)
        manager_random.update(d)

        dists=atoms.get_all_distances(mic=True)
        dists+=1000*np.identity(n_at)
        species_index_map={ v['symbol']:k for k,v in rr.items() }
        for a1 in range(len(atoms)):
            for a2 in range(len(atoms)):
                dists[a1,a2]/=(rr[species_index_map[sampled_species[a1]]]["r_min"]+rr[species_index_map[sampled_species[a2]]]["r_min"])
                
        print("Relaxed min dist: ",np.min(dists), flush=True)





import pickle
   
pickle.dump( manager_random.data, open( "random-ref-data.p", "wb" ) )
pickle.dump( manager_random, open( "random-manager.p", "wb" ) )
        
print(manager_random.cross,manager_random.sum, flush=True)
manager_random.evaluate()


# %%




#datar=pd.DataFrame(np.vstack(manager_random.data))
#print(datar.shape)

#plt.scatter(datar[0],datar[1])
#plt.figure()
#plt.scatter(datar[0],datar[2])



# %%
#set the fixed mean and renormalization matrices
mean=manager_random.sum/manager_random.count
covariance=manager_random.cross/manager_random.count-np.outer(mean,mean)
var=np.sqrt(np.diagonal(covariance))
renorm=np.outer(var,var)
print(mean, flush=True)
#print(np.divide(covariance,renorm))




# %%
#do this to act on the raw information matrix
#mean=np.zeros(mean.shape)
#renorm=np.ones(renorm.shape)

# %%


#plt.scatter((datar[0]-mean[0])/np.sqrt(renorm[0,0]),(datar[1]-mean[1])/np.sqrt(renorm[1,1]))
#plt.figure()
#plt.scatter((datar[1]-mean[1])/np.sqrt(renorm[1,1]),(datar[2]-mean[2])/np.sqrt(renorm[2,2]))


#plt.scatter(datar[0],datar[2])

# %%

"""
plt.semilogy(manager_random.s)



u,s,vh=np.linalg.svd(np.divide(covariance,renorm))

plt.semilogy(s)

plt.figure()
plt.imshow(u)
"""


# %%
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
    
import ase.io.lammpsdata
import os
import glob
from ase.io import read,write



#aggregates the mean and covariances from multiple configurations
manager=entropy_utils.CNManager(n_descriptors_tot,energy_mode=energy_mode,mean=mean,renorm=renorm,epsilon=epsilon)

raw_data=[]
    
current_det=0

n_reject_improve=0
n_reject_dist=0
i_accept=0
n_accept=0

K=K_init

if not os.path.exists(output_prefix):
    os.makedirs(output_prefix) 


if energy_mode:
    files = glob.glob(output_prefix+'dopti-energy.*')
else:
    files = glob.glob(output_prefix+'dopti-forces.*')
for f in files:
    os.remove(f)


i=0

bond_lengths=[]
counter_dict={}

for iii in range(1):
    i_accept_batch=0
    for ii in range(trials_per_cycle):

        mask=None
        #mask=random.choices(base_mask,k=n_descriptors_sample)
        #mask.sort()
        #print("MASK: ",mask)
        rr,rr_by_symbol=sample_radii(n_atoms)
        atom_types={ k:v['species_id'] for k,v in rr_by_symbol.items() }
        print(rr, flush=True)

        specorder=[v['symbol'] for k,v in rr.items()]
        generate_min,generate_zero,snap_descriptors=entropy_utils.generate_lammps_scripts(rr,"./test")

        i+=1
        shape=random.choice(shapes)
        #n_at=random.choice(n_atoms)
        n_at=n_atoms
        
        species_list=[rr[k]['symbol'] for k in rr.keys() ]
        #sampled_species=np.random.choice(species_list, n_at, replace=True)
        sampled_species=species_list
        sampled_species.sort()
        
        #print(sampled_species)
        
        #print(generate_min,n_Be)
        calculator_min = entropy_utils.EntropyCalculator(lmpcmds=generate_min.split("\n"), log_file="lammps.log", model=model, keep_alive=True, atom_types=atom_types)



        target_volume=0.
        for s in sampled_species:
            target_volume+=rr_by_symbol[s]['volume']/len(sampled_species)
        target_volume=np.random.uniform(low=volume_scaling[0],high=volume_scaling[1])*target_volume


        indices=list(range(n_at))
        populations={}
        populations[1]=indices
        model=entropy_utils.CNModel(n_atoms, n_descriptors_tot, energy_mode=energy_mode, populations=None, mask=mask, cross_=manager.cross, renorm_=renorm, mean_=mean, count_=manager.count, epsilon_=epsilon)

    
        if i_accept<10:
            model.active=False
            model.K=0.0
        else: 
            model.active=True
            model.K=K

            
        from  ase.calculators.lammpslib import LAMMPSlib
        calculator_relax=LAMMPSlib(lmpcmds=generate_zero.split("\n"), log_file=None,keep_alive=True,atom_types=atom_types)
        calculator_min = entropy_utils.EntropyCalculator(lmpcmds=generate_min.split("\n"), log_file=None, model=model, keep_alive=True,atom_types=atom_types)
        
        ntry=0
        while ntry<10:
            try:
                atoms = entropy_utils.generate_random_cell(rr, sampled_species, target_volume=target_volume, shape=shape)
                ntry+=1
                break
            except Exception:
                ntry+=1
                traceback.print_exc()


        

        if not ntry==10:

            dists+=1000*np.identity(n_at)
            species_index_map={ v['symbol']:k for k,v in rr.items() }
            for a1 in range(len(atoms)):
                for a2 in range(len(atoms)):
                    dists[a1,a2]/=(rr[species_index_map[sampled_species[a1]]]["r_min"]+rr[species_index_map[sampled_species[a2]]]["r_min"])
                
            print("As prepared min dist: ",np.min(dists), flush=True)
            


            #print(atom_numbers)
            #print(atoms)

            atoms.calc=calculator_relax
            #relax with the core repulsion alone
            opt = BFGSLineSearch(atoms, force_consistent=True,logfile=None)
            opt.run(fmax=0.05, steps=30)

            
            import pickle
            if i%10==0:
                if energy_mode:
                    pickle.dump( manager.data, open( output_prefix+"d-opti-energy.p", "wb" ) )
                else:
                    pickle.dump( manager.data, open( output_prefix+"d-opti-forces.p", "wb" ) )
                    
            atoms.calc=calculator_min
            opt = BFGSLineSearch(atoms, force_consistent=True,logfile=None)
            opt.run(fmax=0.05, steps=100)
                
                
            d=entropy_utils.compute_descriptors(atoms)
            cand_cond,cand_det=manager.evaluate(d)
                
            if i>1:
                print("CANDIDATE: ",cand_det, " CURRENT: ",current_det, flush=True)
            
            dists=atoms.get_all_distances(mic=True)
            dists+=1000*np.identity(n_at)


            species_index_map={ v['symbol']:k for k,v in rr.items() }
            for a1 in range(len(atoms)):
                for a2 in range(len(atoms)):
                    dists[a1,a2]/=(rr[species_index_map[sampled_species[a1]]]["r_min"]+rr[species_index_map[sampled_species[a2]]]["r_min"])
            
            print(i,cand_det-current_det, "min dist: ",np.min(dists),end=" ", flush=True)
            

            
            if np.min(dists) < 1:
                n_reject_dist+=1
            
            if cand_det>current_det:
                n_reject_improve+=1

            
                

            print("*****", i_accept, i_accept_batch, min_accept_per_cycle, np.min(dists), flush=True)
            if (i_accept<=10 or i_accept_batch<min_accept_per_cycle) and np.min(dists) > 1:
                print("ACCEPT BY DEFAULT", flush=True)
                manager.update(d)
                current_cond,current_det=manager.evaluate()

                mapping={}
                original_species=atoms.get_chemical_symbols()
                for k,v in rr.items():
                    mapping[v['symbol']]=v['original_symbol']

                remapped_species=[mapping[k] for k in original_species]
                #print(remapped_species)
                atoms.set_chemical_symbols(remapped_species)
                atoms=ase.build.sort(atoms)
                print(atoms, flush=True)
                #print(atoms.symbols)
                s=atoms.symbols.get_chemical_formula()
                if s in counter_dict:
                    counter_dict[s]+=1
                else:
                    counter_dict[s]=0
                if energy_mode:
                    s2=".e"
                else:
                    s2=".f"

                #output_file=str(Path(c).parents[0])+"/"+Path(c).stem + ".vasp"
                output_file=str(output_prefix+"/"+s+s2+(".%i"%counter_dict[s]))+".vasp"
                print(output_file, flush=True)
                write(output_file,atoms,format='vasp')
                
                """
                if energy_mode:
                    ase.io.lammpsdata.write_lammps_data(output_prefix+"dopti-energy.%i.dat" % i_accept ,atoms,specorder=specorder)
                    pickle.dump(rr,open(output_prefix+"dopti-energy.%i.settings" % i_accept,"wb" ) )

                else:
                    ase.io.lammpsdata.write_lammps_data(output_prefix+"dopti-forces.%i.dat" % i_accept ,atoms,specorder=specorder)
                    pickle.dump(rr,open(output_prefix+"dopti-forces.%i.settings" % i_accept,"wb" ))
                """

                i_accept+=1
                i_accept_batch+=1
                dists=atoms.get_all_distances(mic=True)
                bond_lengths+=dists.flatten().tolist()


            else:
                #if cand_det < current_det and np.min(dists) > min_distance:
                if np.min(dists) > 1 and ( (strict_entropy_decrease and cand_det < current_det ) or not strict_entropy_decrease):
                    print("ACCEPT BY CRITERION", flush=True)

                    #if np.min(dists) > min_distance:
                    n_reject_dist=0
                    n_reject_improve=0
                    manager.update(d)
                    current_cond,current_det=manager.evaluate()
                    #print("***CANDIDATE: ",cand_cond,cand_det, " CURRENT: ", current_cond,current_det)

                    mapping={}
                    original_species=atoms.get_chemical_symbols()
                    for k,v in rr.items():
                        mapping[v['symbol']]=v['original_symbol']

                    remapped_species=[mapping[k] for k in original_species]
                    #print(remapped_species)
                    atoms.set_chemical_symbols(remapped_species)
                    atoms=ase.build.sort(atoms)
                    print(atoms, flush=True)
                    #print(atoms.symbols)
                    s=atoms.symbols.get_chemical_formula()
                    if s in counter_dict:
                        counter_dict[s]+=1
                    else:
                        counter_dict[s]=0

                    #output_file=str(Path(c).parents[0])+"/"+Path(c).stem + ".vasp"
                    if energy_mode:
                        s2=".e"
                    else:
                        s2=".f"

                    output_file=str(output_prefix+"/"+s+s2+(".%i"%counter_dict[s]))+".vasp"
                    print(output_file, flush=True)
                    write(output_file,atoms,format='vasp')
                    #count+=1

                    """
                    if energy_mode:
                        ase.io.lammpsdata.write_lammps_data(output_prefix+"dopti-energy.%i.dat" % i_accept ,atoms,specorder=specorder)
                        pickle.dump(rr,open(output_prefix+"dopti-energy.%i.settings" % i_accept,"wb" ) )
                    else:
                        ase.io.lammpsdata.write_lammps_data(output_prefix+"dopti-forces.%i.dat" % i_accept ,atoms,specorder=specorder)
                        pickle.dump(rr,open(output_prefix+"dopti-forces.%i.settings" % i_accept,"wb" ))
                    """

                    i_accept+=1
                    n_accept+=1
                    i_accept_batch+=1
                
                    dists=atoms.get_all_distances(mic=True)
                    bond_lengths+=dists.flatten().tolist()
                else:
                    print("REJECTED", flush=True)
                    print()
                
            if i%10==0:
                pass
                #manager.print_status()
                
            if n_reject_improve>10:
                K*=1.2
                n_reject_improve=0
                n_reject_dist=0
            if n_reject_dist>10:
                K*=0.8
                n_reject_improve=0
                n_reject_dist=0
            if n_accept>10:
                K*=1.1
                n_accept=0
                
                
            print("K=",K, n_reject_improve,n_reject_dist,model.count, flush=True)
            if i_accept>500:
                break
            
            
            
            
            
            if i%25==0 and not manager.s is None:
                #plt.figure()
                #plt.semilogy(manager.s)
                #plt.show()

                plt.figure()
                plt.hist(bond_lengths,bins=100)
                plt.show()
            
        
    

# %%
#dists.flatten()

# %%


# %%
#import pickle
#pickle.dump( manager.data, open( "d-opti.p", "wb" ) )

# %%


# %%



