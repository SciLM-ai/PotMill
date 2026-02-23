import ase
import ase.build
import ase.calculators.lammpslib
import lammps
import lammps.mliap
import scipy
import scipy.linalg


import numpy as np

from mpi4py import MPI
import pandas as pd


import jax
import jax.numpy as jaxnp
from jax import grad, jit, vmap
from jax import random
from functools import partial

from mendeleev.fetch import fetch_table




import itertools
import numpy as np
from mendeleev.fetch import fetch_table
from scipy import stats
from scipy.stats.sampling import NumericalInverseHermite



def random_combination_with_replacement(iterable, r):
    import random
    "Random selection from itertools.combinations_with_replacement(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n+r-1), k=r))
    return tuple(pool[i-j] for j, i in enumerate(indices))

#the chemical space. 
# If fixed_stoichiometry==False, sampled every combination with replacement with the same probability. 
# If fixed_stoichiometry==True, cycles through the species list.
class MendeleevUniformRadiusSampler():
    def __init__(self,species,width,a,b,fixed_stoichiometry=False):
        from mendeleev.fetch import fetch_table
        self.ptable = fetch_table("elements")
        self.radii_dict=dict(zip(self.ptable['symbol'], self.ptable['covalent_radius_pyykko']))  
        #self.metallic_radii_dict=dict(zip(self.ptable['symbol'], self.ptable['metallic_radius_c12']))
        self.species=species
        self.width=width
        self.a=a
        self.b=b
        self.beta=stats.beta(a=a,b=b)
        self.beta_dist = NumericalInverseHermite(self.beta)
        self.fixed_stoichiometry=fixed_stoichiometry

    def __call__(self,n_atoms, n_species=None, probabilities=None):
        if self.fixed_stoichiometry:
            comb=tuple(list(itertools.islice(itertools.cycle(self.species), n_atoms)))
        else:
            #generate as many species as pseudo-species
            if n_species is None:
                n_species=n_atoms
            
            if not probabilities is None:
                #original version
                #species=np.random.choice(self.species,n_species,p=probabilities)
                species=np.random.choice(self.species,n_species,p=probabilities, replace=False)
            else:
                species=np.random.choice(self.species,n_species)
            comb=random_combination_with_replacement(species,n_atoms)


        print("MendeleevUniformRadiusSampler ",comb)
        radii={}
        radii_by_symbol={}
        for i,c in enumerate(comb):
            r=self.beta_dist.rvs()
            #print(r)
            r=((r-0.5)*self.width*2. + 1)*self.radii_dict[c]/100.
            pseudo_symbol=self.ptable['symbol'][i%len(self.ptable['symbol'])]
            radii[i+1]={ "species_id":i+1, "original_symbol":c,"symbol":pseudo_symbol,"r_atom":r, "r_min": 0.8*r ,"r_core":r,"r_cut":3*r, "volume": (4.*np.pi/3.)*(r)**3  }
            radii_by_symbol[pseudo_symbol]=radii[i+1]
        return radii,radii_by_symbol




class MendeleevCovalentRadiusSampler():

    def __init__(self):
        ptable = fetch_table("elements")
        cols = ["covalent_radius_pyykko"]
        rad=ptable[cols].to_numpy()
        rad=rad[:,0]/100.
        s=np.sort(rad)
        N=s.shape[0]
        y = np.arange(N) / float(N-1) 
        self.interpolator=scipy.interpolate.PchipInterpolator(y,s)

    def __call__(self):
        return self.interpolator(np.random.uniform())
    

class NormalRadiusSampler():
    def __init__(self,parameters):

        rad=[]
        for p in parameters:
            for i in range(int(10000*p[4])):
                rr=np.random.normal(loc=p[0],scale=p[1])
                if rr>=p[2] and rr<=p[3]:
                    rad.append(rr)

        s=np.sort(rad)
        N=s.shape[0]
        y = np.arange(N) / float(N-1) 
        self.interpolator=scipy.interpolate.PchipInterpolator(y,s)

    def __call__(self):
        return self.interpolator(np.random.uniform())

def compute_descriptors(atoms):
    atoms.get_potential_energy()
    return atoms.calc.entropy_model.last_bispectrum



class CNModel:

    def __init__(self, n_elements, n_descriptors_tot, energy_mode=True, populations=None, mask=None, cross_=None, renorm_=None, mean_=None, count_=0, epsilon_=1e-6):
        #required by MLIAPPY
        self.n_params=1
        self.n_elements=n_elements
        self.epsilon=epsilon_
        self.n_descriptors = n_descriptors_tot
        
        self.n_elements = n_elements
        self.energy_mode=energy_mode
        
        self.active=True
        self.count=count_
        if mask is None:
            mask = list(range(n_descriptors_tot))
        self.mask = mask
        self.n_descriptors_keep = len(self.mask)
        
        if self.count==0:
            self.active=False
            self.renorm=None
            self.mean=None
            self.cross=None
            return
        else:
            self.renorm=renorm_[mask,:][:,mask]
            self.mean=mean_[mask]
            self.cross=cross_[mask,:][:,mask]
            
        if self.renorm is None:
            self.renorm=np.identity(self.n_descriptors_keep)
        if self.mean is None:
            self.mean=np.zeros(self.n_descriptors_keep)
            
        self.populations=populations
        self.reg=self.epsilon*np.identity(self.n_descriptors_keep)
        
        self.cn_grad=grad(self.cn)
        
        self.K = 1

        #print(self.renorm.shape)

        
    @partial(jit, static_argnums=(0,))
    def cn(self,descriptors):
        #d=descriptors-self.mean
        d=descriptors-self.mean

        if self.energy_mode:
            d=jaxnp.mean(descriptors,axis=0)
            d=d.reshape((1,-1))
            
        if self.active:
            effective_count=self.count+d.shape[0]
            information=(self.cross + d.T@d)/effective_count 
            #print("**",information.shape,self.renorm.shape,self.reg.shape)
            projected_information=jaxnp.divide(information,self.renorm) + self.reg
            (sign, logabsdet) = jaxnp.linalg.slogdet(projected_information)
            #return -jaxnp.log( jaxnp.linalg.det(projected_information) )
            return -logabsdet
        else:
            return 0
    
        
    
    def __call__(self, elems, bispectrum, beta, energy):
        self.last_bispectrum=bispectrum.copy()
        #print(bispectrum.shape)
        b = bispectrum[:, self.mask]
        
        if self.active:
            energy[:] = 0
            energy[0] = self.K*self.cn(b)         
            b = self.K*self.cn_grad(b)
            beta[:, :] = 0
            beta[:, self.mask] = b
            
            if not jaxnp.all(jaxnp.isfinite(b)):
                print("GRAD ERROR!")
                #print(b)

          
        else:
            energy[:] = 0
            beta[:, :] = 0        
 
            
        #cleanup the jax cache. Seems to be required, otherwise can grow without bound and crash the code

        
        if self.cn._cache_size() > 30:
            self.cn._clear_cache()
            import gc
            import sys
            #jax.clear_caches()

            """
            for module_name, module in sys.modules.items():
                if module_name.startswith("jax"):
                    if module_name not in ["jax.interpreters.partial_eval"]:
                        for obj_name in dir(module):
                          obj = getattr(module, obj_name)
                          if hasattr(obj, "cache_clear"):
                            try:
                              obj.cache_clear()
                            except:
                              pass
            """
            gc.collect()


class CNManager:
    def __init__(self, n_descriptors, epsilon=0, mean=None, renorm=None, energy_mode=True):
        self.epsilon=epsilon
        self.count=0
        self.n_descriptors = n_descriptors
        self.sum=np.zeros((self.n_descriptors,))
        self.cross=np.zeros((self.n_descriptors, self.n_descriptors))
        self.reg=epsilon*np.identity(self.n_descriptors)
        self.data=[]
        self.s=None
        self.mean=mean
        self.renorm=renorm
        self.energy_mode=energy_mode
        
        if self.mean is None:
            self.mean=np.zeros((self.n_descriptors,))
            
        if self.renorm is None:
            self.renorm=np.ones((self.n_descriptors, self.n_descriptors))
            
            
    def print_status(self):
        self.evaluate()
        print("STATUS  -- COUNT ",self.count, " COND: ", np.linalg.cond(self.projected_information), "DET: ", -np.log( np.linalg.det(self.projected_information) + self.epsilon) , flush=True)


    def update(self, dd, key=None):    
        self.data.append(dd)
        dt = dd - self.mean
        
        if self.energy_mode:
            dt = np.mean(dt,axis=0)
            dt = dt.reshape((1,-1))
        
        self.sum += np.sum(dt,axis=0)
        self.cross += dt.T@dt
        self.count += dt.shape[0]
               
        #this is the covariance matrix of the descriptors
        information = self.cross/self.count 
        projected_information = np.divide(information,self.renorm)
        projected_information += self.reg
        
        try:
            u,s,vh = np.linalg.svd(projected_information)
            self.s = s
            #print(s)
            #print("CURRENT DB: ",self.count, keep, " PROJECTED: ", np.linalg.cond(projected_information), -np.log( np.linalg.det(projected_information)), "RAW: ", np.linalg.cond(information), -np.log( np.linalg.det(information)) , flush=True)
            #print(len(self.data))
        except:
            pass
    
        
    def evaluate(self,dd=None,key=None):
        effective_count=self.count
        if not dd is None:
            dt=dd-self.mean
            if self.energy_mode:
                dt=np.mean(dt,axis=0)
                dt=dt.reshape((1,-1))
            
            cross=self.cross.copy()
            cross+=dt.T@dt
            effective_count+=dt.shape[0]
            information=(cross)/effective_count 
        else:
            information= self.cross/effective_count 
            
        projected_information=np.divide(information,self.renorm) + self.reg
        self.projected_information=projected_information
        
        (sign, logabsdet) = np.linalg.slogdet(projected_information)

        return np.linalg.cond(projected_information), -logabsdet

        #return np.linalg.cond(information), -np.log( np.linalg.det(information))
        


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
def generate_random_cell(radii, species, target_volume, shape=[1,1,1]):
    from ase_ga.utilities import closest_distances_generator
    from ase_ga.utilities import get_all_atom_types
    from ase_ga.startgenerator import StartGenerator
    from ase.data import atomic_numbers
    import ase

    species_index_map={ v['symbol']:k for k,v in radii.items() }
    #print("generate_random_cell", species,species_index_map)

    n_atoms=len(species)
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
    unique_atom_types = list(set([ x for x in species ]))
    #blmin = closest_distances_generator(atom_numbers=unique_atom_types,ratio_of_covalent_radii=1)
    ratio=0.75
    blmin={}


    for i in species:
        ii=species_index_map[i]
        blmin[(ii, ii)] = radii[species_index_map[i]]['r_core'] * ratio
        for j in species:
            jj=species_index_map[j]
            if ii == jj:
                continue
            if (ii, jj) in blmin:
                continue
            blmin[(ii, jj)] = blmin[(jj, ii)] = (radii[species_index_map[i]]['r_core'] + radii[species_index_map[j]]['r_core'] )* ratio
    #print(blmin)
    #print("gen start ", blmin)
    sg = StartGenerator(slab, species, blmin,test_too_far=False)
    atoms = sg.get_new_candidate(maxiter=100)

    #print(atoms)
    #print("gen end")
    #use pbc by default
    atoms.set_pbc([True, True, True])
    current_volume = atoms.get_volume()/n_atoms
    #fix the volume
    atoms.set_cell(atoms.get_cell()*(target_volume/current_volume)**0.33333333,scale_atoms=True)
    current_volume=atoms.get_volume()/n_atoms
    #print(target_volume/current_volume)
    return atoms


#range of target first neighbor distances to scan
import itertools
import string
from ase.data import chemical_symbols

#WBe setup
"""
def sample_radii(n_species):
    #r_atom=[0.37,2.0]
    radii={}
    radii_by_symbol={}
    fraction=np.random.uniform()
    print(fraction)
    sampler=NormalRadiusSampler([[0.96,0.96*0.25,0.96*0.75,0.96*1.25,fraction],[1.36,1.36*0.25,1.36*0.75,1.25*1.36,1-fraction]])
    #sampler=MendeleevCovalentRadiusSampler()
    for i in range(1,n_species+1):
        #r=np.random.uniform(low=r_atom[0],high=r_atom[1])
        r=sampler()
        radii[i]={"species_id":i, "r_min": 0.8*r ,"r_core":r,"r_cut":3*r,"r_atom":r,"symbol":chemical_symbols[i],"volume": (4.*np.pi/3.)*(r)**3}
        radii_by_symbol[chemical_symbols[i]]={"species_id":i, "r_min": 0.8*r ,"r_core":r,"r_cut":3*r,"r_atom":r,"symbol":chemical_symbols[i],"volume": (4.*np.pi/3.)*(r)**3}

    print(radii)
    return radii, radii_by_symbol
"""

"""
#WH setup
def sample_radii(n_species):
    #r_atom=[0.37,2.0]
    radii={}
    radii_by_symbol={}
    fraction=np.random.uniform()
    print(fraction)
    sampler=NormalRadiusSampler([[0.36,0.36*0.2,0.36*0.8,0.36*1.2,fraction],[1.36,1.36*0.2,1.36*0.8,1.2*1.36,1-fraction]])
    #sampler=MendeleevCovalentRadiusSampler()
    for i in range(1,n_species+1):
        #r=np.random.uniform(low=r_atom[0],high=r_atom[1])
        r=sampler()
        radii[i]={"species_id":i, "r_min": 0.8*r ,"r_core":r,"r_cut":3*r,"r_atom":r,"symbol":chemical_symbols[i],"volume": (4.*np.pi/3.)*(r)**3}
        radii_by_symbol[chemical_symbols[i]]={"species_id":i, "r_min": 0.8*r ,"r_core":r,"r_cut":3*r,"r_atom":r,"symbol":chemical_symbols[i],"volume": (4.*np.pi/3.)*(r)**3}

    print(radii)
    return radii, radii_by_symbol
"""

"""
UCl setup
def sample_radii(n_species):
    r_atom=[0.32,2.0]
    radii={}
    radii_by_symbol={}
    #fraction=np.random.uniform()
    fraction=np.random.choice([0.666,0.75,0.8,0.8333])

    sampler=NormalRadiusSampler([[1.25,0.25,1,1.5,fraction],[2,0.4,1.6,2.4,1.-fraction]])
    #sampler=MendeleevCovalentRadiusSampler()
    for i in range(1,n_species+1):
        #r=np.random.uniform(low=r_atom[0],high=r_atom[1])
        r=sampler()
        radii[i]={"species_id":i, "r_min": 0.8*r ,"r_core":r,"r_cut":3*r,"r_atom":r,"symbol":chemical_symbols[i],"volume": (4.*np.pi/3.)*(r)**3}
        radii_by_symbol[chemical_symbols[i]]={"species_id":i, "r_min": 0.8*r ,"r_core":r,"r_cut":3*r,"r_atom":r,"symbol":chemical_symbols[i],"volume": (4.*np.pi/3.)*(r)**3}

    return radii, radii_by_symbol
"""

def generate_lammps_scripts(radii,file_prefix):
    n_species=len(radii)

    r_core_max=radii[max(radii, key=lambda key: radii[key]['r_core'])]['r_core']

    mliap_script=["neigh_modify one 10000 \n"]
    mliap_script.append("pair_style hybrid/overlay soft {} mliap model mliappy LATER descriptor sna {} \n".format(r_core_max, file_prefix+".mliap.descriptors") )
    for c in itertools.combinations_with_replacement(radii.keys(),2):
        mliap_script.append("pair_coeff {} {} soft 10 {} \n".format(c[0], c[1], (radii[c[0]]['r_core']+radii[c[1]]['r_core']) ) )
    mliap_script.append("pair_coeff * * mliap "+"".join([ v['symbol']+" "  for k,v in radii.items() ])+"\n")
    mliap_script=''.join(mliap_script)
    #print(mliap_script)

    zero_script=["neigh_modify one 10000 \n"]
    zero_script.append("pair_style soft {} \n".format(r_core_max) )
    for c in itertools.combinations_with_replacement(radii.keys(),2):
        zero_script.append("pair_coeff {} {} 10 {} \n".format(c[0], c[1], (radii[c[0]]['r_core']+radii[c[1]]['r_core']) ) )
    zero_script=''.join(zero_script)
    #print(zero_script)

    snap_descriptors=[ 
        "rcutfac 1 \n",
        "twojmax 8 \n",
        "nelems {} \n".format(n_species),
        "elems "+"".join([ v['symbol']+" "  for k,v in radii.items() ])+"\n",
        "radelems "+"".join([str(v['r_cut'])+" "  for k,v in radii.items() ])+"\n",
        "welems "+"".join([ "1 "  for k,v in radii.items() ])+"\n"
        "rfac0 0.99363 \n"
        "rmin0 0 \n"
        "bzeroflag 1\n"
    ]

    with open(file_prefix+".mliap.descriptors",'w') as wsnap:
        wsnap.writelines(snap_descriptors)


    snap_descriptors=''.join(snap_descriptors)
    #print(snap_descriptors)

    return mliap_script, zero_script, snap_descriptors





