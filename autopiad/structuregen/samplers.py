import numpy as np
import itertools
from scipy import stats
from scipy.stats.sampling import NumericalInverseHermite

# Nearest-neighbor distances from Kittel, Introduction to Solid State Physics, 8th ed.
NN_DISTS = {
    "H": 0.75, "Be": 2.22, "C": 1.54, "Al": 2.86,
    "W": 2.74, "Re": 2.74, "Os": 2.68, "Pu": 3.1,
    "U": 2.75, "O": 1.2, "Sb": 2.91, "Te": 2.86, "Cs": 5.24,
}


def _random_combination_with_replacement(iterable, r):
    """Random selection from itertools.combinations_with_replacement(iterable, r)."""
    import random
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n + r - 1), k=r))
    return tuple(pool[i - j] for j, i in enumerate(indices))


class BinaryRadiusSampler:
    """Radius sampler for binary element systems.

    Generates radii for exactly two elements based on nearest-neighbor
    distances. Samples core radii from a grid around the NN distances
    with specified scaling range and step.

    For the renormalization phase, returns random radii from the grid.
    For the optimization phase, can return fixed radii at a given grid index.
    """

    def __init__(self, elements, scale_min=0.7, scale_max=1.8, scale_step=0.15):
        if len(elements) != 2:
            raise ValueError("BinaryRadiusSampler requires exactly 2 elements, "
                             "got {}".format(len(elements)))
        self.elements = elements
        self.nn_dists = {e: NN_DISTS[e] for e in elements}

        scales = np.arange(scale_min, scale_max, scale_step)
        # Core radius grids per element
        self.core_radii_grids = {e: self.nn_dists[e] * scales for e in elements}
        # Cross-element NN distance
        self.nn_dist_cross = (self.nn_dists[elements[0]] / 2.0 +
                              self.nn_dists[elements[1]] / 2.0)
        self.core_radii_cross_grid = self.nn_dist_cross * scales

        # Build sampling grid: all combinations of element radii and cross radii
        elem0_grid = self.core_radii_grids[elements[0]]
        self.radii_to_sample = [
            [r0, r_cross]
            for r0 in elem0_grid
            for r_cross in self.core_radii_cross_grid
        ]

    def sample_radii(self, n_atoms, n_first_elem):
        """Sample radii for one configuration (renorm phase).

        Returns three independent core radii plus helper data. Element 0
        radius and cross-element radius are randomly sampled from the grid;
        element 1 radius is fixed at its NN distance.

        Args:
            n_atoms: Total number of atoms.
            n_first_elem: Number of atoms of the first element.

        Returns:
            (core_radius_0, core_radius_1, core_radius_cross,
             atom_types, symbols) where:
            - core_radius_0: Core radius for element 0 (sampled).
            - core_radius_1: Core radius for element 1 (fixed at NN dist).
            - core_radius_cross: Cross-element core radius (independently sampled).
            - atom_types: dict mapping symbol -> type_id.
            - symbols: list of element symbols for each atom.
        """
        import random

        rad = random.choice(self.radii_to_sample)
        core_radius_0 = rad[0]
        core_radius_1 = self.nn_dists[self.elements[1]]
        core_radius_cross = rad[1]

        atom_types = {elem: idx + 1 for idx, elem in enumerate(self.elements)}
        symbols = (int(n_first_elem) * [self.elements[0]] +
                   int(n_atoms - n_first_elem) * [self.elements[1]])

        return core_radius_0, core_radius_1, core_radius_cross, atom_types, symbols

    def sample_radii_fixed(self, n_atoms, n_first_elem, grid_index=10,
                           scale_step=0.18):
        """Sample radii with fixed core radii (optimizer phase).

        Uses a coarser grid (scale_step=0.18 by default) and picks a fixed
        index from it, keeping radii constant across all configurations.

        Returns:
            (core_radius_0, core_radius_1, core_radius_cross,
             atom_types, symbols)
        """
        scales = np.arange(0.7, 1.8, scale_step)
        elem0_grid = self.nn_dists[self.elements[0]] * scales
        cross_grid = self.nn_dist_cross * scales

        radii_grid = [
            [r0, r_cross]
            for r0 in elem0_grid
            for r_cross in cross_grid
        ]
        rad = radii_grid[min(grid_index, len(radii_grid) - 1)]
        core_radius_0 = rad[0]
        core_radius_1 = self.nn_dists[self.elements[1]]
        core_radius_cross = rad[1]

        atom_types = {elem: idx + 1 for idx, elem in enumerate(self.elements)}
        symbols = (int(n_first_elem) * [self.elements[0]] +
                   int(n_atoms - n_first_elem) * [self.elements[1]])

        return core_radius_0, core_radius_1, core_radius_cross, atom_types, symbols


class MendeleevUniformRadiusSampler:
    """Radius sampler for multi-element systems using Mendeleev covalent radii.

    Samples element species from a weighted probability distribution and
    generates per-atom radii using a beta distribution centered on the
    Pyykko covalent radius for each element. Each atom becomes a unique
    pseudo-species with its own LAMMPS type and cutoff radius.

    Args:
        species: List of candidate element symbols.
        width: Relative width of the radius distribution.
        a: Alpha parameter of the beta distribution.
        b: Beta parameter of the beta distribution.
        fixed_stoichiometry: If True, cycle through species; if False, sample.
    """

    def __init__(self, species, width, a, b, fixed_stoichiometry=False):
        from mendeleev.fetch import fetch_table
        self.ptable = fetch_table("elements")
        self.radii_dict = dict(zip(
            self.ptable['symbol'], self.ptable['covalent_radius_pyykko']))
        self.species = species
        self.width = width
        self.a = a
        self.b = b
        self.beta = stats.beta(a=a, b=b)
        self.beta_dist = NumericalInverseHermite(self.beta)
        self.fixed_stoichiometry = fixed_stoichiometry

    def __call__(self, n_atoms, n_species=None, probabilities=None):
        """Sample radii for n_atoms pseudo-species.

        Args:
            n_atoms: Number of atoms (each becomes a pseudo-species).
            n_species: Number of distinct real species to sample.
                       If None, defaults to n_atoms.
            probabilities: Sampling weights for self.species.

        Returns:
            (radii, radii_by_symbol) where:
            - radii: dict mapping type_id -> {species_id, original_symbol,
              symbol, r_atom, r_min, r_core, r_cut, volume}
            - radii_by_symbol: dict mapping pseudo_symbol -> same dict
        """
        if self.fixed_stoichiometry:
            comb = tuple(list(
                itertools.islice(itertools.cycle(self.species), n_atoms)))
        else:
            if n_species is None:
                n_species = n_atoms
            if probabilities is not None:
                species = np.random.choice(
                    self.species, n_species, p=probabilities, replace=False)
            else:
                species = np.random.choice(self.species, n_species)
            comb = _random_combination_with_replacement(species, n_atoms)

        radii = {}
        radii_by_symbol = {}
        for i, c in enumerate(comb):
            r = self.beta_dist.rvs()
            r = ((r - 0.5) * self.width * 2.0 + 1) * self.radii_dict[c] / 100.0
            pseudo_symbol = self.ptable['symbol'][i % len(self.ptable['symbol'])]
            entry = {
                "species_id": i + 1,
                "original_symbol": c,
                "symbol": pseudo_symbol,
                "r_atom": r,
                "r_min": 0.8 * r,
                "r_core": r,
                "r_cut": 3 * r,
                "volume": (4.0 * np.pi / 3.0) * r ** 3,
            }
            radii[i + 1] = entry
            radii_by_symbol[pseudo_symbol] = entry

        return radii, radii_by_symbol
