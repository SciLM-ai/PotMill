"""``PotMillCalculator``: an ASE calculator that answers ``get_uncertainty()`` as well as
``get_forces()``.

    from potmill.uq import PotMillCalculator

    atoms.calc = PotMillCalculator("potentials/rcut_5.0__nmax_9_4__lmax_0_4__eweight_10.0")
    atoms.get_potential_energy()          # eV        -- LAMMPS running the exported .yace
    atoms.get_forces()                    # eV/A
    atoms.calc.get_uncertainty(atoms)     # eV/atom   -- calibrated 68% error bar (POPS)
    atoms.calc.get_uncertainty(atoms, level=0.95)
    atoms.calc.get_bounds(atoms)          # (low, high) eV/atom, worst case over the POPS set

Everything comes out of the potential directory the pipeline wrote, so a directory copied to another
machine keeps working: the ``.yace`` and ``.mod`` for energies and forces, and the ``.uq.npz`` --
which carries the coefficients, the descriptor columns, the POPS posterior, the calibration and the
FitSNAP input itself -- for uncertainties.

**Two engines, and why they are checked against each other.** Energies and forces come from LAMMPS
(the same evaluator a production run uses, milliseconds per call), while the uncertainty needs the
DESCRIPTOR row for the structure, which means a FitSNAP featurization (~a second per structure).
The first time both have been computed for the same structure, ``E_lammps`` is compared with
``natoms * (x_E @ beta)`` -- and they must agree, because that identity is the whole reason a linear
model can be shipped as a LAMMPS potential at all.  A mismatch means the ``.uq.npz`` and the
``.yace`` are not describing the same potential, so it raises instead of quietly attaching an error
bar to the wrong model.

**This is a screening calculator, not an MD engine.** Each energy call spins up a LAMMPS instance,
and each uncertainty call runs a featurization. For dynamics, run LAMMPS directly with the exported
``.mod`` (``include <name>.mod``) -- and note the uncertainty is a property of a STRUCTURE (a single
energy row), so screening a trajectory is best done by writing frames and evaluating them in a
batch with :meth:`uncertainties`.
"""

import copy
import glob
import io
import os
import tempfile

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from potmill.uq.artifact import load_uq


class PotMillCalculator(Calculator):
    """ASE calculator over an exported PotMill potential, with POPS uncertainties."""

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, potential_dir, engine="lammps", check=True, **kwargs):
        """``potential_dir``: a ``potentials/<name>/`` directory (or the ``.uq.npz``/``.mod`` in it).

        ``engine`` selects where energies and forces come from: ``"lammps"`` (default, fast) or
        ``"fitsnap"`` (from the same featurization the uncertainty uses -- no LAMMPS needed, and
        exactly what the fit predicts by construction).  ``check=False`` skips the one-off
        cross-check between them.
        """
        super().__init__(**kwargs)
        self.potential_dir, self.potential_name = _resolve(potential_dir)
        self.engine = engine
        self.check = check
        self._uq = None
        self._checked = False
        self._row_key = self._row_cache = None
        if engine not in ("lammps", "fitsnap"):
            raise ValueError(f"engine must be 'lammps' or 'fitsnap', got '{engine}' (stop)")
        self.meta = _mod_metadata(f"{self.potential_dir}/{self.potential_name}.mod")

    # ---- the uncertainty model ------------------------------------------------------------------

    @property
    def uq(self):
        """``(posterior, beta, column_indices, meta)`` from ``<name>.uq.npz``, loaded once."""
        if self._uq is None:
            path = f"{self.potential_dir}/{self.potential_name}.uq.npz"
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{path} does not exist, so this potential carries no uncertainty model. "
                    f"Re-run the run's UQ stage: python -m potmill.uq <run_dir> (stop)"
                )
            self._uq = load_uq(path)
        return self._uq

    def descriptors(self, atoms):
        """The FitSNAP design matrix for one structure, column-filtered to this potential's basis.

        Row 0 is the per-atom energy row; rows ``1 .. 3N`` are the force rows -- the same layout the
        fit consumed, which is what makes ``a @ beta`` the fitted model's own prediction.
        """
        from potmill.featurization.featurize import featurize

        _, beta, columns, meta = self.uq
        fitsnap_config = _parse_ini(str(meta["fitsnap_in"]))
        # The stored column indices address the FULL swept basis, so the structure must be
        # featurized at that basis and filtered -- exactly as the fit did.
        fitsnap_config["ACE"]["nmax"] = str(meta["full_nmax"])
        fitsnap_config["ACE"]["lmax"] = str(meta["full_lmax"])
        rcuts = [float(v) for v in str(meta["rcut"]).split()]

        probe = atoms.copy()
        probe.calc = None  # a labeled calculator lacks stress and would trip the ASE scraper
        prev = os.getcwd()
        with tempfile.TemporaryDirectory() as workdir:
            try:
                featurize([probe], {"FitSNAP": {"mlip": "ACE"}}, fitsnap_config, rcuts, workdir)
            finally:
                os.chdir(prev)  # featurize chdirs into its feature directory and does not return
            matrix = np.load(os.path.join(workdir, "a.npy"))
        if matrix.shape[1] <= int(columns.max()):
            raise ValueError(
                f"featurization produced {matrix.shape[1]} descriptor columns but the uncertainty "
                f"model indexes column {int(columns.max())} -- the installed FitSNAP does not "
                f"build the basis this potential was fitted with (stop)"
            )
        selected = matrix[:, columns]
        if selected.shape[1] != len(beta):
            raise ValueError(
                f"{selected.shape[1]} selected columns != {len(beta)} coefficients (stop)"
            )
        return selected

    def uncertainties(self, atoms_list, level=0.68):
        """Uncertainties for many structures (one featurization each). Returns an array, eV/atom."""
        return np.array([self.get_uncertainty(a, level=level) for a in atoms_list])

    def get_uncertainty(self, atoms=None, level=0.68):
        """Predicted energy uncertainty for ``atoms``, in eV/atom.

        ``level`` is the calibration target: ``0.68`` (default) and ``0.95`` return the raw POPS
        standard deviation scaled by the split-conformal factor measured on HELD-OUT configurations
        of the training run, so "68%" means 68% of those had an error no larger. ``level=None``
        returns the raw, uncalibrated POPS standard deviation.

        Total-energy uncertainty is ``natoms *`` this: the model's energy is a sum of per-atom
        contributions from ONE parameter vector, so the per-atom errors are perfectly correlated
        and do NOT average down.
        """
        atoms = self.atoms if atoms is None else atoms
        posterior, beta, _, meta = self.uq
        row = self._energy_row(atoms, beta)
        sigma = float(posterior.std(row[None, :])[0])
        return sigma * _calibration_factor(meta, level)

    def get_bounds(self, atoms=None):
        """``(low, high)`` per-atom energy offsets: the worst case over the whole POPS set.

        Unlike the standard deviation this is a hard bracket -- the extreme predictions of any model
        in the hypercube of pointwise-optimal parameters -- so ``E/natoms + low`` and
        ``E/natoms + high`` bound what a plausibly-different fit of the same data would have said.
        Requires ``[ourUQ] posterior = hypercube`` (the default).
        """
        atoms = self.atoms if atoms is None else atoms
        posterior, beta, _, _ = self.uq
        low, high = posterior.bounds(self._energy_row(atoms, beta)[None, :])
        return float(low[0]), float(high[0])

    def get_ensemble(self, n_samples=10, seed=0):
        """``n_samples`` coefficient vectors drawn from the POPS set, as a ``(n_samples, p)`` array.

        For running an actual committee: each column is ``beta + theta``, a full alternative
        potential that fits the same data comparably well. Write them out with
        ``potmill.potential.ace.write_yace`` to get a committee of ``.yace`` files.
        """
        posterior, beta, _, _ = self.uq
        return beta[None, :] + posterior.sample(n_samples, seed=seed).T

    def _energy_row(self, atoms, beta):
        """The structure's energy descriptor row, cross-checking LAMMPS against it once.

        Cached on the exact geometry, so asking for 68% and 95% bounds on one structure costs one
        featurization rather than three.
        """
        key = (
            atoms.positions.tobytes(),
            atoms.cell[:].tobytes(),
            atoms.numbers.tobytes(),
            atoms.pbc.tobytes(),
        )
        if key == self._row_key:
            return self._row_cache
        row = self.descriptors(atoms)[0]
        self._row_key, self._row_cache = key, row
        if self.check and not self._checked and self.engine == "lammps":
            self._checked = True  # once per calculator: this tests the FILES, not the structure
            reference = len(atoms) * float(row @ beta)
            energy = self._lammps(atoms)[0]
            if abs(energy - reference) > 1e-6 * max(1, len(atoms)):
                raise ValueError(
                    f"{self.potential_dir}: LAMMPS gives {energy:.9f} eV for this structure but the "
                    f"uncertainty model's own coefficients give {reference:.9f} eV "
                    f"({abs(energy - reference) / len(atoms):.2e} eV/atom apart). The .yace and the "
                    f".uq.npz are not the same potential, so any error bar from this pair would be "
                    f"attached to the wrong model (stop). Re-export both from the same run."
                )
        return row

    # ---- ASE plumbing ---------------------------------------------------------------------------

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if self.engine == "lammps":
            energy, forces = self._lammps(self.atoms)
        else:
            _, beta, _, _ = self.uq
            matrix = self.descriptors(self.atoms)
            natoms = len(self.atoms)
            energy = float(natoms * (matrix[0] @ beta))
            forces = (matrix[1 : 1 + 3 * natoms] @ beta).reshape(natoms, 3)
        self.results = {"energy": energy, "free_energy": energy, "forces": forces}

    def _lammps(self, atoms):
        from potmill.potential.verify import lammps_ef

        return lammps_ef(
            atoms,
            self.potential_dir,
            f"{self.potential_name}.mod",
            self.meta["elements"],
            units=self.meta["units"],
            atom_style=self.meta["atom_style"],
        )


def _calibration_factor(meta, level):
    if level is None:
        return 1.0
    key = f"calib_q{int(round(float(level) * 100))}"
    if key not in meta:
        available = sorted(k[7:] for k in meta if k.startswith("calib_q"))
        raise ValueError(
            f"no calibration for level {level} in this uncertainty model (have: {available}); "
            f"pass level=None for the raw POPS standard deviation (stop)"
        )
    return float(meta[key])


def _resolve(path):
    """``(directory, potential name)`` from a potential directory or any file inside it."""
    path = os.path.abspath(path)
    if os.path.isfile(path):
        directory = os.path.dirname(path)
        base = os.path.basename(path)
        for suffix in (".uq.npz", ".yace", ".mod"):
            if base.endswith(suffix):
                return directory, base[: -len(suffix)]
        raise ValueError(f"{path} is not a .yace / .mod / .uq.npz file (stop)")
    hits = sorted(glob.glob(os.path.join(path, "*.yace")))
    if len(hits) != 1:
        raise ValueError(
            f"{path} contains {len(hits)} .yace files (expected exactly 1) -- point the calculator "
            f"at one potential directory, or at the .yace itself (stop)"
        )
    return path, os.path.basename(hits[0])[: -len(".yace")]


def _mod_metadata(mod_path):
    """Elements, units and atom_style as the ``.mod`` itself declares them.

    Read from the file rather than assumed, because a hybrid ``[REFERENCE]`` changes the pair_coeff
    layout and a non-metal ``units`` would silently rescale every energy.
    """
    if not os.path.exists(mod_path):
        raise FileNotFoundError(
            f"{mod_path} does not exist -- a PotMill potential directory holds <name>.yace and "
            f"<name>.mod (stop)"
        )
    units, atom_style, elements = "metal", "atomic", None
    with open(mod_path) as f:
        for line in f:
            text = line.strip()
            if text.startswith("# Requires:") and "units" in text:
                parts = text.replace("/", " ").split()
                units = parts[parts.index("units") + 1]
                atom_style = parts[parts.index("atom_style") + 1]
            elif text.startswith("pair_coeff") and ".yace" in text:
                tokens = text.split()
                elements = tokens[[t.endswith(".yace") for t in tokens].index(True) + 1 :]
    if not elements:
        raise ValueError(f"{mod_path} has no 'pair_coeff ... .yace <elements>' line (stop)")
    return {"units": units, "atom_style": atom_style, "elements": elements}


def _parse_ini(text):
    """A FitSNAP.in stored as text, back into the ``{section: {key: value}}`` dict FitSNAP wants."""
    import configparser

    parser = configparser.ConfigParser(inline_comment_prefixes="#")
    parser.optionxform = str
    parser.read_file(io.StringIO(text))
    return copy.deepcopy({s: dict(parser.items(s)) for s in parser.sections()})
