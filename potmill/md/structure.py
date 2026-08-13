"""Choosing the structure an MD stability test should run on.

This matters more than it looks. Every configuration PotMill generates is entropy-MAXIMIZED:
deliberately strange, far from equilibrium, and small (2-25 atoms in the reference example).
Starting MD from one of those tells you almost nothing about a potential -- the cell converts its
excess potential energy into heat and runs away (measured: a 300 K start reaching 5200 K) no matter
how good the fit is. So the auto path picks the most equilibrium-like structure the run produced and
makes it big enough for the dynamics to mean something:

1. shortlist by lowest FORMATION energy per atom -- not lowest total energy, which just favours the
   smallest cell, and not lowest energy per atom, which favours whichever composition happens to
   bind strongest. ``analysis._recon.formation_energy`` already removes composition by a
   least-squares per-element reference fit, so its zero point is exactly the region wanted here.
2. from that shortlist, take the LEAST COMPRESSED cell that the potential can evaluate at all.
   Energy alone is not enough: on a real run the same four potentials came back 4/4 stable from one
   low-energy candidate and 0/4 from another, purely because the second had a squeezed contact.
3. replicate until the cell is at least twice the potential's cutoff in every direction (otherwise
   atoms interact with their own periodic images and the test is an artifact) and holds at least
   ``min_atoms`` atoms.

A user-supplied structure is used AS GIVEN -- no replication, no second-guessing: if someone points
the stage at a specific cell, that is the cell they want tested.
"""

import os

import numpy as np


def _need(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"REQUIRED FILE MISSING (stop, do not guess): {path}")
    return path


def min_interatomic_distance(atoms, cutoff=3.0):
    """Closest approach under PBC, or None if no pair is within ``cutoff``."""
    from ase.neighborlist import neighbor_list

    if len(atoms) < 2:
        return None
    distances = neighbor_list("d", atoms, cutoff)
    return float(distances.min()) if len(distances) else None


def closest_contact(atoms, cutoff=3.5):
    """``(ratio, "H-W", distance)`` for the most compressed pair -- what actually collapsed.

    Knowing the ELEMENTS matters when reading a failure: a hydrogen pair at 0.9 A is ordinary, a
    tungsten pair at 1.5 A is a collapse, and "min distance 0.354 A" alone tells you neither which
    interaction the fit got wrong nor where to add training data.
    """
    from ase.data import chemical_symbols, covalent_radii
    from ase.neighborlist import neighbor_list

    if len(atoms) < 2:
        return None
    numbers = atoms.get_atomic_numbers()
    i, j, d = neighbor_list("ijd", atoms, cutoff)
    if not len(d):
        return None
    ratios = d / (covalent_radii[numbers[i]] + covalent_radii[numbers[j]])
    k = int(ratios.argmin())
    pair = "-".join(sorted((chemical_symbols[numbers[i[k]]], chemical_symbols[numbers[j[k]]])))
    return float(ratios[k]), pair, float(d[k])


def compression(atoms, cutoff=3.5):
    """How compressed the tightest contact is, as ``d / (r_i + r_j)`` with covalent radii.

    1.0 means the closest pair sits exactly at the sum of its covalent radii; below 1.0 it is
    squeezed. Normalizing by the PAIR's radii is what makes candidates comparable across
    compositions -- a raw distance would always favour the tungsten-rich cells over the
    hydrogen-rich ones simply because tungsten is bigger.

    Radii come from ``ase.data`` rather than the mendeleev table ``structuregen`` samples: only the
    RANKING of candidates matters here, ase is a hard dependency everywhere this runs (mendeleev is
    not -- it is absent from the CI test environment), and ase.data needs no database read.
    """
    contact = closest_contact(atoms, cutoff)
    return contact[0] if contact else None


def structures_by_job_ids(run_dir, wanted):
    """``{job_id: Atoms}`` for a set of job_ids, in ONE pass over the labeled trajectories.

    One pass matters: at 100k configurations, re-scanning per candidate would dominate the stage.
    """
    import glob

    from ase.io import read

    run_dir = run_dir.rstrip("/") + "/"
    wanted = {int(j) for j in wanted}
    found = {}
    for job_id in list(wanted):  # VASP/LAMMPS layout: one file per configuration
        per_config = f"{run_dir}labeling/{job_id}/atoms_{job_id}.traj"
        if os.path.exists(per_config):
            atoms = read(per_config, index=0)
            atoms.calc = None
            found[job_id] = atoms
    if len(found) == len(wanted):
        return found
    for path in sorted(glob.glob(run_dir + "labeling/labeled_*.traj")):  # UMA/GRACE: per worker
        for atoms in read(path, index=":"):
            job_id = int(atoms.info.get("job_id", -1))
            if job_id in wanted and job_id not in found:
                atoms.calc = None
                found[job_id] = atoms
        if len(found) == len(wanted):
            break
    return found


def structure_by_job_id(run_dir, job_id):
    """The labeled configuration with this job_id, from whichever traj layout the run used."""
    found = structures_by_job_ids(run_dir, [job_id])
    if int(job_id) not in found:
        raise FileNotFoundError(
            f"configuration {job_id} not found under {run_dir}/labeling/ (stop, do not substitute "
            f"another structure)"
        )
    return found[int(job_id)]


def lowest_formation_energy_structure(run_dir, min_distance=0.0, n_candidates=20):
    """The run's most equilibrium-like configuration that MD can actually start from.

    Formation energy shortlists the candidates (``n_candidates``); ``select_evaluable`` then chooses
    among them. Everything about that choice -- which configuration, its energy rank, how compressed
    it is, how many were rejected -- is returned as provenance and written to ``md/structure.txt``.
    """
    from potmill.analysis._recon import formation_energy, load_run

    cwd = os.getcwd()
    try:
        dE = formation_energy(load_run(run_dir))["dE"]
    finally:
        os.chdir(cwd)  # featurize-adjacent helpers chdir; never leave the caller somewhere else
    if not dE:
        raise ValueError(f"no formation energies could be computed for {run_dir} (stop)")

    ranked = sorted(dE, key=dE.get)[:n_candidates]
    return select_evaluable(ranked, structures_by_job_ids(run_dir, ranked), dE, min_distance)


def select_evaluable(ranked, structures, dE, min_distance):
    """The LEAST compressed of the low-formation-energy candidates that MD can start from.

    Two criteria, and both are needed. The hard one: a structure whose closest pair sits inside
    ACE's inner cutoff cannot be evaluated at all (``pair_pace``: "Encountered very small distance"),
    so it would fail every potential for a reason that has nothing to do with any of them. The soft
    one: among what remains, the most relaxed cell is the fairest starting point -- on one real run
    the same four potentials came back 4/4 stable from a candidate at 2.39 A and 0/4 from one at
    1.47 A, so picking on energy alone makes the verdict depend on how strained the winner happened
    to be. Candidates are drawn only from the lowest-energy end of the run, so this stays a choice
    among near-equilibrium structures rather than a hunt for the emptiest cell.
    """
    usable, rejected = [], []
    for rank, job_id in enumerate(ranked):
        atoms = structures.get(int(job_id))
        if atoms is None:
            continue
        closest = min_interatomic_distance(atoms)
        if closest is not None and closest < min_distance:
            rejected.append((int(job_id), closest))
            continue
        usable.append((rank, int(job_id), atoms, closest, compression(atoms)))

    if not usable:
        raise ValueError(
            f"none of the {len(ranked)} lowest formation-energy configurations has all atoms at "
            f"least {min_distance:.2f} A apart (closest approaches: "
            f"{[round(d, 3) for _, d in rejected[:5]]}) -- every structure this run produced is too "
            f"compressed for the potential to evaluate. Set [ourMD] structure = <path> to supply "
            f"one (stop, do not test on a structure MD cannot start from)"
        )

    rank, job_id, atoms, closest, squeeze = max(
        usable, key=lambda c: (c[4] if c[4] is not None else float("inf"), -c[0])
    )
    return atoms, {
        "job_id": job_id,
        "dE_form": float(dE[job_id]),
        "rank_by_formation_energy": rank,
        "min_distance": closest,
        "min_distance_required": min_distance,
        "compression": squeeze,  # d/(r_i+r_j) of the tightest contact; >= 1 is uncompressed
        "candidates_considered": len(ranked),
        "rejected_too_close": len(rejected),
    }


def replicate_for_md(atoms, min_atoms=200, min_cell_length=0.0):
    """Repeat the cell until MD in it is not dominated by self-interaction.

    Each lattice vector is repeated until it is at least ``min_cell_length`` (twice the potential
    cutoff), then the whole cell is repeated further until it holds ``min_atoms`` atoms.
    """
    lengths = np.linalg.norm(np.asarray(atoms.get_cell()), axis=1)
    if np.any(lengths <= 0):
        raise ValueError(f"structure has a degenerate cell {lengths} -- cannot run MD (stop)")
    reps = np.maximum(np.ceil(min_cell_length / lengths).astype(int), 1)
    while len(atoms) * int(np.prod(reps)) < min_atoms:
        reps[int(np.argmin(lengths * reps))] += 1  # grow the shortest direction first
    return atoms * tuple(int(r) for r in reps), tuple(int(r) for r in reps)


def prepare_structure(run_dir, spec="auto", min_atoms=200, min_cell_length=0.0, min_distance=0.0):
    """``(atoms, provenance)`` for the MD test: the run's own best structure, or a user-supplied one."""
    if str(spec).strip().lower() != "auto":
        from ase.io import read

        path = spec if os.path.isabs(spec) else os.path.join(run_dir, spec)
        atoms = read(_need(path), index=0)
        atoms.calc = None
        # A supplied structure is used AS GIVEN -- but its closest approach is reported, because it
        # is the first thing to look at if the potential cannot even start from it.
        return atoms, {
            "source": path,
            "natoms": len(atoms),
            "replicated": "as given",
            "min_distance": min_interatomic_distance(atoms),
        }

    atoms, info = lowest_formation_energy_structure(run_dir, min_distance=min_distance)
    natoms_before = len(atoms)
    atoms, reps = replicate_for_md(atoms, min_atoms=min_atoms, min_cell_length=min_cell_length)
    info.update(
        {
            "source": "auto (lowest formation energy per atom)",
            "natoms_original": natoms_before,
            "natoms": len(atoms),
            "replicated": "x".join(str(r) for r in reps),
        }
    )
    return atoms, info
