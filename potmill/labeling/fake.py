import pandas as pd

from potmill.bfile import write_b


def fake_vasp(force_energy_filename, job_id):
    """Mock labeling for testing: read a precomputed (forces, energy) pickle and write a b file."""
    forces, ener = pd.read_pickle(force_energy_filename).iloc[job_id, [0, 2]]
    write_b("b", job_id, ener, forces.size // 3, forces)
    return job_id
