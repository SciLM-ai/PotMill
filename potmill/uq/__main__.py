"""CLI: attach POPS uncertainties to a finished run's exported potentials, or use them.

    python -m potmill.uq <run_dir>                      # fit + write <name>.uq.npz for every
                                                        # potential in potentials/index.csv
    python -m potmill.uq <run_dir> --potential <name>   # just one of them
    python -m potmill.uq <run_dir> --predict frames.xyz # energy +/- uncertainty for structures,
                                                        # using the run's knee potential (or --potential)

The fitting half is the same code the ``[Main] uq`` pipeline stage runs, so a run that was launched
with ``uq = 0``, or interrupted, or exported again with a wider selection, can be completed
afterwards without re-running anything expensive.
"""

import argparse
import os
import sys

import pandas as pd

from potmill.config import ConfigManager
from potmill.uq.stage import UQ_COLUMNS, merge_uq_task, write_uq


def _potential_names(run_dir, only=None):
    index_path = os.path.join(run_dir, "potentials", "index.csv")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"{index_path} does not exist -- export potentials first: "
            f"python -m potmill.potential {run_dir} (stop)"
        )
    index = pd.read_csv(index_path)
    names = [str(n) for n in index[index["status"] == "ok"]["dir"]]
    if only is None:
        return names, index
    if only not in names:
        raise ValueError(f"'{only}' is not an exported potential in {index_path} (stop)")
    return [only], index


def _predict(run_dir, path, name, index, level):
    from ase.io import read

    from potmill.uq.calculator import PotMillCalculator

    if name is None:
        knee = index[(index["status"] == "ok") & (index.get("knee", 0) == 1)]
        pick = knee if len(knee) else index[index["status"] == "ok"]
        name = str(pick.iloc[0]["dir"])
    calc = PotMillCalculator(os.path.join(run_dir, "potentials", name))
    structures = read(path, index=":")
    print(f"UQ: {name}, {len(structures)} structure(s) from {path}\n")
    print(
        f"{'#':>4}  {'natoms':>6}  {'E (eV)':>14}  {'E/atom':>12}  "
        f"{'+/- (eV/atom)':>13}  {'worst case (eV/atom)':>24}"
    )
    for i, atoms in enumerate(structures):
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        sigma = calc.get_uncertainty(atoms, level=level)
        low, high = calc.get_bounds(atoms)
        print(
            f"{i:>4}  {len(atoms):>6}  {energy:>14.6f}  {energy / len(atoms):>12.6f}  "
            f"{sigma:>13.6f}  {f'[{low:+.6f}, {high:+.6f}]':>24}"
        )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m potmill.uq", description=__doc__)
    parser.add_argument("run_dir", help="a PotMill run directory (the one holding config.ini)")
    parser.add_argument("--potential", default=None, help="only this potential (default: all)")
    parser.add_argument("--predict", default=None, help="ASE-readable structure file to evaluate")
    parser.add_argument(
        "--level",
        type=float,
        default=0.68,
        help="calibration level for --predict (0.68 or 0.95; default 0.68)",
    )
    parser.add_argument("--posterior", default=None, choices=["hypercube", "ensemble"])
    parser.add_argument("--minimum-relative-error", type=float, default=None)
    parser.add_argument("--percentile-clipping", type=float, default=None)
    parser.add_argument(
        "--batch", type=int, default=None, help="checkpoint to use (default: the newest on disk)"
    )
    args = parser.parse_args(argv)

    run_dir = os.path.abspath(args.run_dir) + "/"
    if not os.path.exists(run_dir + "config.ini"):
        parser.error(f"{run_dir}config.ini not found -- is this a PotMill run directory?")
    names, index = _potential_names(run_dir, args.potential)

    if args.predict:
        return _predict(run_dir, args.predict, args.potential, index, args.level)

    settings = dict(ConfigManager(run_dir + "config.ini")["ourUQ"])
    for key, value in (
        ("posterior", args.posterior),
        ("minimum_relative_error", args.minimum_relative_error),
        ("percentile_clipping", args.percentile_clipping),
    ):
        if value is not None:
            settings[key] = value

    records, failed = [], []
    for name in names:
        try:
            records.append(write_uq(run_dir, name, settings, batch=args.batch))
        except Exception as exc:  # noqa: BLE001 -- one potential must not lose the others
            import traceback

            failed.append(name)
            print(
                f"ERROR: UQ FAILED for {name} ({type(exc).__name__}: {exc})\n"
                f"{traceback.format_exc()}",
                flush=True,
            )
            records.append({"dir": name, "uq_note": f"{type(exc).__name__}: {exc}"})
    merge_uq_task(run_dir, *records)
    print(f"\nColumns joined into potentials/index.csv: {', '.join(UQ_COLUMNS)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
