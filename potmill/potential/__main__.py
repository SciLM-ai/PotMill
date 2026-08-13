"""CLI: write LAMMPS potentials for a finished run.

    python -m potmill.potential <run_dir> [--which pareto|knee|all|none] [--batch N]
                                          [--verify [N]] [--md-steps N]

Does exactly what the in-pipeline ``[Main] potential = 1`` stage does, for runs that finished
before the stage existed, were run with it disabled, or were interrupted (the export uses the
latest completed checkpoint).

``--verify`` re-derives energies and forces through LAMMPS and compares them against the fitted
model. That check is intentionally not part of the pipeline -- it tests the writer and the
installed FitSNAP/LAMMPS rather than the run, so it is worth running after upgrading either.
"""

import argparse
import sys

from potmill.potential.export import WHICH_CHOICES, export_potentials


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m potmill.potential", description=__doc__)
    parser.add_argument("run_dir", help="PotMill run directory (holds config.ini, fits/, ...)")
    parser.add_argument("--which", default="pareto", choices=WHICH_CHOICES)
    parser.add_argument(
        "--batch", type=int, default=None, help="checkpoint index (default: the latest completed)"
    )
    parser.add_argument(
        "--verify",
        nargs="?",
        type=int,
        const=3,
        default=0,
        metavar="N",
        help="cross-check each written potential against the fitted model on N structures",
    )
    parser.add_argument(
        "--md-steps", type=int, default=0, help="also run this many NVE steps per potential"
    )
    args = parser.parse_args(argv)

    result = export_potentials(args.run_dir, which=args.which, batch=args.batch)
    ok = True
    if args.verify:
        from potmill.potential.verify import verify_written

        records = verify_written(
            args.run_dir, result, n_structures=args.verify, md_steps=args.md_steps
        )
        ok = all(r["ok"] for r in records)
    return 0 if (ok and not result["failed"]) else 1


if __name__ == "__main__":
    sys.exit(main())
