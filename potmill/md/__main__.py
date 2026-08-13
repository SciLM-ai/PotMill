"""CLI: MD-test the exported potentials of a run.

    python -m potmill.md <run_dir> [--limit N] [--steps N] [--temperature K]
                                   [--ensemble nve|nvt] [--structure auto|PATH] [--no-minimize]

Same work the in-pipeline ``[Main] md = 1`` stage does, run serially here, for runs exported before
the stage existed or to re-test with different MD settings. Command-line values override
``[ourMD]``; anything not given comes from the run's own ``config.ini``.
"""

import argparse
import sys

from potmill.md.stage import md_task, merge_md_task, prepare_structure_task


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m potmill.md", description=__doc__)
    parser.add_argument("run_dir", help="PotMill run directory (must already have potentials/)")
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N", help="test only the first N potentials"
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--timestep", type=float, default=None, help="ps")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--ensemble", choices=("nve", "nvt"), default=None)
    parser.add_argument("--structure", default=None, help="'auto' or a path to a structure file")
    parser.add_argument("--no-minimize", action="store_true", help="skip relaxation before MD")
    args = parser.parse_args(argv)

    from potmill.config import ConfigManager

    overrides = {
        "steps": args.steps,
        "timestep": args.timestep,
        "temperature": args.temperature,
        "ensemble": args.ensemble,
        "structure": args.structure,
        "minimize": 0 if args.no_minimize else None,
    }

    structure_path = prepare_structure_task(args.run_dir, overrides=overrides)
    if structure_path is None:
        return 1
    config = ConfigManager(args.run_dir.rstrip("/") + "/config.ini")
    limit = args.limit if args.limit is not None else int(config["ourMD"]["max_potentials"])
    records = []
    for position in range(limit):
        record = md_task(args.run_dir, position, structure_path, overrides=overrides)
        if record is None:
            break  # positions are contiguous: the first miss means there are no more potentials
        records.append(record)
    merge_md_task(args.run_dir, *records)
    return 0 if records and all(r.get("md_ok") for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
