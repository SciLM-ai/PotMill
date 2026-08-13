"""The ``[Main] md`` stage: MD-test every exported potential, in parallel.

Three task types, all submitted UPFRONT with futures as dependencies -- nothing here inspects a
future's result at setup time, so the pipeline's dynamic overlap is untouched:

1. ``prepare_structure_task`` -- ONE task: pick and replicate the test structure (see
   ``md/structure.py``) and write it to ``md/structure.traj``. Done once because the auto path scans
   the run's labeled trajectories, which at 100k configurations is far too expensive to repeat per
   potential.
2. ``md_task`` -- ``[ourMD] max_potentials`` tasks, each claiming row ``i`` of
   ``potentials/index.csv`` and MD-testing that potential. A fixed task count is what keeps the
   setup data-independent: how many potentials the export actually wrote is a future's RESULT, so it
   cannot size the submission. Tasks whose row does not exist return immediately.
3. ``merge_md_task`` -- ONE task: writes ``potentials/md.csv`` (authoritative) and joins its columns
   into ``potentials/index.csv`` for convenience. Warns if the front was larger than the task cap,
   so a truncated test set can never look like a complete one.

Minimization is per potential and deliberately so: each fit is tested in its OWN relaxed structure,
which is the question a user actually has ("is this potential stable?"). It does mean the tested
geometry differs slightly between potentials, so a collapsed cell is reported rather than hidden.
"""

import os
import traceback

import pandas as pd

from potmill.config import ConfigManager, load_fitsnap_config

MD_COLUMNS = (
    "md_ok",
    "md_drift_per_atom_per_ps",
    "md_T_final",
    "md_min_dist",
    "md_natoms",
    "md_note",
)


def _need(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"REQUIRED FILE MISSING (stop, do not guess): {path}")
    return path


def _run_context(start_path, overrides=None):
    """Everything the MD tasks need out of the run's own config: elements, units, cutoff.

    ``overrides`` (the CLI's flags) are applied on top of ``[ourMD]`` so the CLI and the pipeline
    stage run exactly the same code with different settings.
    """
    config = ConfigManager(_need(start_path + "config.ini"))
    fitsnap_config = load_fitsnap_config(_need(start_path + config["FitSNAP"]["filename"]))
    section = "ACE" if config["FitSNAP"]["mlip"] == "ACE" else "BISPECTRUM"
    reference = fitsnap_config.get("REFERENCE", {})
    max_rcut = config["ourHyperparameters"]["max_rcut"]
    max_rcut = max(max_rcut) if isinstance(max_rcut, list) else float(max_rcut)
    md = dict(config["ourMD"])
    md.update({k: v for k, v in (overrides or {}).items() if v is not None})
    # ACE cannot evaluate below its inner cutoff (pair_pace raises "Encountered very small
    # distance"), so that is the floor a test structure has to clear -- with a small margin, since a
    # structure sitting exactly on the floor is where the potential is least trustworthy anyway.
    rcinner = [float(v) for v in str(fitsnap_config[section].get("rcinner", "0.0")).split()]
    return {
        "config": config,
        "md": md,
        "min_distance": max(rcinner) + 0.1 if rcinner else 0.1,
        "elements": str(fitsnap_config[section]["type"]).split(),
        "units": str(reference.get("units", "metal")).strip(),
        "atom_style": str(reference.get("atom_style", "atomic")).strip(),
        "max_rcut": float(max_rcut),
    }


def prepare_structure_task(start_path, *dependencies, overrides=None):  # noqa: ARG001 -- futures only
    """Pick + replicate the MD test structure once for the whole stage. Returns its path."""
    from ase.io import write

    from potmill.md.structure import prepare_structure

    try:
        start_path = os.path.abspath(start_path) + "/"
        ctx = _run_context(start_path, overrides)
        atoms, info = prepare_structure(
            start_path,
            spec=ctx["md"]["structure"],
            min_atoms=int(ctx["md"]["min_atoms"]),
            min_cell_length=2.0 * (ctx["max_rcut"] + 1.0),
            min_distance=ctx["min_distance"],
        )
        os.makedirs(start_path + "md", exist_ok=True)
        path = start_path + "md/structure.traj"
        write(path, atoms)
        with open(start_path + "md/structure.txt", "w") as f:
            for key, value in info.items():
                f.write(f"{key}: {value}\n")
        print(
            f"MD: test structure = {info['source']}, {info['natoms']} atoms "
            f"(replicated {info['replicated']}) -> {path}",
            flush=True,
        )
        return path
    except Exception as exc:  # noqa: BLE001 -- the run's artifacts are already safe
        print(
            f"ERROR: MD structure preparation FAILED ({type(exc).__name__}: {exc})\n"
            f"{traceback.format_exc()}",
            flush=True,
        )
        return None


def md_task(start_path, position, structure_path, *dependencies, overrides=None):  # noqa: ARG001
    """MD-test the potential on row ``position`` of ``potentials/index.csv`` (or return None)."""
    from ase.io import read

    from potmill.md.runner import run_md

    try:
        start_path = os.path.abspath(start_path) + "/"
        if not structure_path:
            return None  # preparation failed; it already reported why
        index_path = start_path + "potentials/index.csv"
        if not os.path.exists(index_path):
            return None
        index = pd.read_csv(index_path)
        index = index[index["status"] == "ok"].reset_index(drop=True)
        if position >= len(index):
            return None  # fewer potentials than tasks -- the normal case, nothing to report

        row = index.iloc[position]
        name = str(row["dir"])
        pot_dir = start_path + "potentials/" + name
        if not os.path.isdir(pot_dir):
            return None
        ctx = _run_context(start_path, overrides)
        md = ctx["md"]
        atoms = read(structure_path, index=0)
        atoms.calc = None

        result = run_md(
            atoms,
            pot_dir,
            name,
            ctx["elements"],
            ensemble=str(md["ensemble"]).lower(),
            temperature=float(md["temperature"]),
            timestep=float(md["timestep"]),
            steps=int(md["steps"]),
            minimize=bool(int(md["minimize"])),
            relax_box=bool(int(md["relax_box"])),
            # A trajectory that reaches the potential's inner cutoff has collapsed, whatever its
            # energy says -- the same floor the structure picker applies at the start.
            fused_distance=ctx["min_distance"],
            units=ctx["units"],
            atom_style=ctx["atom_style"],
        )
        record = {
            "dir": name,
            "md_ok": result["ok"],
            "md_drift_per_atom_per_ps": result["drift_per_atom_per_ps"],
            "md_T_final": result["T_final"],
            "md_min_dist": result["min_dist"],
            "md_natoms": result["natoms"],
            "md_note": result["note"],
        }
        print(
            f"MD {name}: ok={result['ok']} drift={result['drift_per_atom_per_ps']} eV/atom/ps "
            f"T_final={result['T_final']} min_dist={result['min_dist']} {result['note']}",
            flush=True,
        )
        return record
    except Exception as exc:  # noqa: BLE001 -- one potential must not lose the others
        print(
            f"ERROR: MD task {position} FAILED ({type(exc).__name__}: {exc})\n"
            f"{traceback.format_exc()}",
            flush=True,
        )
        return {"dir": f"<task {position}>", "md_ok": 0, "md_note": f"{type(exc).__name__}: {exc}"}


def merge_md_task(start_path, *records):
    """Collect the MD records into ``potentials/md.csv`` and join them into ``index.csv``."""
    try:
        start_path = os.path.abspath(start_path) + "/"
        rows = [r for r in records if isinstance(r, dict)]
        out_dir = start_path + "potentials/"
        if not rows:
            print("MD: no potentials were tested (nothing written).", flush=True)
            return None
        md_df = pd.DataFrame(rows)
        md_df.to_csv(out_dir + "md.csv", index=False)

        index_path = out_dir + "index.csv"
        if os.path.exists(index_path):
            index = pd.read_csv(index_path)
            index = index.drop(columns=[c for c in MD_COLUMNS if c in index.columns])
            index.merge(md_df, on="dir", how="left").to_csv(index_path, index=False)

        tested = len(rows)
        exported = (
            int((pd.read_csv(index_path)["status"] == "ok").sum())
            if os.path.exists(index_path)
            else tested
        )
        if exported > tested:
            print(
                f"WARNING: {exported} potentials were exported but only {tested} were MD-tested "
                f"([ourMD] max_potentials caps the number of MD tasks). Raise max_potentials to "
                f"test them all; md.csv covers only the {tested} tested.",
                flush=True,
            )
        unstable = [r["dir"] for r in rows if not r.get("md_ok")]
        print(
            f"MD: {tested - len(unstable)}/{tested} potentials ran stable MD -> {out_dir}md.csv"
            + (f"; UNSTABLE: {', '.join(unstable)}" if unstable else ""),
            flush=True,
        )
        return out_dir + "md.csv"
    except Exception as exc:  # noqa: BLE001
        print(
            f"ERROR: MD merge FAILED ({type(exc).__name__}: {exc})\n{traceback.format_exc()}",
            flush=True,
        )
        return None
