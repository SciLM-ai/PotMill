"""VASP labeling backend. Single-point DFT defaults (from vasp-ase-sp.py) are applied to every
configuration and any [Vasp] config key overrides them, so the user can tune any setting from
config.ini while the defaults work for any element. Per-atom initial MAGMOMs (also from
vasp-ase-sp.py; unknown elements -> 1.0) are set on the structure unless ispin=1. The special
keys ``pp_path`` and ``command`` set VASP_PP_PATH / the run command via environment; ``setups``
is parsed from a string into ASE's setups dict (config.ini cannot carry a dict)."""

import glob
import os
import traceback
import xml.etree.ElementTree as ET
from shutil import make_archive

import numpy as np
from ase import Atoms
from ase.calculators.vasp import Vasp
from ase.io import read, write

from potmill.bfile import b_rows, write_b

# Exact single-point DFT settings from vasp-ase-sp.py; each is overridable via a [Vasp] key.
_VASP_DEFAULTS = {
    "xc": "pbe",
    "encut": 500,
    "ismear": 0,
    "sigma": 0.1,
    "lwave": False,
    "pp": "pbe",
    "lcharg": False,
    "prec": "Accurate",
    "nelm": 200,
    "ediff": 1e-6,
    "kspacing": 0.125,
    "lorbit": 11,
}

# Element initial magnetic moments (pymatgen convention) from vasp-ase-sp.py; unknown -> 'default'.
# Setting these makes ASE write MAGMOM + ISPIN=2 generically for any element.
_MAGMOM = {
    "default": 1.0,
    "Co": 2.0,
    "Cr": 2.0,
    "Fe": 2.5,
    "Mn": 5.0,
    "Ni": 1.5,
    "Ce": 5.0,
    "Eu": 10.0,
    "Pr": 3.58,
    "Nd": 3.62,
    "Pm": 2.68,
    "Sm": 0.85,
    "Gd": 7.94,
    "Tb": 9.72,
    "Dy": 10.65,
    "Ho": 10.6,
    "Er": 9.58,
    "Tm": 7.56,
    "Yb": 4.54,
}


def parse_setups(value):
    """Turn the [Vasp] ``setups`` value into ASE's setups dict. Tokens without ':' set the base PAW
    set; ``El:label`` tokens set a per-element override -- e.g. 'recommended W:_sv' ->
    {'base': 'recommended', 'W': '_sv'}. interpret_string hands us a str (one token) or a list."""
    if isinstance(value, dict):
        return value
    tokens = value if isinstance(value, list) else str(value).split()
    setups = {}
    for raw in tokens:
        tok = str(raw)
        if ":" in tok:
            el, label = tok.split(":", 1)
            setups[el] = label
        else:
            setups["base"] = tok
    return setups


def make_init_vasp(config):
    """executorlib init_function: forward the [Vasp] kwargs to every labeling task on this worker."""
    vasp_kwargs = dict(config.get("Vasp", {}))

    def init_vasp():
        return {"vasp_kwargs": vasp_kwargs}

    return init_vasp


def vasp(start_path, input_file, job_id, dirpath, vasp_kwargs):
    os.makedirs(dirpath, exist_ok=True)
    os.chdir(dirpath)
    kwargs = {**_VASP_DEFAULTS, **vasp_kwargs}  # user [Vasp] keys override the defaults
    pp_path = kwargs.pop("pp_path", None)
    command = kwargs.pop("command", None)
    # interpret_string splits a spaced value (e.g. "flux run -n 24 ... vasp") into a token list;
    # rejoin it into the single shell command string ASE expects.
    if isinstance(command, list):
        command = " ".join(str(x) for x in command)
    if pp_path:
        os.environ["VASP_PP_PATH"] = pp_path
    if command:
        os.environ["VASP_COMMAND"] = command
        os.environ["ASE_VASP_COMMAND"] = command
    kwargs["setups"] = parse_setups(kwargs.get("setups", "recommended"))

    atoms = (
        input_file
        if isinstance(input_file, Atoms)
        else read(start_path + input_file, index=0, format="vasp")
    )
    atoms.pbc = True
    # Per-atom initial magnetic moments (any element; default 1.0) -> ASE writes MAGMOM + ISPIN=2,
    # reproducing vasp-ase-sp.py. Skipped when the user forces a non-spin run via [Vasp] ispin = 1.
    if int(kwargs.get("ispin", 2)) != 1:
        atoms.set_initial_magnetic_moments(
            [_MAGMOM.get(s, _MAGMOM["default"]) for s in atoms.get_chemical_symbols()]
        )
    print("RUN DIRECTORY: ", os.getcwd(), " INPUT FILE: ", input_file, flush=True)

    rows = None
    try:
        atoms.calc = Vasp(**kwargs)
        energy, forces = atoms.get_potential_energy(), atoms.get_forces()
        rows = b_rows(job_id, energy, len(atoms), forces)
        write_b("b", job_id, energy, len(atoms), forces)
        atoms.info["job_id"] = int(
            job_id
        )  # self-describing labeled traj (downstream keys composition on this)
        write(f"atoms_{job_id}.traj", images=atoms, format="traj")
        convert_xml_to_jason("vasprun.xml", "atoms_%i_" % job_id)
        _cleanup_vasp_files(job_id)
    except Exception:
        print(f"Error while running VASP or writing the output for job {job_id}", flush=True)
        traceback.print_exc()

    atoms.calc = None
    return {"job_ID": job_id, "b_rows": rows, "atoms": atoms}


def _cleanup_vasp_files(job_id):
    """Prune the VASP run directory, archive what's left, then keep only the essentials."""
    try:
        keep = [
            "b",
            "INCAR",
            "OUTCAR",
            "POSCAR",
            "vasprun.xml",
            "vasp.out",
            "vasp_output.log",
            f"atoms_{job_id}.traj",
            f"atoms_{job_id}_0.json",
            f"features_{job_id}.p",
        ]
        for f in glob.glob("*"):
            if f not in keep:
                os.remove(f)
        make_archive("archive", "gztar")
        keep = [
            "b",
            "archive.tar.gz",
            f"atoms_{job_id}.traj",
            f"atoms_{job_id}_0.json",
            f"features_{job_id}.p",
        ]
        for f in glob.glob("*"):
            if f not in keep:
                os.remove(f)
    except Exception:
        print("Error while cleaning up VASP files", flush=True)
        traceback.print_exc()


def write_json(data, jsonfilename):
    header = {
        "EnergyStyle": "electronvolt",
        "StressStyle": "kB",
        "AtomTypeStyle": "chemicalsymbol",
        "PositionsStyle": "angstrom",
        "ForcesStyle": "electronvoltperangstrom",
        "LatticeStyle": "angstrom",
        "Data": [data],
    }
    import json

    with open(jsonfilename, "w") as f:
        json.dump({"Dataset": header}, f, indent=2, sort_keys=True)


def convert_xml_to_jason(xml_file, JSON_file):
    """Convert each converged ionic step in a vasprun.xml into a FitSNAP JSON config file."""
    listAtomTypes, list_POTCARS = [], []
    config_number = 0
    natoms = 0
    all_lattice = atom_coords = None
    NELM = 0

    for event, elem in ET.iterparse(xml_file, events=["start", "end"]):
        if elem.tag == "parameters" and event == "end":
            NELM = int(
                elem.find(
                    'separator[@name="electronic"]/separator[@name="electronic convergence"]/i[@name="NELM"]'
                ).text
            )
        elif elem.tag == "atominfo" and event == "end":
            for entry in elem.find("array[@name='atoms']/set"):
                listAtomTypes.append(entry[0].text.strip())
            natoms = len(listAtomTypes)
            for entry in elem.find("array[@name='atomtypes']/set"):
                list_POTCARS.append(entry[4].text.strip().split())
        elif elem.tag == "structure" and not elem.attrib.get("name") and event == "end":
            all_lattice = [
                [float(x) for x in entry.text.split()]
                for entry in elem.find("crystal/varray[@name='basis']")
            ]
            frac = [
                [float(x) for x in entry.text.split()]
                for entry in elem.find("varray[@name='positions']")
            ]
            atom_coords = np.dot(frac, all_lattice).tolist()
        elif elem.tag == "calculation" and event == "end":
            force_block = elem.find("varray[@name='forces']")
            atom_force = (
                [[float(x) for x in entry.text.split()] for entry in force_block]
                if force_block
                else []
            )
            stress_block = elem.find("varray[@name='stress']")
            stress = (
                [[float(x) for x in entry.text.split()] for entry in stress_block]
                if stress_block
                else []
            )
            total_energy = float(elem.find('energy/i[@name="e_0_energy"]').text)
            converged = len(elem.findall("scstep")) != NELM

            data = {
                "Positions": atom_coords,
                "Lattice": all_lattice,
                "Energy": total_energy,
                "AtomTypes": listAtomTypes,
                "NumAtoms": natoms,
                "computation_code": "VASP",
                "pseudopotential_information": list_POTCARS,
            }
            if atom_force:
                data["Forces"] = atom_force
            if stress:
                data["Stress"] = stress
            if converged:
                write_json(data, JSON_file + str(config_number) + ".json")
            config_number += 1
