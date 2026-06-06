"""VASP labeling backend. All calculator keywords come from the [Vasp] config section and are
forwarded verbatim to ase.calculators.vasp.Vasp; anything omitted uses ASE's own defaults. The
special keys ``pp_path`` and ``command`` set VASP_PP_PATH / the run command via environment."""

import os
import glob
import traceback
import xml.etree.ElementTree as ET
from shutil import make_archive

import numpy as np
from ase import Atoms
from ase.calculators.vasp import Vasp
from ase.io import read, write

from potmill.bfile import write_b


def make_init_vasp(config):
    """executorlib init_function: forward the [Vasp] kwargs to every labeling task on this worker."""
    vasp_kwargs = dict(config.get("Vasp", {}))

    def init_vasp():
        return {"vasp_kwargs": vasp_kwargs}
    return init_vasp


def vasp(start_path, input_file, job_id, dirpath, vasp_kwargs):
    os.chdir(dirpath)
    kwargs = dict(vasp_kwargs)
    pp_path = kwargs.pop("pp_path", None)
    command = kwargs.pop("command", None)
    if pp_path:
        os.environ["VASP_PP_PATH"] = pp_path
    if command:
        os.environ["VASP_COMMAND"] = command
        os.environ["ASE_VASP_COMMAND"] = command

    atoms = input_file if isinstance(input_file, Atoms) else read(start_path + input_file, index=0, format='vasp')
    atoms.pbc = True
    print("RUN DIRECTORY: ", os.getcwd(), " INPUT FILE: ", input_file, flush=True)

    try:
        atoms.calc = Vasp(**kwargs)
        write_b("b", job_id, atoms.get_potential_energy(), len(atoms), atoms.get_forces())
        write(f"atoms_{job_id}.traj", images=atoms, format='traj')
        convert_xml_to_jason("vasprun.xml", "atoms_%i_" % job_id)
        _cleanup_vasp_files(job_id)
    except Exception:
        print(f"Error while running VASP or writing the output for job {job_id}", flush=True)
        traceback.print_exc()

    atoms.calc = None
    return {"job_ID": job_id, "atoms": atoms}


def _cleanup_vasp_files(job_id):
    """Prune the VASP run directory, archive what's left, then keep only the essentials."""
    try:
        keep = ["b", "INCAR", "OUTCAR", "POSCAR", "vasprun.xml", "vasp.out", "vasp_output.log",
                f"atoms_{job_id}.traj", f"atoms_{job_id}_0.json", f"features_{job_id}.p"]
        for f in glob.glob("*"):
            if f not in keep:
                os.remove(f)
        make_archive("archive", "gztar")
        keep = ["b", "archive.tar.gz", f"atoms_{job_id}.traj", f"atoms_{job_id}_0.json", f"features_{job_id}.p"]
        for f in glob.glob("*"):
            if f not in keep:
                os.remove(f)
    except Exception:
        print("Error while cleaning up VASP files", flush=True)
        traceback.print_exc()


def write_json(data, jsonfilename):
    header = {
        "EnergyStyle": "electronvolt", "StressStyle": "kB", "AtomTypeStyle": "chemicalsymbol",
        "PositionsStyle": "angstrom", "ForcesStyle": "electronvoltperangstrom",
        "LatticeStyle": "angstrom", "Data": [data],
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

    for event, elem in ET.iterparse(xml_file, events=['start', 'end']):
        if elem.tag == 'parameters' and event == 'end':
            NELM = int(elem.find('separator[@name="electronic"]/separator[@name="electronic convergence"]/i[@name="NELM"]').text)
        elif elem.tag == 'atominfo' and event == 'end':
            for entry in elem.find("array[@name='atoms']/set"):
                listAtomTypes.append(entry[0].text.strip())
            natoms = len(listAtomTypes)
            for entry in elem.find("array[@name='atomtypes']/set"):
                list_POTCARS.append(entry[4].text.strip().split())
        elif elem.tag == 'structure' and not elem.attrib.get('name') and event == 'end':
            all_lattice = [[float(x) for x in entry.text.split()]
                           for entry in elem.find("crystal/varray[@name='basis']")]
            frac = [[float(x) for x in entry.text.split()]
                    for entry in elem.find("varray[@name='positions']")]
            atom_coords = np.dot(frac, all_lattice).tolist()
        elif elem.tag == 'calculation' and event == 'end':
            force_block = elem.find("varray[@name='forces']")
            atom_force = [[float(x) for x in entry.text.split()] for entry in force_block] if force_block else []
            stress_block = elem.find("varray[@name='stress']")
            stress = [[float(x) for x in entry.text.split()] for entry in stress_block] if stress_block else []
            total_energy = float(elem.find('energy/i[@name="e_0_energy"]').text)
            converged = len(elem.findall("scstep")) != NELM

            data = {"Positions": atom_coords, "Lattice": all_lattice, "Energy": total_energy,
                    "AtomTypes": listAtomTypes, "NumAtoms": natoms, "computation_code": "VASP",
                    "pseudopotential_information": list_POTCARS}
            if atom_force:
                data["Forces"] = atom_force
            if stress:
                data["Stress"] = stress
            if converged:
                write_json(data, JSON_file + str(config_number) + ".json")
            config_number += 1
