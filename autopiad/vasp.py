import numpy as np
import os, glob, sys, json
import xml.etree.ElementTree as ET
from ase import Atoms
from ase.calculators.vasp import Vasp
from ase.io import read, write
from shutil import make_archive


def write_json(data, jsonfilename):
    jsonfile = open(jsonfilename, "w")
    allData = [data]
    allDataHeader = {}
    allDataHeader["EnergyStyle"] = "electronvolt"
    allDataHeader["StressStyle"] = "kB"
    allDataHeader["AtomTypeStyle"] = "chemicalsymbol"
    allDataHeader["PositionsStyle"] = "angstrom"
    allDataHeader["ForcesStyle"] = "electronvoltperangstrom"
    allDataHeader["LatticeStyle"] = "angstrom"
    allDataHeader["Data"] = allData

    myDataset = {}

    myDataset["Dataset"] = allDataHeader
    #jsonfile.write(json.dumps(myDataset))  # if you want a condensed string
    json.dump(myDataset, jsonfile, indent=2, sort_keys=True)  #if you want the expanded, multi-line format
    jsonfile.close()
    return


def convert_xml_to_jason(xml_file, JSON_file):
    write_unconverged_steps_anyway = False

    order_atom_types = []
    listAtomTypes = []
    list_POTCARS = []
    config_number = 0

    # Start parsing through vasprun.xml looking for entries that are associated with the
    # different values for the data needed, such as forces or positions
    tree = ET.iterparse(xml_file, events=['start', 'end'])
    for event, elem in tree:
        if elem.tag == 'parameters' and event=='end': #once at the start
            NELM = int(elem.find('separator[@name="electronic"]/separator[@name="electronic convergence"]/i[@name="NELM"]').text)
            
        elif elem.tag == 'atominfo' and event == 'end': #once at the start
            for entry in elem.find("array[@name='atoms']/set"):
                listAtomTypes.append(entry[0].text.strip())
            natoms = len(listAtomTypes)
            for entry in elem.find("array[@name='atomtypes']/set"):
                list_POTCARS.append(entry[4].text.strip().split())
            
        elif (elem.tag == 'structure' and not elem.attrib.get('name')) and event=='end': #only the empty name ones - not primitive cell, initial, or final (those are repeats) - so each ionic step
            all_lattice = []
            for entry in elem.find("crystal/varray[@name='basis']"):
                lattice_row = [float(x) for x in entry.text.split()]
                all_lattice.append(lattice_row)

            frac_atom_coords = []
            for entry in elem.find("varray[@name='positions']"):
                frac_atom_coords.append([float(x) for x in entry.text.split()])
            atom_coords = np.dot(frac_atom_coords, all_lattice).tolist()
            
        elif elem.tag == 'calculation' and event=='end': #this triggers each ionic step
            atom_force = []
            force_block = elem.find("varray[@name='forces']")
            if force_block:
                for entry in force_block:
                    atom_force.append([float(x) for x in entry.text.split()])

            stress_block = elem.find("varray[@name='stress']")
            stress_component = []
            if stress_block:
                for entry in stress_block:
                    stress_component.append([float(x) for x in entry.text.split()])
            totalEnergy = float(elem.find('energy/i[@name="e_0_energy"]').text)  ##NOTE! this value is incorrectly reported by VASP in version 5.4 (fixed in 6.1), see https://www.vasp.at/forum/viewtopic.php?t=17839
            ## ASE vasprun.xml io reader has a more complex workaround to get the correct energy - we can update to include if needed

            if len(elem.findall("scstep")) == NELM:
                electronic_convergence = False ##This isn't the best way to check this, but not sure if info is directly available. Could try to calculate energy diff from scstep entries and compare to EDIFF
            else:
                electronic_convergence = True
            
            # Here is where all the data is put together for each ionic step
            # After this, all these values will be overwritten
            # once the next configuration appears in the sequence when parsing
            data = {}
            data["Positions"] = atom_coords
            if atom_force:
                data["Forces"] = atom_force
            if stress_component:
                data["Stress"] = stress_component
            data["Lattice"] = all_lattice
            data["Energy"] = totalEnergy
            data["AtomTypes"] = listAtomTypes
            data["NumAtoms"] = natoms
            data["computation_code"] = "VASP"
            data["pseudopotential_information"] = list_POTCARS

            # Specify jsonfilename and put this and data into the write_json function.  All
            # json files should be output now.  The configuration number will be increased by one
            # to keep track of which configuration is associated with which json file.

            jsonfilename = JSON_file + str(config_number) + ".json"

            if electronic_convergence:
                write_json(data, jsonfilename)
            else:
                if write_unconverged_steps_anyway:
                    write_json(data, jsonfilename)

            config_number += 1 


def vasp(start_path, input_file, job_id, first_index, dirpath):

    os.chdir(dirpath)

    os.environ['VASP_PP_PATH'] = "/users/baghishov/pyiron/resources/vasp/potentials/"
    os.environ['VASP_COMMAND'] = start_path+"run_vasp_6.3.2_std_ase.sh > vasp_output.log 2>&1"
    os.environ['ASE_VASP_COMMAND'] = start_path+"run_vasp_6.3.2_std_ase.sh > vasp_output.log 2>&1"

    # #check whether this task has been executed already. If so, skip it
    # if os.path.isfile("b"):
    #     sys.exit(0)

    #have to set this up accordingly
    try:
        calc = Vasp(xc='pbe',  # Select exchange-correlation functional
                    encut=300, # Plane-wave cutoff
                    ismear=1, lwave=False, lcharg=False, prec='Normal', nelm=100, ediff=1e-6, kspacing=1.0,
                    setups={'Re':'','W':''})#, directory=run_directory)  # setups='recommended'
    except:
        try:
            calc = Vasp(xc='pbe',  # Select exchange-correlation functional
                        encut=500, # Plane-wave cutoff
                        ismear=1, lwave=False, lcharg=False, prec='Normal', nelm=100, ediff=1e-6, kspacing=0.5,
                        setups='recommended')#, directory=run_directory)  # setups='recommended'
        except:
            raise

    if isinstance(input_file, Atoms):
        atoms = input_file
    else:
        atoms = read(start_path+input_file, index=0, format='vasp')
    atoms.pbc = True
    atoms.calc = calc

    print("RUN DIRECTORY: ", os.getcwd(), " INPUT FILE: ", input_file)

    #execute the calculation
    try:
        ener = atoms.get_potential_energy()
        forces = atoms.get_forces().ravel()

        n_atoms = len(atoms)
        b = np.vstack([np.arange(first_index,first_index+1+3*n_atoms),
                       np.full(1+3*n_atoms,job_id),
                       np.concatenate([np.array([ener])/n_atoms,forces])]).T
        np.savetxt("b", b, delimiter=',', fmt=['%i','%i','%.10f'])

        #write the output in ASE traj format
        write(f"atoms_{job_id}.traj", images=atoms, format='traj')

        #look into using Custodian here to do error detection/validation

        #write a json file for fitsnap
        convert_xml_to_jason("vasprun.xml", "atoms_%i_" % job_id )

        try:
            #do some cleanup
            files_to_keep=["b","INCAR","OUTCAR","POSCAR","vasprun.xml","vasp.out","vasp_output.log",
                           "atoms_%i.traj" % job_id, "atoms_%i_0.json" % job_id , "features_%i.p" % job_id ]
            absolute_files_to_keep=[]
            for file in files_to_keep:
                absolute_files_to_keep.append(file)

            output_files = glob.glob("*")

            for file in output_files:
                if not file in absolute_files_to_keep:
                    #pass
                    os.remove(file)

            archive_name = "archive"
            make_archive(archive_name, 'gztar')

            #Keep only some files. Clean up the rest
            output_files = glob.glob("*")
            files_to_keep = ["b","archive.tar.gz","atoms_%i.traj"%job_id, "atoms_%i_0.json"%job_id, "features_%i.p"%job_id]
            absolute_files_to_keep = []
            for file in files_to_keep:
                absolute_files_to_keep.append(file)

            output_files = glob.glob("*")

            for file in output_files:
                if not file in absolute_files_to_keep:
                    os.remove(file)
            
        except:
            print("Error while cleaning up files")
    except:
        print("Error while running VASP or writing the output files")

    atoms.calc = None
    return {"job_ID":job_id, "atoms":atoms}