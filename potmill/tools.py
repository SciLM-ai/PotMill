import numpy as np
import configparser
import sys
from itertools import product


def rcuts_to_string(rcuts, delimiter=" "):
    if isinstance(rcuts,int) or isinstance(rcuts,float): return str(rcuts)
    if isinstance(rcuts,list): return delimiter.join([str(rcut) for rcut in rcuts])


def nmaxes_to_string(nmaxes, delimiter=" "):
    if isinstance(nmaxes,int): return str(nmaxes)
    if isinstance(nmaxes,list): return delimiter.join([str(nmax) for nmax in nmaxes])


def lmaxes_to_string(lmaxes, delimiter=" "):
    if isinstance(lmaxes,int): return str(lmaxes)
    if isinstance(lmaxes,list): return delimiter.join([str(lmax) for lmax in lmaxes])


def twojmaxes_to_string(twojmaxes, delimiter=" "):
    if isinstance(twojmaxes,int): return str(twojmaxes)
    if isinstance(twojmaxes,list): return delimiter.join([str(twojmax) for twojmax in twojmaxes])


def hyperparameters_to_string(mlip, hyperparameters, delimiter=" ", w_eweight=True):
    if mlip.upper() == "ACE":
        return ace_hyperparameters_to_string(hyperparameters, delimiter, w_eweight)
    elif mlip.upper() == "SNAP":
        return snap_hyperparameters_to_string(hyperparameters, delimiter, w_eweight)
    else:
        print("MLIP types supported are only ACE and SNAP", flush=True)


def ace_hyperparameters_to_string(hyperparameters, delimiter=" ", w_eweight=True):
    rcut_string = rcuts_to_string(hyperparameters[0], delimiter)
    nmax_string = nmaxes_to_string(hyperparameters[1], delimiter)
    lmax_string = lmaxes_to_string(hyperparameters[2], delimiter)
    if w_eweight:
        return rcut_string + delimiter + nmax_string + delimiter + lmax_string + delimiter + "%.1f"%hyperparameters[3]
    else:
        return rcut_string + delimiter + nmax_string + delimiter + lmax_string


def snap_hyperparameters_to_string(hyperparameters, delimiter=" ", w_eweight=True):
    rcut_string = rcuts_to_string(hyperparameters[0], delimiter)
    twojmax_string = twojmaxes_to_string(hyperparameters[1], delimiter)
    if w_eweight:
        return rcut_string + delimiter + twojmax_string + delimiter + "%.1f"%hyperparameters[2]
    else:
        return rcut_string + delimiter + twojmax_string


def create_rcut_range(min_rcut,max_rcut,num_rcut):
    if isinstance(min_rcut,list):
        rcut_range = [np.linspace(min_rcut[i],max_rcut[i],num_rcut[i]).tolist() for i in range(len(min_rcut))]
        rcut_range = [list(rcut_list) for rcut_list in product(*rcut_range)]
    if isinstance(min_rcut,float) or isinstance(min_rcut,int):
        rcut_range = np.linspace(min_rcut,max_rcut,num_rcut).reshape(-1,1).tolist()
    return rcut_range


def create_nmax_range(min_nmax,max_nmax):
    if isinstance(min_nmax,list):
        nmax_range = [np.arange(min_nmax[i],max_nmax[i]+1).tolist() for i in range(len(min_nmax))]
        # nmax_range = np.vstack(nmax_range).T.tolist()
        nmax_range = [list(nmax_list) for nmax_list in product(*nmax_range)]
    if isinstance(min_nmax,int):   
        nmax_range = [[i] for i in range(min_nmax,max_nmax+1)]
    return nmax_range


def create_lmax_range(min_lmax,max_lmax):
    if isinstance(min_lmax,list):
        lmax_range = [np.arange(min_lmax[i],max_lmax[i]+1).tolist() for i in range(len(min_lmax))]
        # lmax_range = np.vstack(lmax_range).T.tolist()
        lmax_range = [list(lmax_list) for lmax_list in product(*lmax_range)]
    if isinstance(min_lmax,int):   
        lmax_range = [[i] for i in range(min_lmax,max_lmax+1)]
    return lmax_range


def create_twojmax_range(min_twojmax,max_twojmax):
    if isinstance(min_twojmax,list):
        twojmax_range = [np.arange(min_twojmax[i],max_twojmax[i]+1).tolist() for i in range(len(min_twojmax))]
        twojmax_range = np.vstack(twojmax_range).T.tolist()
        # twojmax_range = [list(twojmax_list) for twojmax_list in product(*twojmax_range)]
    if isinstance(min_twojmax,int):   
        twojmax_range = [[i] for i in range(min_twojmax,max_twojmax+1)]
    return twojmax_range


def create_eweight_range(middle_eweight,n_eweights):
    eweight_list = [middle_eweight*2**i for i in range((1-n_eweights)//2,(n_eweights+1)//2)]
    return eweight_list


def combined_ace_hyperparameters(config, w_eweight=True):
    rcut_range = create_rcut_range(config['RCUT']["min_rcut"], config['RCUT']["max_rcut"], config['RCUT']["num_rcut"])
    nmax_range = create_nmax_range(config['NMAX']["min_nmax"], config['NMAX']["max_nmax"])
    lmax_range = create_lmax_range(config['LMAX']["min_lmax"], config['LMAX']["max_lmax"])
    eweight_range = create_eweight_range(config['EWEIGHT']["middle_eweight"], config['EWEIGHT']["num_eweights"])
    if w_eweight:
        hyperparameters_list = [[rcut_list,nmax_list,lmax_list,eweight] for rcut_list in rcut_range
                                for nmax_list in nmax_range for lmax_list in lmax_range for eweight in eweight_range]
    else:
        hyperparameters_list = [[rcut_list,nmax_list,lmax_list] for rcut_list in rcut_range for nmax_list in nmax_range
                                for lmax_list in lmax_range]
    return hyperparameters_list


def combined_snap_hyperparameters(config, w_eweight=True):
    rcut_range = create_rcut_range(config['RCUT']["min_rcut"], config['RCUT']["max_rcut"], config['RCUT']["num_rcut"])
    twojmax_range = create_twojmax_range(config['TWOJMAX']["min_twojmax"], config['TWOJMAX']["max_twojmax"])
    eweight_range = create_eweight_range(config['EWEIGHT']["middle_eweight"], config['EWEIGHT']["num_eweights"])
    if w_eweight:
        hyperparameters_list = [[rcut_list,twojmax_list,eweight] for rcut_list in rcut_range for twojmax_list in twojmax_range
                                       for eweight in eweight_range]
    else:
        hyperparameters_list = [[rcut_list,twojmax_list] for rcut_list in rcut_range for twojmax_list in twojmax_range]
    return hyperparameters_list


def interpret_string(string):
    try:
        return int(string)
    except:
        try:
            return float(string)
        except:
            if ' ' in string:
                return [interpret_string(substring) for substring in string.split()]
            else:
                return string


def configparse(input_path):
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.optionxform = str
    config.read(input_path)
    return config


def parse_inputfile(input_path):
    config = configparse(input_path)
    config_dict = {}
    for section in config.sections():
        config_dict[section] = {}
        for option in config.options(section):
            config_dict[section][option] = interpret_string(config[section][option])
    return config_dict


def update_fitsnap_config(config,chem_elem,rcut_list,twojmax_list):
    config['BISPECTRUM']['radelem'] = rcuts_to_string([rcut/2 for rcut in rcut_list])
    config['BISPECTRUM']['twojmax'] = twojmaxes_to_string(twojmax_list)
    config['BISPECTRUM']['type'] = ' '.join(chem_elem)
    return config


def interpret_data_format(parent_dirpath, config, fitsnap_config):
    if config['DATA']['format'] == 'ase':
        pass
    
    elif config['DATA']['format'] == 'json':
        if not 'SCRAPER' in fitsnap_config.sections():
            fitsnap_config.add_section('SCRAPER')
            fitsnap_config.set('SCRAPER','scraper','JSON')
        else:
            fitsnap_config['SCRAPER']['scraper'] = 'JSON'

        if not 'PATH' in fitsnap_config.sections():
            fitsnap_config.add_section('PATH')
            fitsnap_config.set('PATH','dataPath',parent_dirpath+config['DATA']['data_path'])
        else:
            fitsnap_config['PATH']['dataPath'] = parent_dirpath+config['DATA']['data_path']

        with open(parent_dirpath+config['FitSNAP']['filename'],'w') as configfile:
            fitsnap_config.write(configfile)

    else:
        sys.exit("Invalid data format. Please check the inputfile.")
    return 0