import numpy as np
import configparser
from itertools import product


def seq_to_string(value, delimiter=" "):
    """Format a hyperparameter value (scalar or list) as a delimited string."""
    if isinstance(value, (int, float)):
        return str(value)
    return delimiter.join(str(v) for v in value)


rcuts_to_string = nmaxes_to_string = lmaxes_to_string = twojmaxes_to_string = seq_to_string


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


def _int_grid(mins, maxs, combine):
    """Integer hyperparameter grid. Scalar -> [[v]] per value; list -> per-rank ranges
    combined either by cartesian 'product' (nmax/lmax) or element-wise 'zip' (twojmax)."""
    if isinstance(mins, int):
        return [[i] for i in range(mins, maxs + 1)]
    ranges = [np.arange(mins[i], maxs[i] + 1).tolist() for i in range(len(mins))]
    if combine == "product":
        return [list(c) for c in product(*ranges)]
    return np.vstack(ranges).T.tolist()


def create_nmax_range(min_nmax, max_nmax):
    return _int_grid(min_nmax, max_nmax, "product")


def create_lmax_range(min_lmax, max_lmax):
    return _int_grid(min_lmax, max_lmax, "product")


def create_twojmax_range(min_twojmax, max_twojmax):
    return _int_grid(min_twojmax, max_twojmax, "zip")


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
    except ValueError:
        try:
            return float(string)
        except ValueError:
            if ' ' in string:
                return [interpret_string(substring) for substring in string.split()]
            else:
                return string


def configparse(input_path):
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.optionxform = str
    config.read(input_path)
    return config