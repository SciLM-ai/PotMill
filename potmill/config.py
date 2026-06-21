"""Configuration management for PotMill, modeled on SaddleMill's ConfigManager.

The pipeline is configured by a ``config.ini`` whose sections are of two kinds:

* "our" sections (``Main``, ``FitSNAP``, and the per-stage ``our*`` sections) carry PotMill's
  own parameters. Their defaults live in ``ConfigManager.DEFAULTS`` (the single source of truth),
  values are type-coerced, and unknown keys are warned about to catch typos.
* passthrough sections (``FAIRChemCalculator``, ``Vasp``, ``LAMMPS``) carry keyword arguments
  for external calculator classes. Their keys are forwarded verbatim and are NOT validated --
  anything the user omits falls back to that library's own default. ``ourStructureGen`` is also
  left raw because its defaults are method-dependent and resolved inside ``structuregen``.
"""

import configparser
import copy
import os
import re

from potmill.tools import configparse, interpret_string

PASSTHROUGH_SECTIONS = ("FAIRChemCalculator", "Vasp", "LAMMPS")
RAW_SECTIONS = ("ourStructureGen", *PASSTHROUGH_SECTIONS)


class ConfigManager:
    """Parse ``config.ini``, apply defaults to the "our" sections, and validate."""

    DEFAULTS = {
        "Main": {
            "resume": 0,
            "entropy": 1,
            "featurize": 1,
            "labeling": 1,
            "fit": 1,
            "pareto": 1,
            "pops": 0,
            "nconfigurations": 1000,
            "batch_size": 1000,
            # device drives the labeling + fitting executors: "cuda" = one GPU per job (today's
            # behavior); "cpu" = cores per job (VASP/CPU). entropy + featurize are always CPU.
            "device": "cuda",
        },
        "FitSNAP": {"mlip": "ACE", "chem_elem": None, "filename": "FitSNAP.in"},
        # Per-stage layout uses one consistent scheme: <stage>_jobs_per_node + <stage>_cores_per_job.
        # In cuda mode each labeling/fit job takes 1 GPU and cores_per_job is its CPU thread count;
        # in cpu mode cores_per_job is the cores reserved for that job. (entropy_* live in the raw
        # [ourStructureGen] section; defaults resolved in resources.worker_layout.)
        "ourLabeling": {
            "calculator": "FAIRChemCalculator",
            "label_batch_size": 1,
            "labeling_jobs_per_node": 1,
            "labeling_cores_per_job": 1,
        },
        "ourFeaturization": {"featurize_jobs_per_node": 1, "featurize_cores_per_job": 4},
        "ourFit": {
            "fit_jobs_per_node": 2,
            "fit_method": "svd",
            "n_fold": 3,
            "fit_engine": "incremental",
            "fit_cores_per_job": 1,
        },
        "ourHyperparameters": {
            "min_rcut": 5.0,
            "max_rcut": 6.5,
            "num_rcut": 4,
            "min_nmax": 5,
            "max_nmax": 9,
            "min_lmax": 0,
            "max_lmax": 4,
            "min_twojmax": 6,
            "max_twojmax": 8,
            "middle_eweight": 10,
            "num_eweights": 5,
        },
    }

    def __init__(self, config_file="config.ini"):
        self._config = copy.deepcopy(self.DEFAULTS)
        for section in RAW_SECTIONS:
            self._config.setdefault(section, {})
        if os.path.exists(config_file):
            self._load(config_file)
        else:
            print(f"Warning: {config_file} not found; using defaults.", flush=True)

    def _load(self, config_file):
        parser = configparser.ConfigParser(inline_comment_prefixes="#")
        parser.optionxform = str
        parser.read(config_file)
        for section in parser.sections():
            self._config.setdefault(section, {})
            for key, value in parser.items(section):
                self._config[section][key] = interpret_string(value)
        self._warn_unknown(parser)

    def _warn_unknown(self, parser):
        for section in parser.sections():
            if section in self.DEFAULTS:
                for key in sorted(set(parser.options(section)) - set(self.DEFAULTS[section])):
                    print(f"Warning: unrecognized key '{key}' in [{section}].", flush=True)
            elif section not in RAW_SECTIONS:
                print(f"Warning: unrecognized section [{section}].", flush=True)

    def __getitem__(self, key):
        return self._config[key]

    def __contains__(self, key):
        return key in self._config

    def get(self, key, default=None):
        return self._config.get(key, default)

    def get_value(self, section, key, default=None):
        return self._config.get(section, {}).get(key, default)

    @property
    def as_dict(self):
        return self._config

    def validate(self, fitsnap_config):
        """Warn (do NOT override -- users may have custom pair_style setups) when the FitSNAP.in
        [REFERENCE] pair_style cutoff is below [ourHyperparameters] max_rcut. LAMMPS compute pace
        aborts every featurize task with rcut > pair_style cutoff (src/ML-PACE/compute_pace.cpp:129)."""
        match = re.match(
            r"\s*zero\s+([0-9.]+)", fitsnap_config.get("REFERENCE", {}).get("pair_style", "")
        )
        if not match:
            return
        max_rcut = self._config["ourHyperparameters"]["max_rcut"]
        max_rcut = max(max_rcut) if isinstance(max_rcut, list) else float(max_rcut)
        ps_cut = float(match.group(1))
        if ps_cut < max_rcut:
            print(
                f"WARNING: FitSNAP.in [REFERENCE] pair_style cutoff ({ps_cut}) < [ourHyperparameters] max_rcut "
                f"({max_rcut}). LAMMPS will abort featurize tasks with 'compute pace cutoff > "
                f"pairwise cutoff'. FIX FitSNAP.in:  pair_style = zero {max_rcut + 0.1}",
                flush=True,
            )


def load_fitsnap_config(path):
    """Parse a FitSNAP.in into a plain ``{section: {key: value}}`` dict (values left as strings)."""
    parser = configparse(path)
    return {section: dict(parser.items(section)) for section in parser.sections()}
