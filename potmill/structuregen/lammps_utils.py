import itertools


def count_snap_bispectrum(twojmax):
    """Count the number of SNAP bispectrum components for a given twojmax.

    Uses the standard LAMMPS SNAP counting with default diagonal=3:
    all (j1, j2, j) with j2 <= j1, |j1-j2| <= j <= min(twojmax, j1+j2),
    j1+j2+j even (enforced by step of 2), and j >= j1.

    Reference values: twojmax=4 -> 14, twojmax=6 -> 30, twojmax=8 -> 55.
    """
    count = 0
    for j1 in range(twojmax + 1):
        for j2 in range(j1 + 1):
            for j in range(abs(j1 - j2), min(twojmax, j1 + j2) + 1, 2):
                if j >= j1:
                    count += 1
    return count


def compute_n_descriptors(twojmax, n_elements, chemflag, bzeroflag):
    """Compute total number of SNAP descriptors per atom.

    With chemflag=1, each bispectrum component is expanded into
    n_elements^3 chemical-aware components.
    """
    n_bispec = count_snap_bispectrum(twojmax)
    if chemflag:
        return n_bispec * (n_elements ** 3)
    return n_bispec


def write_mliap_descriptor(filename, elements, rcutfac, twojmax, radelems,
                           chemflag=0, bzeroflag=0):
    """Write a LAMMPS MLIAP SNAP descriptor file.

    Args:
        filename: Output file path.
        elements: List of element symbols in LAMMPS type order.
        rcutfac: Cutoff factor (multiplied by radelems to get per-element cutoff).
        twojmax: Maximum angular momentum quantum number (2j).
        radelems: List of per-element radial parameters, or a pre-formatted string.
        chemflag: Whether to use chemically-aware descriptors (0 or 1).
        bzeroflag: Whether to include B0 components (0 or 1).
    """
    if isinstance(radelems, (list, tuple)):
        radelems_str = " ".join(str(r) for r in radelems)
    else:
        radelems_str = str(radelems)

    with open(filename, "w") as f:
        f.write("# required\n")
        f.write("rcutfac {} \n".format(rcutfac))
        f.write("twojmax {} \n".format(twojmax))
        f.write("# elements\n")
        f.write("nelems {} \n".format(len(elements)))
        f.write("elems {} \n".format(" ".join(elements)))
        f.write("radelems {} \n".format(radelems_str))
        f.write("welems {} \n".format(" ".join(["1"] * len(elements))))
        if chemflag:
            f.write("chemflag 1 \n")
        f.write("# optional\n")
        f.write("rfac0 0.99363\n")
        f.write("rmin0 0\n")
        f.write("bzeroflag {}\n".format(bzeroflag))


def generate_lammps_scripts(radii, descriptor_filename):
    """Generate LAMMPS input scripts for entropy-driven structure generation.

    Returns two scripts:
    - mliap_script: hybrid/overlay of soft repulsion + MLIAP entropy model
    - zero_script: soft repulsion only (for initial relaxation)

    Args:
        radii: Dict mapping type_id -> {'symbol': str, 'r_core': float, ...}
        descriptor_filename: Path to the MLIAP SNAP descriptor file.

    Returns:
        (mliap_script, zero_script) as newline-joined strings.
    """
    r_core_max = max(v['r_core'] for v in radii.values())

    # Entropy model + soft repulsion
    mliap_lines = []
    mliap_lines.append("neigh_modify one 10000")
    mliap_lines.append(
        "pair_style hybrid/overlay soft {} mliap model mliappy LATER "
        "descriptor sna {}".format(r_core_max, descriptor_filename))
    for c in itertools.combinations_with_replacement(sorted(radii.keys()), 2):
        r_sum = radii[c[0]]['r_core'] + radii[c[1]]['r_core']
        mliap_lines.append(
            "pair_coeff {} {} soft 10 {}".format(c[0], c[1], r_sum))
    elem_str = " ".join(radii[k]['symbol'] for k in sorted(radii.keys()))
    mliap_lines.append("pair_coeff * * mliap " + elem_str)
    mliap_lines.append("compute pe_peratom all pe/atom")
    mliap_script = "\n".join(mliap_lines)

    # Soft repulsion only
    zero_lines = []
    zero_lines.append("neigh_modify one 10000")
    zero_lines.append("pair_style soft {}".format(r_core_max))
    for c in itertools.combinations_with_replacement(sorted(radii.keys()), 2):
        r_sum = radii[c[0]]['r_core'] + radii[c[1]]['r_core']
        zero_lines.append(
            "pair_coeff {} {} 10 {}".format(c[0], c[1], r_sum))
    zero_script = "\n".join(zero_lines)

    return mliap_script, zero_script


def generate_binary_lammps_scripts(elements, descriptor_filename,
                                   core_radius_0, core_radius_1,
                                   core_radius_cross,
                                   min_dist_0, min_dist_1, min_dist_cross):
    """Generate LAMMPS scripts for binary element entropy structure generation.

    Uses the original binary_entropy parameterization:
    - Zero script: pair_style soft 5.0, A values 10/8/5 for 1-1/1-2/2-2,
      cutoffs are min_dist values (core_radius * 0.9).
    - Min script: pair_style hybrid/overlay soft 5, A=10 for all pairs,
      cutoffs are full core radii.

    Args:
        elements: List of two element symbols [elem0, elem1].
        descriptor_filename: Path to the MLIAP SNAP descriptor file.
        core_radius_0: Core radius for element 0.
        core_radius_1: Core radius for element 1.
        core_radius_cross: Core radius for cross-element interaction.
        min_dist_0: Minimum distance for elem0-elem0 (core_radius_0 * 0.9).
        min_dist_1: Minimum distance for elem1-elem1 (core_radius_1 * 0.9).
        min_dist_cross: Minimum distance for cross interaction.

    Returns:
        (mliap_script, zero_script) as newline-joined strings.
    """
    zero_script = (
        "pair_style soft 5.0\n"
        "pair_coeff 1 1 10 {}\n"
        "pair_coeff 1 2 8 {}\n"
        "pair_coeff 2 2 5 {}"
    ).format(min_dist_0, min_dist_cross, min_dist_1)

    mliap_script = (
        "pair_style hybrid/overlay soft 5 mliap model mliappy LATER "
        "descriptor sna {}\n"
        "pair_coeff 1 1 soft 10 {}\n"
        "pair_coeff 1 2 soft 10 {}\n"
        "pair_coeff 2 2 soft 10 {}\n"
        "pair_coeff * * mliap {} {}\n"
        "compute pe_peratom all pe/atom"
    ).format(descriptor_filename,
             core_radius_0, core_radius_cross, core_radius_1,
             elements[0], elements[1])

    return mliap_script, zero_script


def write_mliap_descriptor_multi(filename, radii, twojmax, bzeroflag=1):
    """Write MLIAP SNAP descriptor file for multi-element pseudo-species.

    Unlike write_mliap_descriptor, this uses rcutfac=1 and sets radelems
    to the per-species r_cut values (so effective cutoff = 1 * r_cut).

    Args:
        filename: Output file path.
        radii: Dict mapping type_id -> {'symbol': str, 'r_cut': float, ...}
        twojmax: Maximum angular momentum quantum number (2j).
        bzeroflag: Whether to include B0 components (0 or 1).
    """
    elements = [radii[k]['symbol'] for k in sorted(radii.keys())]
    radelems = [str(radii[k]['r_cut']) for k in sorted(radii.keys())]

    with open(filename, "w") as f:
        f.write("rcutfac 1 \n")
        f.write("twojmax {} \n".format(twojmax))
        f.write("nelems {} \n".format(len(elements)))
        f.write("elems {} \n".format(" ".join(elements)))
        f.write("radelems {} \n".format(" ".join(radelems)))
        f.write("welems {} \n".format(" ".join(["1"] * len(elements))))
        f.write("rfac0 0.99363 \n")
        f.write("rmin0 0 \n")
        f.write("bzeroflag {}\n".format(bzeroflag))
