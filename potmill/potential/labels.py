"""ACE descriptor-label reconstruction (the bridge between a fitted beta and a LAMMPS .yace).

``featurize`` returns only ``blist`` -- the NUMERIC descriptor labels (``[i, mu0, mu..., n..., l...]``)
that name the design-matrix columns.  Writing a ``.yace`` needs the SYMBOLIC ``nu`` labels that
``AcePot`` keys its basis functions by, and ``featurize`` never returns those.  So this module
regenerates BOTH, in FitSNAP's exact ordering, from the very same ``fitsnap3lib`` primitives
``Ace._generate_b_list`` uses.

The reconstruction is never trusted on faith: every caller asserts the regenerated ``blist``
against the ``feature_names`` featurization actually returned (``check_against_feature_names``),
so a FitSNAP version bump that reorders the basis fails loudly instead of writing a potential
whose coefficients are silently attached to the wrong functions.
"""


def _ace_section_params(ace_section):
    """Pull the basis-defining keys out of a FitSNAP.in [ACE] dict, applying FitSNAP's own
    defaults and its ``mumax = len(types)`` override. Raises on options this writer has not
    been validated for, rather than emitting a potential that may not match the fit."""
    types = str(ace_section.get("type", "H")).split()
    numtypes = int(ace_section.get("numTypes", 1))
    if numtypes != len(types):
        raise ValueError(
            f"FitSNAP.in [ACE] numTypes ({numtypes}) != number of type names ({len(types)}: "
            f"{types}) (stop, do not guess)"
        )
    ranks = [int(r) for r in str(ace_section.get("ranks", "3")).split()]
    lmin = [int(v) for v in str(ace_section.get("lmin", "0")).split()]
    if len(lmin) == 1:
        lmin = lmin * len(ranks)

    b_basis = str(ace_section.get("b_basis", "pa_tabulated"))
    if b_basis != "pa_tabulated":
        raise NotImplementedError(
            f"FitSNAP.in [ACE] b_basis = '{b_basis}': the LAMMPS potential writer is only "
            f"validated for 'pa_tabulated' (stop, do not guess)"
        )
    if str(ace_section.get("manuallabs", "None")) != "None":
        raise NotImplementedError(
            "FitSNAP.in [ACE] manuallabs is set: the LAMMPS potential writer is only validated "
            "for the tabulated permutation-adapted basis (stop, do not guess)"
        )
    # bzeroflag=1 removes the per-element constant column from FitSNAP's design matrix, but
    # potmill.featurization.featurize unconditionally prepends a [[0]] label per element -- so a
    # bzeroflag=1 run has more labels than design-matrix columns and the fit itself is already
    # inconsistent. Refuse rather than paper over it.
    if int(ace_section.get("bzeroflag", 0)):
        raise NotImplementedError(
            "FitSNAP.in [ACE] bzeroflag = 1 is not supported by PotMill (featurize always adds a "
            "per-element constant column, which bzeroflag=1 removes from the design matrix) (stop)"
        )
    return {
        "types": types,
        "numtypes": numtypes,
        "ranks": ranks,
        "lmin": lmin,
        "mumax": len(types),  # FitSNAP overrides the mumax key with len(types)
        "b_basis": b_basis,
        "nmaxbase": int(ace_section.get("nmaxbase", 16)),
        "wigner_flag": bool(int(ace_section.get("wigner_flag", 1))),
        "lmbda": [float(v) for v in str(ace_section.get("lambda", "1.35")).split()],
        "rcinner": [float(v) for v in str(ace_section.get("rcinner", "0.0")).split()],
        "drcinner": [float(v) for v in str(ace_section.get("drcinner", "0.01")).split()],
    }


def ace_labels(ace_section, nmax, lmax):
    """``(blist, nus)`` for an ACE basis with these ``nmax``/``lmax``, in FitSNAP's order.

    A verbatim re-tracing of ``fitsnap3lib.io.sections.calculator_sections.ace.Ace._generate_b_list``
    (same primitives, same six stable sorts, same per-element regrouping), so ``blist[i]`` names the
    same design-matrix column FitSNAP would have produced and ``nus[i]`` is that column's symbolic
    basis-function label.
    """
    from fitsnap3lib.lib.sym_ACE.pa_gen import get_mu_n_l, pa_labels_raw, srt_by_attyp

    p = _ace_section_params(ace_section)
    if not (len(nmax) == len(lmax) == len(p["ranks"])):
        raise ValueError(
            f"nmax ({nmax}), lmax ({lmax}) and [ACE] ranks ({p['ranks']}) must have the same "
            f"length (stop, do not guess)"
        )

    ranked = [
        pa_labels_raw(rank, int(nmax[i]), int(lmax[i]), p["mumax"], lmin=p["lmin"][i])[0]
        for i, rank in enumerate(p["ranks"])
    ]
    nus_unsort = [item for sublist in ranked for item in sublist]
    nus = nus_unsort.copy()
    mu0s, mus, ns, ls = [], [], [], []
    for nu in nus_unsort:
        mu0ii, muii, nii, lii = get_mu_n_l(nu)
        mu0s.append(mu0ii)
        mus.append(tuple(muii))
        ns.append(tuple(nii))
        ls.append(tuple(lii))
    nus.sort(key=lambda x: mus[nus_unsort.index(x)])
    nus.sort(key=lambda x: ns[nus_unsort.index(x)])
    nus.sort(key=lambda x: ls[nus_unsort.index(x)])
    nus.sort(key=lambda x: mu0s[nus_unsort.index(x)])
    nus.sort(key=len)
    nus.sort(key=lambda x: mu0s[nus_unsort.index(x)])
    byattyp = srt_by_attyp(nus)

    blist, out_nus = [], []
    i = 0
    for atype in range(p["numtypes"]):
        for nu in byattyp[str(atype)]:
            i += 1
            mu0, mu, n, ll, L = get_mu_n_l(nu, return_L=True)
            flat_nu = [mu0] + mu + n + ll + (list(L) if L is not None else [])
            blist.append([i] + flat_nu)
            out_nus.append(nu)
    return blist, out_nus


def feature_names_from_blist(blist, numtypes):
    """The ``bnames`` structure ``featurize`` returns: per element, a ``[0]`` constant-column label
    followed by that element's ``blist`` block."""
    if len(blist) % numtypes:
        raise ValueError(
            f"blist length {len(blist)} is not divisible by numtypes {numtypes} (stop)"
        )
    ncoeff = len(blist) // numtypes
    names = []
    for ielem in range(numtypes):
        names += [[0]] + blist[ielem * ncoeff : (ielem + 1) * ncoeff]
    return names


def check_against_feature_names(blist, numtypes, feature_names):
    """Assert the regenerated labels ARE the ones featurization produced (ground truth).

    ``feature_names`` is whatever ``featurize`` returned for the full swept basis. Any mismatch
    means the reconstruction and the design matrix disagree about what a column is, which would
    attach fitted coefficients to the wrong basis functions -- so this raises instead of warning.
    """
    if feature_names and isinstance(feature_names[0][0], list):
        feature_names = feature_names[0]  # unwrap the [bnames] nesting featurize can return
    rebuilt = feature_names_from_blist(blist, numtypes)
    if len(rebuilt) != len(feature_names):
        raise ValueError(
            f"regenerated ACE labels ({len(rebuilt)}) != featurized feature_names "
            f"({len(feature_names)}) -- basis reconstruction disagrees with the design matrix (stop)"
        )
    for i, (a, b) in enumerate(zip(rebuilt, feature_names, strict=True)):
        if [int(v) for v in a] != [int(v) for v in b]:
            raise ValueError(
                f"regenerated ACE label {i} = {a} != featurized {b} -- basis reconstruction "
                f"disagrees with the design matrix (stop, do not guess)"
            )
    return True
