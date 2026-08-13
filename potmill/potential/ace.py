"""Turn a fitted ACE coefficient vector into a LAMMPS-ready ``.yace`` potential.

The column layout of a PotMill beta is fixed by ``featurize``/``_feature_indices``: per element, a
constant column followed by that element's selected descriptor columns::

    beta = [E0_el0, B_el0..., E0_el1, B_el1..., ...]

which is exactly what a LAMMPS ACE potential wants -- ``E0`` is the per-element reference energy in
the ``.yace`` header and the rest are the basis-function coefficients.

The swept ``(nmax, lmax)`` of a hyperparameter point selects a SUBSET of the full swept basis's
columns, and that subset is itself a valid ACE basis (verified: the column-filtered full label list
is set- AND order-identical to the labels a basis generated directly at the subset's nmax/lmax
produces).  So the written potential carries only the functions that point actually uses -- a
minimal basis, i.e. the fastest MD -- not a full-basis potential padded with zeros.

Coefficients are attached to basis functions BY SYMBOLIC LABEL (``nu``), never by array position,
and the two label sets are asserted equal element by element first.
"""

import itertools

from potmill.fitting.fit import _feature_indices
from potmill.potential.labels import (
    _ace_section_params,
    ace_labels,
    check_against_feature_names,
    feature_names_from_blist,
)

_CCS_CACHE = {}


def _coupling(ranks, lmax, wigner_flag):
    """Generalized coupling coefficients for this (ranks, lmax), cached per process.

    Mirrors ``Ace._write_couple`` but does NOT write FitSNAP's ``*.pickle`` cache files into the
    caller's working directory (the run dirs stay clean; regenerating costs ~0.1 s).
    """
    key = (tuple(ranks), tuple(lmax), bool(wigner_flag))
    if key not in _CCS_CACHE:
        ldict = {int(r): int(ll) for r, ll in zip(ranks, lmax, strict=True)}
        if wigner_flag:
            from fitsnap3lib.lib.sym_ACE.wigner_couple import get_wig_coupling

            ccs = get_wig_coupling(ldict, 0)
        else:
            from fitsnap3lib.lib.sym_ACE.clebsch_couple import get_cg_coupling

            ccs = get_cg_coupling(ldict, L_R=0)
        _CCS_CACHE[key] = ccs[0]  # M_R = 0, as FitSNAP does
    return dict(_CCS_CACHE[key])  # AcePot mutates its ccs argument -- hand it a shallow copy


def _per_bond(values, numtypes, what):
    """Expand a FitSNAP.in per-bond list to one value per (i, j) bond, the way featurize expands
    the swept rcut: a single value applies to every bond, otherwise there must be numtypes^2."""
    nbonds = numtypes**2
    if len(values) == 1:
        return list(values) * nbonds
    if len(values) != nbonds:
        raise ValueError(
            f"FitSNAP.in [ACE] {what} has {len(values)} values but there are {nbonds} bond types "
            f"for {numtypes} elements (stop, do not guess)"
        )
    return list(values)


def split_beta(beta, selected_labels, selected_nus, numtypes):
    """Split a fitted beta into ``(E0 per element, {mu0: {nu: coefficient}})``.

    Walks the selected columns in order: each constant label (``[0]``) opens that element's block.
    The element index implied by the walk is cross-checked against the ``mu0`` carried inside every
    descriptor label, so a mis-ordered or mis-filtered column set cannot pass silently.
    """
    if len(beta) != len(selected_labels):
        raise ValueError(
            f"beta length {len(beta)} != selected columns {len(selected_labels)} (stop)"
        )
    e0 = []
    betas_dict = {}
    ielem = -1
    block_sizes = []
    for coeff, label, nu in zip(beta, selected_labels, selected_nus, strict=True):
        if len(label) == 1:  # the [[0]] constant column that opens an element block
            ielem += 1
            e0.append(float(coeff))
            betas_dict[ielem] = {}
            block_sizes.append(0)
            continue
        if ielem < 0:
            raise ValueError("selected columns start with a descriptor, not a constant (stop)")
        mu0 = int(label[1])
        if mu0 != ielem:
            raise ValueError(
                f"column block {ielem} carries a descriptor with mu0={mu0} ({nu}) -- element "
                f"blocks and descriptor labels disagree (stop, do not guess)"
            )
        betas_dict[ielem][nu] = float(coeff)
        block_sizes[-1] += 1
    if len(e0) != numtypes:
        raise ValueError(
            f"found {len(e0)} constant columns but [ACE] has {numtypes} element types (stop)"
        )
    if len(set(block_sizes)) != 1:
        raise ValueError(f"unequal per-element coefficient blocks {block_sizes} (stop)")
    return e0, betas_dict


def ace_coefficients(ace_section, rcuts, full_nmax, full_lmax, nmax, lmax, beta, feature_names):
    """Map a fitted beta onto ACE basis functions.

    Regenerates the FULL swept basis, asserts it against the ``feature_names`` featurization
    actually produced, applies the pipeline's own ``_feature_indices`` column filter for this
    hyperparameter point, and returns ``(E0 list, {mu0: {nu: coefficient}}, selected nu labels)``.
    """
    params = _ace_section_params(ace_section)
    numtypes = params["numtypes"]

    full_blist, full_nus = ace_labels(ace_section, full_nmax, full_lmax)
    check_against_feature_names(full_blist, numtypes, feature_names)

    full_names = feature_names_from_blist(full_blist, numtypes)
    ncoeff = len(full_blist) // numtypes
    padded_nus = []
    for ielem in range(numtypes):
        padded_nus += [None] + full_nus[ielem * ncoeff : (ielem + 1) * ncoeff]

    indices = _feature_indices("ACE", full_names, [list(rcuts), list(nmax), list(lmax)])
    selected_labels = [full_names[i] for i in indices]
    selected_nus = [padded_nus[i] for i in indices]
    e0, betas_dict = split_beta(beta, selected_labels, selected_nus, numtypes)
    return e0, betas_dict, [nu for nu in selected_nus if nu is not None]


def build_acepot(ace_section, rcuts, nmax, lmax, e0, betas_dict, selected_nus):
    """An ``AcePot`` for this hyperparameter point, carrying the fitted coefficients.

    The basis is generated at the point's OWN ``nmax``/``lmax`` (minimal basis -> fastest MD), and
    its label set is asserted equal, per element, to the labels the column filter selected -- so
    the potential evaluates exactly the functions the fit used, no more and no fewer.
    """
    from fitsnap3lib.lib.sym_ACE.pa_gen import get_mu_n_l
    from fitsnap3lib.lib.sym_ACE.yamlpace_tools.potential import AcePot

    params = _ace_section_params(ace_section)
    numtypes = params["numtypes"]
    bondstrs = ["[%d, %d]" % b for b in itertools.product(range(numtypes), range(numtypes))]

    rcvals = _per_bond([float(r) for r in rcuts], numtypes, "rcutfac (swept)")
    lmbdavals = _per_bond(params["lmbda"], numtypes, "lambda")
    rcinnervals = _per_bond(params["rcinner"], numtypes, "rcinner")
    drcinnervals = _per_bond(params["drcinner"], numtypes, "drcinner")
    if len(bondstrs) > 1:  # AcePot takes per-bond dicts unless there is a single bond type
        rcvals = dict(zip(bondstrs, rcvals, strict=True))
        lmbdavals = dict(zip(bondstrs, lmbdavals, strict=True))
        rcinnervals = dict(zip(bondstrs, rcinnervals, strict=True))
        drcinnervals = dict(zip(bondstrs, drcinnervals, strict=True))

    apot = AcePot(
        params["types"],
        [0.0] * numtypes,
        params["ranks"],
        [int(v) for v in nmax],
        [int(v) for v in lmax],
        params["nmaxbase"],
        rcvals,
        lmbdavals,
        rcinnervals,
        drcinnervals,
        params["lmin"],
        params["b_basis"],
        **{"ccs": _coupling(params["ranks"], lmax, params["wigner_flag"])},
    )

    basis_by_elem = {i: set() for i in range(numtypes)}
    for nu in apot.nus:
        basis_by_elem[get_mu_n_l(nu)[0]].add(nu)
    for ielem in range(numtypes):
        fitted = set(betas_dict[ielem])
        if fitted != basis_by_elem[ielem]:
            raise ValueError(
                f"element {params['types'][ielem]}: {len(fitted)} fitted descriptor labels but "
                f"{len(basis_by_elem[ielem])} basis functions at nmax={list(nmax)} lmax={list(lmax)} "
                f"({len(fitted - basis_by_elem[ielem])} fitted-only, "
                f"{len(basis_by_elem[ielem] - fitted)} basis-only) -- the selected columns are not "
                f"this basis (stop, do not guess)"
            )
    if len(selected_nus) != len(apot.nus):
        raise ValueError(
            f"{len(selected_nus)} selected descriptor columns != {len(apot.nus)} basis functions (stop)"
        )

    apot.set_betas(betas_dict)  # dict form: keyed by label, never by position
    apot.E0 = [float(v) for v in e0]
    apot.set_funcs(nulst=apot.nus)
    return apot


def _rewrite_e0_full_precision(yace_path, e0):
    """Rewrite the ``E0:`` line at full float precision.

    ``AcePot.write_pot`` formats the per-element reference energies with ``'%f'`` -- six decimals.
    Every other number in the file goes through ``json.dumps`` (full precision), so E0 is the one
    place a potential silently stops reproducing the fit: rounding E0 shifts the predicted energy by
    up to ~5e-7 eV/atom (measured), while leaving forces exact because E0 is a constant. The values
    are already in hand, so write them exactly instead of accepting the loss.
    """
    with open(yace_path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith("E0:"):
            lines[i] = "E0: [%s] \n" % ", ".join(repr(float(v)) for v in e0)
            break
    else:
        raise ValueError(f"{yace_path} has no E0 line to rewrite (stop)")
    with open(yace_path, "w") as f:
        f.writelines(lines)


def write_yace(out_base, ace_section, rcuts, full_nmax, full_lmax, nmax, lmax, beta, feature_names):
    """Write ``<out_base>.yace`` for one fitted hyperparameter point. Returns its path."""
    e0, betas_dict, selected_nus = ace_coefficients(
        ace_section, rcuts, full_nmax, full_lmax, nmax, lmax, beta, feature_names
    )
    apot = build_acepot(ace_section, rcuts, nmax, lmax, e0, betas_dict, selected_nus)
    apot.write_pot(out_base)
    _rewrite_e0_full_precision(out_base + ".yace", e0)
    return out_base + ".yace"
