"""The ``.mod`` include file: how LAMMPS should be told to use an exported potential.

A user's input script needs only ``include <name>.mod`` after ``units``/``atom_style``/``read_data``.

Two things this file has to get right, neither of them cosmetic:

**The evaluator.** ``pair_style pace`` has two interchangeable algorithms for the same basis --
``product`` and ``recursive`` -- which give identical energies and forces (measured to <1e-15
eV/atom for every basis shape PotMill can sweep). ``recursive`` is ~18% faster on CPU at a
production basis (1254 columns: 86.7 vs 106.0 us/atom/step), but LAMMPS's KOKKOS implementation
REFUSES it outright::

    src/KOKKOS/pair_pace_kokkos.cpp:570
    if (recursive) error->all(FLERR,"Must use 'product' algorithm with pair pace/kk on the GPU");

That check sits in ``PairPACEKokkos::compute()`` with no execution-space branch, and ``pace/kk/host``
is a registered style too -- so the rule is "KOKKOS implies product", not merely "GPU implies
product". Rather than make the user choose (and abort their first GPU run), the ``.mod`` picks at
runtime with LAMMPS's own ``is_active(package,kokkos)`` feature function: product under KOKKOS,
recursive otherwise. Same numbers either way; the user chooses nothing.

**The reference potential.** Featurization fits ``E - E_ref``, so whatever reference the FitSNAP
``[REFERENCE]`` section subtracted must be added back in LAMMPS or the exported potential is quietly
wrong. ``pair_style zero <cut>`` (every PotMill example) subtracts nothing; a ``hybrid``/
``hybrid/overlay`` declaration is passed through the way FitSNAP's own writer does; anything else
stops with an error naming the file.
"""

ACE_STYLE = "pace"
KOKKOS_TEST = '"$(is_active(package,kokkos))"'


def lmp_pairdecl(reference_section):
    """FitSNAP's ``lmp_pairdecl`` rebuilt from a parsed ``[REFERENCE]`` dict: the pair_style line
    followed by every ``pair_coeff*`` line (FitSNAP accepts pair_coeff1, pair_coeff2, ... keys)."""
    decl = ["pair_style " + str(reference_section.get("pair_style", "zero 10.0")).strip()]
    for key, value in reference_section.items():
        if key.startswith("pair_coeff"):
            decl.append("pair_coeff " + str(value).strip())
    return decl


def _strip_zero(pair_style_line):
    """Drop ``zero <cutoff>`` from a hybrid pair_style declaration (its energy is identically 0)."""
    tokens = pair_style_line.split()
    if "zero" in tokens:
        i = tokens.index("zero")
        del tokens[i]
        if i < len(tokens):
            del tokens[i]  # the zero style's cutoff argument
    return " ".join(tokens)


def _style_lines(prefix, evaluator):
    """The ``pair_style`` command(s). ``evaluator=None`` emits the runtime KOKKOS branch; an explicit
    'product'/'recursive' emits one plain line (used by the tests, which check both algorithms)."""
    if evaluator is not None:
        return [f"{prefix} {evaluator}"]
    return [
        f"if {KOKKOS_TEST} then &",
        f'   "{prefix} product" &',
        "else &",
        f'   "{prefix} recursive"',
    ]


def pair_commands(potential_filename, elements, reference_section, evaluator=None, source=""):
    """The ``pair_style``/``pair_coeff`` lines for an exported ACE potential, as a list."""
    decl = lmp_pairdecl(reference_section)
    style_line = decl[0]
    style = style_line.split(maxsplit=1)[1] if len(style_line.split()) > 1 else ""
    elem_map = " ".join(elements)

    if "hybrid" in style:
        lines = _style_lines(f"{_strip_zero(style_line)} {ACE_STYLE}", evaluator)
        lines += [c for c in decl[1:] if "zero" not in c]
        lines.append(f"pair_coeff * * {ACE_STYLE} {potential_filename} {elem_map}")
        return lines
    if style.split()[:1] == ["zero"]:
        return _style_lines(f"pair_style {ACE_STYLE}", evaluator) + [
            f"pair_coeff * * {potential_filename} {elem_map}"
        ]
    raise ValueError(
        f"[REFERENCE] pair_style = '{style}' in {source or 'FitSNAP.in'} is neither 'zero <cut>' "
        f"nor a 'hybrid' declaration. Featurization fitted E - E_ref with this reference, so the "
        f"LAMMPS potential must reproduce it; the writer will not guess how (stop)"
    )


def write_mod(
    path, potential_filename, elements, reference_section, evaluator=None, meta=None, source=""
):
    """Write the ``.mod`` include file. ``meta`` lines are added as comments (provenance)."""
    units = str(reference_section.get("units", "metal")).strip()
    atom_style = str(reference_section.get("atom_style", "atomic")).strip()
    lines = [
        "# LAMMPS potential written by PotMill.",
        f"# Requires:  units {units}   /   atom_style {atom_style}",
        "# Usage:     include this file after read_data. Runs as-is on CPU, and on GPU with",
        f"#            KOKKOS (-k on g <ngpu> -sf kk), which selects {ACE_STYLE}/kk automatically.",
    ]
    for line in meta or []:
        lines.append(f"# {line}")
    if evaluator is None:
        lines += [
            "#",
            "# The two pace evaluators give identical energies and forces; 'recursive' is ~18%",
            "# faster on CPU but ABORTS under KOKKOS (src/KOKKOS/pair_pace_kokkos.cpp:570), so the",
            "# line below picks the right one at runtime. Replace it with a plain",
            "# 'pair_style pace product' if your LAMMPS predates is_active().",
        ]
    lines.append("")
    lines += pair_commands(
        potential_filename, elements, reference_section, evaluator=evaluator, source=source
    )
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path
