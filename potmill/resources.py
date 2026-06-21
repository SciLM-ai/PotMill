"""Query the Flux allocation and map it to PotMill's per-stage executor worker counts."""

from dataclasses import dataclass


@dataclass
class Resources:
    nnodes: int
    ncores: int
    ngpus: int
    gpus_per_node: int
    device: str
    n_label_workers: int
    label_cores_per_job: int
    n_fit_workers: int
    fit_cores_per_job: int
    n_entropy_workers: int
    entropy_cores_per_job: int
    n_featurize_workers: int
    featurize_cores_per_job: int


def query_flux():
    """Return (resource_status, ncores, ngpus, nnodes) for the current Flux allocation."""
    import flux
    import flux.resource

    handle = flux.Flux()
    rs = flux.resource.status.ResourceStatusRPC(handle).get()
    rl = flux.resource.list.resource_list(handle).get()
    nnodes = len(list(rs.nodelist))
    print("NODELIST:", rs.nodelist, " #CORES:", rl.all.ncores, " #GPUS:", rl.all.ngpus, flush=True)
    return rs, rl.all.ncores, rl.all.ngpus, nnodes


# core(s)/node kept free for the dynamic `exe` (combine_b is on the critical path
# labeling->combine_b->featurize/fit; cost/pareto run there too). combine_b is chained (one at a
# time) so a single free core keeps it moving; freed VASP cores give cost/pareto transient room.
DYNAMIC_RESERVE_CORES = 1


def worker_layout(config, nnodes, ncores, ngpus):
    """Map the allocation to per-stage worker counts with one consistent scheme: each stage has
    <stage>_jobs_per_node concurrent jobs, each using <stage>_cores_per_job cores. [Main] device
    selects how labeling/fitting jobs are placed:

    * cuda: each labeling/fit job takes one GPU; (labeling+fit) jobs/node must fit in gpus/node.
    * cpu : each job takes cores_per_job cores; the per-node sum across all stages must leave
      DYNAMIC_RESERVE_CORES free for the dynamic exe (combine_b/cost/pareto).

    entropy + featurize are always CPU. entropy_* come from the raw [ourStructureGen] section."""
    device = config["Main"]["device"]
    gpus_per_node = ngpus // nnodes if nnodes else 0
    cores_per_node = ncores // nnodes if nnodes else 0

    entropy_jpn = config["ourStructureGen"].get("entropy_jobs_per_node", 1)
    entropy_cpj = config["ourStructureGen"].get("entropy_cores_per_job", 1)
    # Strict entropy increase is only well-defined for a SINGLE serial worker (parallel workers keep
    # eventually-consistent state). When requested, entropy runs as exactly one worker regardless of
    # entropy_jobs_per_node / nnodes; entropy_cores_per_job then gives that worker its threads.
    strict_entropy = config["ourStructureGen"].get("strict_entropy_decrease", 0)
    n_entropy_workers = 1 if strict_entropy else entropy_jpn * nnodes
    label_jpn = config["ourLabeling"]["labeling_jobs_per_node"]
    label_cpj = config["ourLabeling"]["labeling_cores_per_job"]
    feat_jpn = config["ourFeaturization"]["featurize_jobs_per_node"]
    feat_cpj = config["ourFeaturization"]["featurize_cores_per_job"]
    fit_jpn = config["ourFit"]["fit_jobs_per_node"]
    fit_cpj = config["ourFit"]["fit_cores_per_job"]

    if device == "cuda":
        assert label_jpn > 0 and fit_jpn > 0 and label_jpn + fit_jpn <= gpus_per_node, (
            f"cuda: labeling_jobs_per_node ({label_jpn}) + fit_jobs_per_node ({fit_jpn}) must be "
            f">0 and fit in gpus_per_node ({gpus_per_node})"
        )
    elif device == "cpu":
        # Each VASP labeling job is a 1-core Python worker plus label_cpj cores grabbed from the
        # flux instance by `flux run -n label_cpj` (the [Vasp] command). entropy/fit reserve their
        # cores as threads_per_core; featurize as an MPI job of feat_cpj cores.
        used = (
            entropy_jpn * entropy_cpj
            + label_jpn * (1 + label_cpj)
            + feat_jpn * feat_cpj
            + fit_jpn * fit_cpj
        )
        free = cores_per_node - used
        assert used <= cores_per_node - DYNAMIC_RESERVE_CORES, (
            f"cpu core budget/node exceeded: entropy {entropy_jpn}x{entropy_cpj} + labeling "
            f"{label_jpn}x(1+{label_cpj}) + featurize {feat_jpn}x{feat_cpj} + fit "
            f"{fit_jpn}x{fit_cpj} = {used} cores, but only {cores_per_node}/node available and >= "
            f"{DYNAMIC_RESERVE_CORES} must stay free for combine_b/cost/pareto"
        )
        print(
            f"CPU core budget/node: entropy {entropy_jpn}x{entropy_cpj} + labeling "
            f"{label_jpn}x(1+{label_cpj}) + featurize {feat_jpn}x{feat_cpj} + fit "
            f"{fit_jpn}x{fit_cpj} = {used}/{cores_per_node} cores, {free} free for the dynamic exe",
            flush=True,
        )
    else:
        raise ValueError(f"[Main] device must be 'cuda' or 'cpu', got {device!r}")

    return Resources(
        nnodes=nnodes,
        ncores=ncores,
        ngpus=ngpus,
        gpus_per_node=gpus_per_node,
        device=device,
        n_label_workers=label_jpn * nnodes,
        label_cores_per_job=label_cpj,
        n_fit_workers=fit_jpn * nnodes,
        fit_cores_per_job=fit_cpj,
        n_entropy_workers=n_entropy_workers,
        entropy_cores_per_job=entropy_cpj,
        n_featurize_workers=feat_jpn * nnodes,
        featurize_cores_per_job=feat_cpj,
    )
