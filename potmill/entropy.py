def max_entropy_atoms_iterator(structuregen_config):

    import os
    import time

    # Set threading environment BEFORE importing LAMMPS/JAX/numpy.
    # LAMMPS SNAP bispectrum computation uses OpenMP, and JAX/MKL/OpenBLAS
    # also respect these variables. Must be set before library import.
    n_threads = str(structuregen_config.get('n_threads', 1))
    os.environ['OMP_NUM_THREADS'] = n_threads
    os.environ['MKL_NUM_THREADS'] = n_threads
    os.environ['OPENBLAS_NUM_THREADS'] = n_threads

    # Configure JAX for CPU with 64-bit precision
    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")

    from potmill.structuregen.renorm import RandomEntropyInitializer
    from potmill.structuregen.optimizer import EntropyMaximizer

    os.makedirs("renorm_configs", exist_ok=True)
    os.makedirs("configs", exist_ok=True)

    worker_id = structuregen_config.get('_worker_id', 0)
    shared_dir = structuregen_config.get('shared_state_dir', None)

    if shared_dir:
        phase1_signal = os.path.join(shared_dir, "phase1_done")
        if worker_id == 0:
            # Worker 0 runs Phase 1 and shares results
            rand_entropy = RandomEntropyInitializer(structuregen_config)
            rand_entropy.looping()
            import shutil
            for fname in ["renormalization_matrix.pckl", "random-manager.p"]:
                shutil.copy(fname, shared_dir)
            # Flush filesystem buffers before creating signal file.
            # On networked filesystems (Lustre), metadata can propagate
            # faster than data, so other workers might see the signal
            # before data files are fully visible without this.
            fd = os.open(shared_dir, os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)
            open(phase1_signal, 'w').close()
        else:
            # Other workers wait for Phase 1 results
            while not os.path.exists(phase1_signal):
                time.sleep(0.5)
            import shutil
            for fname in ["renormalization_matrix.pckl", "random-manager.p"]:
                shutil.copy(os.path.join(shared_dir, fname), ".")
    else:
        rand_entropy = RandomEntropyInitializer(structuregen_config)
        rand_entropy.looping()

    entropy_maximizer = EntropyMaximizer(structuregen_config)
    first_index = [0]
    for entropy_atoms in entropy_maximizer.looping():
        n_atoms = len(entropy_atoms)
        first_index.append(first_index[-1] + 1 + 3 * n_atoms)
        yield entropy_atoms
