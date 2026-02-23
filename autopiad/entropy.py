def max_entropy_atoms_iterator(structuregen_config):

    from autopiad.structuregen.renorm import RandomEntropyInitializer
    from autopiad.structuregen.optimizer import EntropyMaximizer
    import os

    os.makedirs("renorm_configs", exist_ok=True)
    os.makedirs("configs", exist_ok=True)

    rand_entropy = RandomEntropyInitializer(structuregen_config)
    rand_entropy.looping()

    entropy_maximizer = EntropyMaximizer(structuregen_config)
    first_index = [0]
    for entropy_atoms in entropy_maximizer.looping():
        n_atoms = len(entropy_atoms)
        first_index.append(first_index[-1] + 1 + 3 * n_atoms)
        yield entropy_atoms
