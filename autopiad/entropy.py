def max_entropy_atoms_iterator():

    from autopiad.binary_entropy.renorm import RandomEntropyInitializer
    from autopiad.binary_entropy.optimizer import EntropyMaximizer

    rand_entropy = RandomEntropyInitializer()
    rand_entropy.looping()

    entropy_maximizer = EntropyMaximizer()
    first_index = [0]
    for entropy_atoms in entropy_maximizer.loopiong():
        n_atoms = len(entropy_atoms)
        first_index.append(first_index[-1]+1+3*n_atoms)
        yield entropy_atoms, first_index[-1]
