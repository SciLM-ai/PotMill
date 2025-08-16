def max_entropy_atoms_iterator():

    from autopiad.binary_entropy.renorm import RandomEntropyInitializer
    from autopiad.binary_entropy.optimizer import EntropyMaximizer

    print("\n\n\n\n\n\n\JUSTSTARTED\n\n\JUSTSTARTED\n\n\JUSTSTARTED\n\n\JUSTSTARTED\n\n\JUSTSTARTED\n\n\JUSTSTARTED\n\n\n\n\n\n")

    rand_entropy = RandomEntropyInitializer()
    rand_entropy.looping()
    print("\n\n\n\n\n\n\IAMDONE\n\n\IAMDONE\n\n\IAMDONE\n\n\IAMDONE\n\n\IAMDONE\n\n\IAMDONE\n\n\n\n\n\n")

    entropy_maximizer = EntropyMaximizer()
    first_index = [0]
    for entropy_atoms in entropy_maximizer.loopiong():
        n_atoms = len(entropy_atoms)
        first_index.append(first_index[-1]+1+3*n_atoms)
        yield entropy_atoms#, first_index[-1]
