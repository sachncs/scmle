"""Iterative stratification of multi-label targets (Sechidis et al., 2011).

Plain stratified splitting looks at one label. With several labels, rare ones
can vanish from a fold. This procedure handles the rarest label first,
repeatedly giving each example to the subset that still wants most of that
label, so every subset keeps roughly the label proportions of the whole data.
"""

import numpy as np


def iterative_assignment(
    labels: np.ndarray, proportions: list[float], seed: int
) -> np.ndarray:
    """Assigns each row to one of several subsets.

    Args:
        labels: Binary matrix, one row per example and one column per label.
        proportions: Desired share of the rows for each subset (sums to 1).
        seed: Random seed for tie-breaking.

    Returns:
        The subset index of every row.
    """
    generator = np.random.default_rng(seed)
    matrix = np.asarray(labels).astype(int)
    rows = matrix.shape[0]
    shares = np.asarray(proportions, dtype=float)
    want_rows = shares * rows
    want_labels = shares[:, None] * matrix.sum(axis=0)[None, :]
    assigned = np.full(rows, -1)
    free = np.ones(rows, dtype=bool)
    while free.any():
        counts = matrix[free].sum(axis=0)
        open_labels = np.flatnonzero(counts > 0)
        if open_labels.size == 0:
            chosen = np.flatnonzero(free)
            for row in generator.permutation(chosen):
                subset = int(
                    np.argmax(want_rows + generator.random(len(shares)) * 1e-9)
                )
                assigned[row] = subset
                want_rows[subset] -= 1
            break
        label = open_labels[np.argmin(counts[open_labels])]
        candidates = np.flatnonzero(free & (matrix[:, label] == 1))
        for row in generator.permutation(candidates):
            need = want_labels[:, label]
            best = np.flatnonzero(need == need.max())
            if len(best) > 1:
                ties = want_rows[best]
                best = best[np.flatnonzero(ties == ties.max())]
            subset = int(generator.choice(best))
            assigned[row] = subset
            free[row] = False
            want_rows[subset] -= 1
            want_labels[subset] -= matrix[row]
    return assigned
