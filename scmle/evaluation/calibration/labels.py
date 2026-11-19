"""Calibration of ordinal levels and of multiple labels."""

from typing import Any

import numpy

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.evaluation.calibration import verdict


def ordinal(
    truth: Any, probabilities: Any, level: float = verdict.LEVEL
) -> verdict.Claim:
    """Tests cumulative probabilities at every threshold between levels.

    Args:
        truth: True level rank of each row, `0..levels-1`.
        probabilities: Predicted probability of each level, one column per
            level in order; rows sum to one.
        level: Family-wise size of the tests.

    Returns:
        The ordinal claim, one check per threshold `level <= k`.

    Raises:
        InputError: If a row does not sum to one, a rank is not a column, or
            the shapes disagree.
    """
    matrix = numpy.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or not numpy.allclose(matrix.sum(axis=1), 1.0):  # noqa: PLR2004 -- a matrix has two dimensions
        raise errors.InputError("level probabilities must sum to one per row")
    ranks = verdict.vector(truth, "levels", matrix.shape[0])
    if ((ranks < 0) | (ranks >= matrix.shape[1]) | (ranks % 1 != 0)).any():
        raise errors.InputError("every level must name a probability column")
    cumulative = numpy.clip(numpy.cumsum(matrix, axis=1), 0.0, 1.0)
    checks = tuple(
        verdict.bernoulli(f"level <= {k}", cumulative[:, k], ranks <= k)
        for k in range(matrix.shape[1] - 1)
    )
    return verdict.Claim(
        tasks.Task.ORDINAL,
        verdict.STATEMENTS[tasks.Task.ORDINAL],
        checks,
        (),
        level,
    )


def multilabel(
    truth: Any,
    probabilities: Any,
    names: tuple[str, ...] | None = None,
    level: float = verdict.LEVEL,
) -> verdict.Claim:
    """Tests the probability of every label.

    Args:
        truth: 0/1 matrix, one column per label.
        probabilities: Predicted probability matrix of the same shape.
        names: Label names in column order; positions when None.
        level: Family-wise size of the tests.

    Returns:
        The multilabel claim, one check per label.

    Raises:
        InputError: If the shapes disagree, a probability is outside
            `[0, 1]` or the names do not match the columns.
    """
    matrix = verdict.probable(
        numpy.asarray(probabilities, dtype=float), "probabilities"
    )
    marks = numpy.asarray(truth, dtype=float)
    if matrix.ndim != 2 or marks.shape != matrix.shape:  # noqa: PLR2004 -- a matrix has two dimensions
        raise errors.InputError("truth and probabilities must share a shape")
    titles = names or tuple(str(i) for i in range(matrix.shape[1]))
    if len(titles) != matrix.shape[1]:
        raise errors.InputError("give as many names as label columns")
    checks = tuple(
        verdict.bernoulli(f"label {title}", matrix[:, j], marks[:, j])
        for j, title in enumerate(titles)
    )
    return verdict.Claim(
        tasks.Task.MULTILABEL,
        verdict.STATEMENTS[tasks.Task.MULTILABEL],
        checks,
        (),
        level,
    )
