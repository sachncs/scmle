"""Keyed random streams for qualification code.

Each stream is derived from (run seed, component name, index) through
`numpy.random.SeedSequence`. The name is hashed with SHA-256, never with
Python's salted `hash`, so a stream is identical across processes and
interpreter builds. Because a stream depends only on its own key, adding or
reordering a check never changes another check's draws.
"""

import hashlib

import numpy as np

from scmle.contracts import qualification

NAME_WORDS = 4
UINT32_BYTES = 4


def validate_key(run_seed: int, component: str, index: int) -> str:
    """Checks a stream key and returns the trimmed component name.

    Args:
        run_seed: The run's seed; must be a non-negative integer (a bool is
            rejected).
        component: Name of the check or detector; surrounding whitespace is
            removed.
        index: Repeat counter; must be a non-negative integer.

    Returns:
        The component name with surrounding whitespace removed.

    Raises:
        ValueError: If the seed or index is not a non-negative integer or the
            component name is blank.
    """
    qualification.count(run_seed, "run_seed")
    qualification.count(index, "index")
    name = component.strip()
    if not name:
        raise ValueError("component must not be empty")
    return name


def name_words(component: str) -> tuple[int, ...]:
    """Returns stable 32-bit words derived from a component name.

    Args:
        component: The component name, hashed with SHA-256.

    Returns:
        The first four words of the digest, each read as a big-endian unsigned
        32-bit integer.
    """
    digest = hashlib.sha256(component.encode("utf-8")).digest()
    return tuple(
        int.from_bytes(digest[i : i + UINT32_BYTES], "big")
        for i in range(0, NAME_WORDS * UINT32_BYTES, UINT32_BYTES)
    )


def component_seed_sequence(
    run_seed: int, component: str, index: int = 0
) -> np.random.SeedSequence:
    """Returns the seed sequence for one component of one run.

    Args:
        run_seed: The run's seed.
        component: Stable name of the check or detector.
        index: Repeat counter for components that draw several streams.

    Returns:
        A sequence that depends only on the three arguments.
    """
    name = validate_key(run_seed, component, index)
    return np.random.SeedSequence(
        entropy=run_seed, spawn_key=(*name_words(name), index)
    )


def component_rng(
    run_seed: int, component: str, index: int = 0
) -> np.random.Generator:
    """Returns the random generator for one component of one run.

    Args:
        run_seed: The run's seed.
        component: Stable name of the check or detector.
        index: Repeat counter for components that draw several streams.

    Returns:
        A NumPy generator that depends only on the three arguments.
    """
    return np.random.default_rng(
        component_seed_sequence(run_seed, component, index)
    )


def derive_int_seed(run_seed: int, component: str, index: int = 0) -> int:
    """Returns a 32-bit integer seed for libraries that take `random_state`.

    Args:
        run_seed: The run's seed.
        component: Stable name of the check or detector.
        index: Repeat counter for components that draw several streams.

    Returns:
        An integer derived from the component's seed sequence, the same for
        the same three arguments.
    """
    state = component_seed_sequence(run_seed, component, index).generate_state(
        1
    )
    return int(state[0])


def seed_path(
    run_seed: int, component: str, index: int = 0
) -> qualification.SeedPath:
    """Returns the key to record on a `Check` or `Certificate`.

    Args:
        run_seed: The run's seed.
        component: Stable name of the check or detector.
        index: Repeat counter for components that draw several streams.

    Returns:
        The triple (run seed, trimmed component name, index) that identifies
        the random stream.
    """
    return (run_seed, validate_key(run_seed, component, index), index)
