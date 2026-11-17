"""Near-duplicate grouping of image samples."""

from collections.abc import Mapping

from scmle.contracts import errors
from scmle.data.modalities.image.sample import Sample


def islands(neighbors: Mapping[int, set[int]]) -> list[set[int]]:
    """Finds the connected groups of a neighbour graph.

    Args:
        neighbors: For each node ``0..n-1``, the nodes it neighbours.

    Returns:
        The connected components that hold at least two nodes, in order of
        their smallest node.
    """
    components = []
    visited: set[int] = set()
    for start in range(len(neighbors)):
        if start in visited or not neighbors[start]:
            continue
        pending = [start]
        component: set[int] = set()
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            visited.add(current)
            pending.extend(neighbors[current] - component)
        components.append(component)
    return components


class Duplicates:
    """Near-duplicate review of an image collection.

    Attributes:
        samples: The samples, in caller order.
    """

    __slots__ = ()

    samples: tuple[Sample, ...]

    def near_duplicate_groups(
        self,
        max_samples: int = 1024,
        hash_size: int = 8,
        max_hamming_distance: int = 8,
    ) -> tuple[tuple[str, ...], ...]:
        """Returns bounded groups of visually similar, distinct images.

        The signature is a quantized, resized luminance representation. It is
        review evidence for likely near duplicates, not a semantic similarity
        model or a proof that two images depict the same object.

        Args:
            max_samples: Number of leading samples considered.
            hash_size: Side of the luminance thumbnail used for the
                signature.
            max_hamming_distance: Largest number of differing signature bits
                for two images to count as neighbours.

        Returns:
            Connected groups of neighbouring samples that contain more than
            one distinct content hash. Each group is a sorted tuple of
            sources; the groups are sorted.

        Raises:
            ValueError: If ``max_samples`` is not a positive integer or
                ``max_hamming_distance`` is not a non-negative integer. An
                invalid ``hash_size`` also raises ValueError, from
                `Sample.fingerprint`.
        """
        samples_limit: object = max_samples
        if not isinstance(samples_limit, int) or isinstance(
            samples_limit, bool
        ):
            raise errors.InputError("max_samples must be a positive integer")
        if max_samples <= 0:
            raise errors.InputError("max_samples must be a positive integer")
        distance_limit: object = max_hamming_distance
        if (
            not isinstance(distance_limit, int)
            or isinstance(distance_limit, bool)
            or distance_limit < 0
        ):
            raise errors.InputError(
                "max_hamming_distance must be a non-negative integer"
            )
        candidates = self.samples[:max_samples]
        signatures = [
            int(sample.fingerprint(hash_size), 16) for sample in candidates
        ]
        neighbors: dict[int, set[int]] = {
            index: set() for index in range(len(candidates))
        }
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                if (
                    signatures[left] ^ signatures[right]
                ).bit_count() <= max_hamming_distance:
                    neighbors[left].add(right)
                    neighbors[right].add(left)
        groups = [
            tuple(sorted(candidates[index].source for index in component))
            for component in islands(neighbors)
            if len({candidates[index].content_hash for index in component}) > 1
        ]
        return tuple(sorted(groups))
