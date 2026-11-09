"""Separate roles for training and serving.

A role lists the object-store prefixes it may read and write. Training may
read data and write its own run records, checkpoints and artifacts; serving
may read released bundles and write only its audit trail. Neither can do what
the other does: training cannot write `bundles/` (so it cannot activate a
model) and serving cannot write `runs/` (so it cannot alter a training
record). Releasing a bundle is a third act that needs an approval, outside
both roles.

`RoleStore` enforces a role in process; on shared infrastructure the same
table is the policy of two distinct identities (`check_distinct` refuses to
run a job with the serving identity).
"""

import dataclasses

from scmle.contracts import errors
from scmle.data.store import objectstore


class RoleDeniedError(errors.ScmleError):
    """A role tried to touch a prefix it was not given."""

    category = "role_denied"


@dataclasses.dataclass(frozen=True, slots=True)
class Role:
    """What one identity may do.

    Attributes:
        name: Role name.
        reads: Key prefixes it may read, list and head.
        writes: Key prefixes it may write and delete.
    """

    name: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]

    def allows_read(self, key: str) -> bool:
        """Reports whether `key` may be read.

        Args:
            key: Object key.

        Returns:
            True if the key starts with one of the readable prefixes.
        """
        return key.startswith(self.reads)

    def allows_write(self, key: str) -> bool:
        """Reports whether `key` may be written.

        Args:
            key: Object key.

        Returns:
            True if the key starts with one of the writable prefixes.
        """
        return key.startswith(self.writes)


TRAINING_ROLE = Role(
    "training",
    reads=("data/", "runs/", "artifacts/"),
    writes=("runs/", "artifacts/", "evidence/"),
)
SERVING_ROLE = Role(
    "serving",
    reads=("bundles/", "artifacts/"),
    writes=("audit/",),
)


def check_separation(training: Role, serving: Role) -> None:
    """Refuses roles that overlap where they must not.

    Args:
        training: The training role.
        serving: The serving role.

    Raises:
        ConfigError: If the two share a name, a write prefix, or if training can
            write `bundles/` or serving can write `runs/`.
    """
    if training.name == serving.name:
        raise errors.ConfigError("training and serving need distinct roles")
    for mine in training.writes:
        for theirs in serving.writes:
            if mine.startswith(theirs) or theirs.startswith(mine):
                raise errors.ConfigError(
                    f"training and serving both write {mine!r}/{theirs!r}"
                )
    if any(p.startswith("bundles/") for p in training.writes):
        raise errors.ConfigError("training must not write released bundles")
    if any(p.startswith("runs/") for p in serving.writes):
        raise errors.ConfigError("serving must not write training records")


def check_distinct(training_identity: str, serving_identity: str) -> None:
    """Refuses to run training as the serving identity.

    Args:
        training_identity: Identity the training job runs as.
        serving_identity: Identity the serving side uses.

    Raises:
        ConfigError: If the identities are equal or empty.
    """
    if not training_identity or not serving_identity:
        raise errors.ConfigError("both identities must be named")
    if training_identity == serving_identity:
        raise errors.ConfigError(
            "a training job must not run as the serving identity"
        )


class RoleStore(objectstore.ObjectStore):
    """A store that refuses what its role does not allow.

    Attributes:
        inner: The real store.
        role: The role enforced.
    """

    def __init__(self, inner: objectstore.ObjectStore, role: Role) -> None:
        """Initializes the wrapper.

        Args:
            inner: The real store.
            role: The role to enforce.
        """
        super().__init__(inner.location)
        self.scheme = inner.scheme
        self.inner = inner
        self.role = role

    def need_read(self, key: str) -> None:
        """Raises unless the role may read `key`.

        Args:
            key: Object key or prefix.

        Raises:
            RoleDeniedError: If the role may not read it.
        """
        if not self.role.allows_read(key):
            raise RoleDeniedError(f"role {self.role.name} may not read {key}")

    def need_write(self, key: str) -> None:
        """Raises unless the role may write `key`.

        Args:
            key: Object key.

        Raises:
            RoleDeniedError: If the role may not write it.
        """
        if not self.role.allows_write(key):
            raise RoleDeniedError(f"role {self.role.name} may not write {key}")

    def put(
        self, key: str, data: bytes, *, if_absent: bool = False
    ) -> objectstore.ObjectMeta:
        """Stores `data` if the role may write `key` (see `ObjectStore.put`).

        Args:
            key: The object key.
            data: The content to store.
            if_absent: Create the object only if none exists.

        Returns:
            The metadata of the stored object.
        """
        self.need_write(key)
        return self.inner.put(key, data, if_absent=if_absent)

    def get(self, key: str) -> tuple[bytes, objectstore.ObjectMeta]:
        """Reads an object if the role may read it (see `ObjectStore.get`).

        Args:
            key: The object key.

        Returns:
            The content and the object's metadata.
        """
        self.need_read(key)
        return self.inner.get(key)

    def head(self, key: str) -> objectstore.ObjectMeta:
        """Reads metadata if the role may read the key (see `ObjectStore.head`).

        Args:
            key: The object key.

        Returns:
            The object's metadata.
        """
        self.need_read(key)
        return self.inner.head(key)

    def list_objects(self, prefix: str = "") -> list[objectstore.ObjectMeta]:
        """Lists objects if the role may read the prefix (see `ObjectStore`).

        Args:
            prefix: Key prefix to list.

        Returns:
            The metadata of the matching objects.
        """
        self.need_read(prefix)
        return self.inner.list_objects(prefix)

    def delete(self, key: str) -> None:
        """Deletes an object if the role may write the key (see `ObjectStore`).

        Args:
            key: The object key.
        """
        self.need_write(key)
        self.inner.delete(key)
