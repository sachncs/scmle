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
        """Whether `key` may be read."""
        return key.startswith(self.reads)

    def allows_write(self, key: str) -> bool:
        """Whether `key` may be written."""
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

    Raises:
        ConfigError: If the two share a name, a write prefix, or if either
            can write what the other reads as its release input
            (`bundles/`).
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
    """A store that refuses what its role does not allow."""

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
        """Raises unless the role may read `key`."""
        if not self.role.allows_read(key):
            raise RoleDeniedError(f"role {self.role.name} may not read {key}")

    def need_write(self, key: str) -> None:
        """Raises unless the role may write `key`."""
        if not self.role.allows_write(key):
            raise RoleDeniedError(f"role {self.role.name} may not write {key}")

    def put(
        self, key: str, data: bytes, *, if_absent: bool = False
    ) -> objectstore.ObjectMeta:
        """See `ObjectStore.put`."""
        self.need_write(key)
        return self.inner.put(key, data, if_absent=if_absent)

    def get(self, key: str) -> tuple[bytes, objectstore.ObjectMeta]:
        """See `ObjectStore.get`."""
        self.need_read(key)
        return self.inner.get(key)

    def head(self, key: str) -> objectstore.ObjectMeta:
        """See `ObjectStore.head`."""
        self.need_read(key)
        return self.inner.head(key)

    def list_objects(self, prefix: str = "") -> list[objectstore.ObjectMeta]:
        """See `ObjectStore.list_objects`."""
        self.need_read(prefix)
        return self.inner.list_objects(prefix)

    def delete(self, key: str) -> None:
        """See `ObjectStore.delete`."""
        self.need_write(key)
        self.inner.delete(key)
