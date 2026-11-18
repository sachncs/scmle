"""The sealed test partition: unavailable to search, released once.

`Lockbox` is the interface every holder of test rows implements; `Resident`
keeps the rows in this process. The sealed holder, which keeps them in a
separate process, is `scmle.assurance.custody.vault.SealedTestSet`.
"""

import abc
import dataclasses
import secrets

import pandas

from scmle.contracts import errors


class Lockbox(abc.ABC):
    """The test partition: unavailable to search, released once.

    Attributes:
        ids: Row ids of the test rows.
        opened: Whether the rows were released.
    """

    ids: tuple[int, ...]
    opened: bool

    @abc.abstractmethod
    def mint(self) -> str:
        """Returns the release token; call only once selection is frozen.

        Returns:
            The token that `open` requires.
        """

    @abc.abstractmethod
    def bind(self, binding: str) -> None:
        """Fixes what the single release is for; call before `open`.

        After a binding is set, `open` releases the rows only to a caller
        that presents the same binding. A set can be bound once.

        Args:
            binding: Digest of the selection the release is meant to test.

        Raises:
            LeakageError: If a binding is already set or the rows were
                released.
        """

    @abc.abstractmethod
    def open(
        self, token: str, binding: str | None = None
    ) -> tuple[pandas.DataFrame, pandas.Series]:
        """Releases the test data a single time.

        Args:
            token: The token from `mint`.
            binding: The binding fixed by `bind`; required once one is set.

        Returns:
            Test features and labels.

        Raises:
            LeakageError: On a wrong token, a repeated release or a binding
                that differs from the one set.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Releases any resource held for the lock."""

    def __len__(self) -> int:
        """Returns the number of test rows.

        Returns:
            The length of `ids`.
        """
        return len(self.ids)


@dataclasses.dataclass(slots=True)
class Resident(Lockbox):
    """Test rows kept in this process, behind a one-time token.

    Python cannot hide attributes, so the lock is a protocol: trial code only
    receives development data, and `open` enforces a one-time, token-gated
    release that is recorded. The sealed variant moves the rows to a separate
    process (see `scmle.assurance.custody.vault`).

    Attributes:
        features: Test features.
        target: Test labels.
        ids: Row ids of the test rows.
        token: Secret issued by `mint` (excluded from repr).
        opened: Whether the set has been released.
        binding: Digest the release is bound to, or None.
    """

    features: pandas.DataFrame = dataclasses.field(repr=False)
    target: pandas.Series = dataclasses.field(repr=False)
    ids: tuple[int, ...]
    token: str = dataclasses.field(
        default_factory=lambda: secrets.token_hex(8), repr=False
    )
    opened: bool = False
    binding: str | None = None

    def mint(self) -> str:
        """Returns the release token; call only once selection is frozen.

        Returns:
            The token that `open` requires.
        """
        return self.token

    def bind(self, binding: str) -> None:
        """Fixes what the single release is for.

        Args:
            binding: Digest of the selection the release is meant to test.

        Raises:
            LeakageError: If a binding is already set or the rows were
                released.
        """
        if self.binding is not None or self.opened:
            raise errors.LeakageError("test set is already bound or used")
        self.binding = binding

    def open(
        self, token: str, binding: str | None = None
    ) -> tuple[pandas.DataFrame, pandas.Series]:
        """Releases the test data a single time.

        Args:
            token: The token from `mint`.
            binding: The binding fixed by `bind`, which must come first.

        Returns:
            Test features and labels.

        Raises:
            LeakageError: On a wrong token, a repeated release, a set that was
                never bound, or a binding that differs from the one set. The
                token and binding are compared in constant time (GATE-3).
        """
        if not secrets.compare_digest(token.encode(), self.token.encode()):
            raise errors.LeakageError("test set is locked: invalid token")
        if self.opened:
            raise errors.LeakageError(
                "test set was already used for final evaluation"
            )
        if self.binding is None:
            raise errors.LeakageError(
                "test set is locked: bind it to the frozen selection first"
            )
        if binding is None or not secrets.compare_digest(
            binding.encode(), self.binding.encode()
        ):
            raise errors.LeakageError("test set is locked: binding mismatch")
        self.opened = True
        return self.features, self.target

    def close(self) -> None:
        """Nothing to release: the rows live in this process."""
