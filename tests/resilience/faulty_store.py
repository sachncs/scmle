"""A store wrapper that fails the calls it is told to, then behaves."""

from scmle.data.store import objectstore


def throttled():
    return objectstore.TransientStoreError("throttled")


class FaultyStore(objectstore.ObjectStore):
    """Raises an error for a window of the calls that match a filter.

    A call matches when its operation is in `operations` and its key (the
    prefix, for a listing) starts with `prefix`. Among the matching calls,
    the first `skip` pass, the next `failures` raise, the rest pass.

    Attributes:
        inner: The wrapped store.
        operations: Operation names: put, get, head, list_objects, delete.
        prefix: Key prefix a call must start with to match.
        failures: How many matching calls raise.
        skip: How many matching calls pass before the failures start.
        error: Zero-argument callable building the error to raise.
        matching: How many matching calls have been made.
        calls: Every call made, as (operation, key).
    """

    def __init__(
        self,
        inner,
        failures=0,
        *,
        operations=("put", "get", "head", "list_objects", "delete"),
        prefix="",
        skip=0,
        error=throttled,
    ):
        super().__init__(inner.location)
        self.scheme = inner.scheme
        self.inner = inner
        self.operations = tuple(operations)
        self.prefix = prefix
        self.failures = failures
        self.skip = skip
        self.error = error
        self.matching = 0
        self.calls = []

    def heal(self):
        self.failures = 0

    def strike(self, operation, key):
        self.calls.append((operation, key))
        if operation in self.operations and key.startswith(self.prefix):
            number = self.matching
            self.matching += 1
            if self.skip <= number < self.skip + self.failures:
                raise self.error()

    def put(self, key, data, *, if_absent=False):
        self.strike("put", key)
        return self.inner.put(key, data, if_absent=if_absent)

    def get(self, key):
        self.strike("get", key)
        return self.inner.get(key)

    def head(self, key):
        self.strike("head", key)
        return self.inner.head(key)

    def list_objects(self, prefix=""):
        self.strike("list_objects", prefix)
        return self.inner.list_objects(prefix)

    def delete(self, key):
        self.strike("delete", key)
        self.inner.delete(key)
