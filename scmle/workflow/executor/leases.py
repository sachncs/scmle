"""Run leases with fencing tokens, and the writes they fence (CON-1)."""

from collections.abc import Generator
from collections.abc import Mapping
import contextlib
import json
import logging
import threading
from typing import Any

from scmle.storage import objectstore
from scmle.workflow.executor import failures
from scmle.workflow.executor import records

LEASE_SECONDS = 300.0
# Tries at taking a lease that changes hands under us (deleted between the
# failed create and the read); more means someone else keeps winning.
LEASE_ATTEMPTS = 3
LOGGER = logging.getLogger(__name__)
# A conditional write on a lease that changed or vanished under us.
CHANGED = (objectstore.ConflictError, objectstore.AbsentError)


class Leases(records.Records):
    """Takes, holds, renews and gives up run leases; fences every write."""

    def write(self, key: str, record: Mapping[str, Any]) -> None:
        """Stores a JSON record atomically, fenced by the run's lease.

        A record under a run this executor leases carries the lease's
        fencing token. It is written only if the stored record's token is
        not newer than ours, `hold` confirms the lease is still ours, and
        the record is still the version read first (compare-and-swap), so
        an executor that lost its lease, even between the check and the
        write, stops instead of overwriting the new holder's state (CON-1).

        Args:
            key: Object key.
            record: The record to store.

        Raises:
            EvictionError: If the run's lease is no longer ours, or a later
                holder wrote the record.
        """
        prefix = f"workflows/{self.workflow.name}/runs/"
        run_id = key[len(prefix) :].split("/", 1)[0]
        if (
            not key.startswith(prefix)
            or key.endswith("/lock")
            or run_id not in self.fences
        ):
            super().write(key, record)
            return
        fence = self.fences[run_id]
        body = json.dumps(
            {**record, "fence": fence}, indent=2, sort_keys=True
        ).encode()
        try:
            data, meta = self.store.get(key)
        except objectstore.AbsentError:
            # handled: no record yet; it is created only if still absent
            version = None
        else:
            version = meta.version
            stored = records.parse(key, data).get("fence", 0)
            if stored > fence:
                raise failures.EvictionError(
                    f"{key} was written under fence {stored}, ours {fence}"
                )
        self.hold(run_id)
        try:
            if version is None:
                self.store.put(key, body, if_absent=True)
            else:
                self.store.put(key, body, if_version=version)
        except (objectstore.ExistsError, *CHANGED) as error:
            raise failures.EvictionError(
                f"{key} changed while writing"
            ) from error

    def lease(self, run_id: str, fence: int, floor: float = 0.0) -> bytes:
        """The body of this executor's lease on a run.

        Args:
            run_id: The run.
            fence: The fencing token: one more than the lease it replaced.
            floor: An expiry the new one never falls below, so a wall
                clock stepped back never shortens a renewed lease (CON-20).

        Returns:
            The JSON bytes to store.
        """
        del run_id
        return json.dumps(
            {
                "owner": self.owner,
                "expires": max(self.clock() + self.tenure, floor),
                "fence": fence,
            }
        ).encode()

    def acquire(self, run_id: str) -> None:
        """Acquires the run's lease, or raises `BusyError`.

        A free run is taken with a conditional create. An expired lease is
        taken over with one compare-and-swap from the version just read, so
        of two executors racing for it exactly one wins (CON-1); the winner's
        fencing token is one more than the expired lease's.

        Args:
            run_id: The run.

        Raises:
            BusyError: If another executor holds a valid lease, or wins
                the race for an expired one.
        """
        key = self.base(run_id) + "lock"
        for ignored in range(LEASE_ATTEMPTS):
            try:
                self.store.put(key, self.lease(run_id, 1), if_absent=True)
            except objectstore.ExistsError:
                # handled: existing lease is expected; falls through to read and
                # check it
                pass
            else:
                self.fences[run_id] = 1
                return
            try:
                data, meta = self.store.get(key)
            except objectstore.AbsentError:
                # handled: lease vanished mid-read; bounded retry then
                # BusyError
                continue
            held = records.parse(key, data)
            if held["expires"] > self.clock():
                raise failures.BusyError(
                    f"run {run_id} is held by {held['owner']}"
                )
            fence = int(held.get("fence", 0)) + 1
            try:
                self.store.put(
                    key, self.lease(run_id, fence), if_version=meta.version
                )
            except (
                objectstore.ConflictError,
                objectstore.AbsentError,
            ) as error:
                raise failures.BusyError(
                    f"run {run_id} was just taken"
                ) from error
            self.fences[run_id] = fence
            return
        raise failures.BusyError(f"run {run_id} is changing hands; retry")

    def hold(
        self, run_id: str
    ) -> tuple[Mapping[str, Any], objectstore.ObjectMeta]:
        """Checks that this executor still holds the run's lease.

        Args:
            run_id: The run.

        Returns:
            The lease record, and its metadata, whose version a renewal or
            release must match.

        Raises:
            EvictionError: If the lease is gone, or another executor or a
                later fencing token holds it.
        """
        fence = self.fences.get(run_id)
        try:
            data, meta = self.store.get(self.base(run_id) + "lock")
        except objectstore.AbsentError:
            raise failures.EvictionError(
                f"the lease on run {run_id} is gone"
            ) from None
        held = records.parse(self.base(run_id) + "lock", data)
        if (
            fence is None
            or held.get("owner") != self.owner
            or held.get("fence") != fence
        ):
            raise failures.EvictionError(
                f"run {run_id} is now held by {held.get('owner')} "
                f"(fence {held.get('fence')}, ours {fence})"
            )
        return held, meta

    def renew(self, run_id: str) -> None:
        """Extends the lease, only if this executor still holds it.

        Args:
            run_id: The run.

        Raises:
            EvictionError: If the lease was taken over or changed between
                the check and the renewal.
        """
        held, meta = self.hold(run_id)
        try:
            self.store.put(
                self.base(run_id) + "lock",
                self.lease(run_id, self.fences[run_id], float(held["expires"])),
                if_version=meta.version,
            )
        except CHANGED as error:
            raise failures.EvictionError(
                f"the lease on run {run_id} changed while renewing"
            ) from error

    def relinquish(self, run_id: str) -> None:
        """Gives the lease up, never one another executor now holds.

        Args:
            run_id: The run.
        """
        try:
            meta = self.hold(run_id)[1]
            self.store.delete(
                self.base(run_id) + "lock", if_version=meta.version
            )
        except (failures.EvictionError, objectstore.ConflictError):
            LOGGER.warning(
                "WARNING run %s: the lease was no longer ours at release",
                run_id,
            )
        finally:
            self.fences.pop(run_id, None)

    @contextlib.contextmanager
    def pulse(self, run_id: str) -> Generator[None, None, None]:
        """Keeps the run's lease renewed by a heartbeat while steps run.

        Args:
            run_id: The run.

        Yields:
            None, while the heartbeat runs; it stops on exit.
        """
        stop = threading.Event()
        heartbeat = threading.Thread(
            target=self.beat, args=(run_id, stop), daemon=True
        )
        heartbeat.start()
        try:
            yield
        finally:
            stop.set()
            heartbeat.join()

    def beat(self, run_id: str, stop: threading.Event) -> None:
        """Renews the lease every third of its validity until `stop` is set.

        A failed renewal is logged and ends the heartbeat; the next write
        under the run then refuses (`hold`), so a lost lease never writes.

        Args:
            run_id: The run.
            stop: Set when the run's steps are done.
        """
        while not stop.wait(self.tenure / 3):
            try:
                self.renew(run_id)
            except failures.EvictionError:
                LOGGER.warning("WARNING run %s: the lease was lost", run_id)
                return

    def scan(self) -> list[str]:
        """Lists the runs of this workflow whose lease is valid.

        Returns:
            The run ids.
        """
        prefix = f"workflows/{self.workflow.name}/runs/"
        active = []
        for meta in self.store.list_objects(prefix):
            if not meta.key.endswith("/lock"):
                continue
            held = self.read(meta.key)
            if held and held["expires"] > self.clock():
                active.append(meta.key.split("/")[3])
        return active
