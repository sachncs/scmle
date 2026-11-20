"""Tenant isolation at the storage layer.

`Tenancy` wraps any `Depot` and confines every operation to the namespace
`tenants/<tenant>/` of the store underneath. Callers name keys relative to
their own namespace; the prefix is added here and never taken from the key,
so no key can name another tenant's object:

* keys are checked by `objectstore.validate` before the prefix is added, so
  `..`, empty segments, absolute paths, backslashes and percent-encoded
  separators are refused;
* a key whose first segment is the namespace root (`tenants`) is refused
  with `TenantError`, so a key that tries to name another tenant is visible
  as such rather than silently nested;
* tenant names are lowercase DNS labels, so two tenants can never map to
  the same directory on a case-insensitive file system;
* metadata a backend returns for a key outside the namespace is refused, so
  a backend that ignored the prefix cannot leak another tenant's listing.

Symbolic links inside a local root are refused by
`objectstore.Disk.locate`, which closes the remaining route from
one tenant's directory into another's.

Stores written before namespacing keep their objects at the root. `adopt`
moves them into one tenant's namespace, and can be re-run after an
interruption.
"""

from collections.abc import Mapping
import dataclasses
import os
import pathlib
import re

from scmle.storage import objectstore

ROOT = "tenants"
# A tenant name is an RFC 1123 DNS label: at most 63 lowercase letters,
# digits and inner hyphens. The bound is not statistical; it is the largest
# name every place a tenant is named in can hold (a DNS label, a Kubernetes
# namespace, a bucket prefix segment), so a tenant name never needs
# shortening, and shortening is where two names could collide.
MAX_TENANT_LENGTH = 63
NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?")
VARIABLE = "SCMLE_TENANT"


class TenantError(objectstore.StoreError):
    """A tenant name is invalid or an operation reached another tenant."""

    category = "store_tenant"


def qualify(tenant: str) -> str:
    """Checks that a tenant name is a safe namespace.

    Args:
        tenant: The candidate name.

    Returns:
        `tenant`, unchanged.

    Raises:
        TenantError: If the name is not a string of 1 to 63 lowercase
            letters, digits and inner hyphens.
    """
    candidate: object = tenant
    if (
        not isinstance(candidate, str)
        or len(candidate) > MAX_TENANT_LENGTH
        or NAME.fullmatch(candidate) is None
    ):
        raise TenantError(f"invalid tenant name {tenant!r}")
    return tenant


def directory(root: str | pathlib.Path, tenant: str) -> pathlib.Path:
    """The directory that holds one tenant's files under a root.

    Args:
        root: The shared root directory.
        tenant: The tenant.

    Returns:
        `<root>/tenants/<tenant>`, absolute; it is not created.

    Raises:
        TenantError: If the tenant name is invalid.
        EscapeError: If an existing part of the path is a symbolic link.
    """
    base = pathlib.Path(os.path.abspath(root))  # noqa: PTH100 -- lexical, see Disk.locate
    target = base / ROOT / qualify(tenant)
    real = pathlib.Path(os.path.realpath(base), ROOT, tenant)
    if os.path.normcase(os.path.realpath(target)) != os.path.normcase(real):
        raise objectstore.EscapeError(
            f"the directory of tenant {tenant!r} is a symbolic link"
        )
    return target


class Tenancy(objectstore.Depot):
    """A store confined to one tenant's namespace of another store.

    Attributes:
        inner: The shared store underneath.
        tenant: The tenant whose objects this store holds.
        prefix: The inner key prefix of the namespace, with a trailing `/`.
    """

    def __init__(self, inner: objectstore.Depot, tenant: str) -> None:
        """Initializes the store.

        Args:
            inner: The shared store underneath.
            tenant: The tenant whose objects this store holds.

        Raises:
            TenantError: If the tenant name is invalid.
        """
        self.tenant = qualify(tenant)
        self.inner = inner
        self.prefix = f"{ROOT}/{tenant}/"
        super().__init__(f"{inner.location}/{ROOT}/{tenant}")
        self.scheme = inner.scheme

    def place(self, key: str) -> str:
        """Maps a key of this tenant to the key underneath.

        Args:
            key: The key, relative to the tenant's namespace.

        Returns:
            The inner key.

        Raises:
            StoreError: If the key is not valid.
            TenantError: If the key starts with the namespace root.
        """
        head = objectstore.validate(key).split("/", 1)[0]
        if head.lower() == ROOT:
            raise TenantError(
                f"key {key!r} names the tenant namespace", {"key": key}
            )
        return self.prefix + key

    def strip(self, meta: objectstore.Metadata) -> objectstore.Metadata:
        """Maps metadata from underneath back to this tenant's key.

        Args:
            meta: Metadata the inner store returned.

        Returns:
            The same metadata under the key relative to the namespace.

        Raises:
            TenantError: If the key lies outside this tenant's namespace.
        """
        if not meta.key.startswith(self.prefix):
            raise TenantError("the store returned an object of another tenant")
        return dataclasses.replace(meta, key=meta.key.removeprefix(self.prefix))

    def put(
        self,
        key: str,
        data: bytes,
        *,
        if_absent: bool = False,
        if_version: str | None = None,
    ) -> objectstore.Metadata:
        """See `Depot.put`; the key is confined to the namespace.

        Args:
            key: The object key.
            data: The content to store.
            if_absent: Create the object only if none exists.
            if_version: Replace only if the current version is this one.

        Returns:
            The metadata of the stored object.
        """
        inner = self.place(key)
        return self.strip(
            self.inner.put(
                inner, data, if_absent=if_absent, if_version=if_version
            )
        )

    def get(self, key: str) -> tuple[bytes, objectstore.Metadata]:
        """See `Depot.get`; the key is confined to the namespace.

        Args:
            key: The object key.

        Returns:
            The content and metadata.
        """
        data, meta = self.inner.get(self.place(key))
        return data, self.strip(meta)

    def head(self, key: str) -> objectstore.Metadata:
        """See `Depot.head`; the key is confined to the namespace.

        Args:
            key: The object key.

        Returns:
            The metadata.
        """
        return self.strip(self.inner.head(self.place(key)))

    def catalog(self, prefix: str = "") -> list[objectstore.Metadata]:
        """See `Depot.catalog`; only this tenant's objects are listed.

        Args:
            prefix: Key prefix within the namespace; empty for every object.

        Returns:
            The metadata of each matching object, sorted by key.
        """
        if prefix:
            self.place(prefix.removesuffix("/"))
        found = self.inner.catalog(self.prefix + prefix)
        return [self.strip(meta) for meta in found]

    def delete(self, key: str, *, if_version: str | None = None) -> None:
        """See `Depot.delete`; the key is confined to the namespace.

        Args:
            key: The object key.
            if_version: Remove only if the current version is this one.
        """
        self.inner.delete(self.place(key), if_version=if_version)

    def adopt(self) -> list[str]:
        """Moves every object outside all namespaces into this tenant's.

        This is the migration of a store written before tenants were
        namespaced; run it once, for the tenant the store served. Each
        object is copied with `if_absent`, then its original is removed
        only if unchanged, so an interrupted run is finished by running it
        again: a copy already in place with the same content counts as done.

        Returns:
            The keys moved, sorted.

        Raises:
            ConflictError: If a different object already sits at the
                destination, or the original changed during the move.
        """
        moved = []
        for meta in self.inner.catalog():
            if meta.key.split("/", 1)[0].lower() == ROOT:
                continue
            data, current = self.inner.get(meta.key)
            target = self.prefix + meta.key
            try:
                self.inner.put(target, data, if_absent=True)
            except objectstore.ExistsError:
                if self.inner.head(target).sha256 != current.sha256:
                    raise objectstore.ConflictError(
                        f"another object already sits at {target!r}",
                        {"key": meta.key},
                    ) from None
            self.inner.delete(meta.key, if_version=current.version)
            moved.append(meta.key)
        return moved


def confine(
    store: objectstore.Depot, environ: Mapping[str, str]
) -> objectstore.Depot:
    """Confines a store to the tenant the environment names.

    Args:
        store: The shared store.
        environ: The environment; `SCMLE_TENANT` names the tenant.

    Returns:
        A `Tenancy` of the tenant, or `store` itself when no tenant is set.

    Raises:
        TenantError: If the tenant name is invalid.
    """
    tenant = environ.get(VARIABLE, "")
    return Tenancy(store, tenant) if tenant else store
