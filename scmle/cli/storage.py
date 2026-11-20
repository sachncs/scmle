"""The `store` command: maintenance of a shared object store.

    scmle store adopt --root store --tenant acme

`adopt` moves every object a store holds outside the tenant namespaces
into one tenant's namespace (`tenants/<tenant>/`), the migration of a store
written before tenants were namespaced. Each object is copied first and its
original removed only when unchanged, so running the same command again
after an interruption finishes the move.
"""

import argparse
import logging

from scmle.cli import command
from scmle.storage import objectstore
from scmle.storage import tenancy

LOGGER = logging.getLogger(__name__)


class Storage(command.Command):
    """Maintains a local object store."""

    name = "store"
    help = "maintain a local object store (adopt objects into a tenant)"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds the store actions.

        Args:
            parser: The argument parser of this command.
        """
        actions = parser.add_subparsers(dest="store_action", required=True)
        adopt = actions.add_parser(
            "adopt",
            help="move every object outside the tenant namespaces into one",
        )
        adopt.add_argument("--root", required=True, help="store directory")
        adopt.add_argument(
            "--tenant", required=True, help="tenant the store served"
        )

    def execute(self, args: argparse.Namespace) -> int:
        """Moves the objects and prints each key moved.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0); a run with nothing left to move
            succeeds too.
        """
        store = objectstore.Disk(args.root)
        moved = tenancy.Tenancy(store, args.tenant).adopt()
        for key in moved:
            print(key)
        LOGGER.info(
            "%d object(s) moved into %s/%s/",
            len(moved),
            tenancy.ROOT,
            args.tenant,
        )
        return command.EXIT_OK
