#!/usr/bin/env python3
"""List the resources an environment left behind after `terraform destroy`.

Reads saved listings (see `scmle.deploy.orphans`); calls no cloud API.
Exit 0 only when nothing is left in the account or in the state.
"""

import sys

from scmle.deploy import orphans

if __name__ == "__main__":
    sys.exit(orphans.main())
