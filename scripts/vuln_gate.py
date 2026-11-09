#!/usr/bin/env python3
"""Vulnerability-scan gate: see `scmle.deploy.vuln_gate` for the policy."""

import sys

from scmle.deploy import vuln_gate

if __name__ == "__main__":
    sys.exit(vuln_gate.main())
