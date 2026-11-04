"""The closed loop and its escalation code cannot import custody.

Modules of the loop (`scmle.governance.loop`, `escalation`, `never_list`,
`deployment_ledger`) and its monitors (`scmle.monitoring.sequential_tests`,
`labels`) may not import the custody package, the vault, the splitters module
or the orchestrator, directly or through any scmle module they import. A
sealed set therefore reaches them only as a callable handed over by a person's
attested request (`Loop.reopen_sealed`), never through an import.
"""

import pytest
from test_exploration_imports import closure
from test_exploration_imports import forbidden
from test_exploration_imports import imported_modules
from test_exploration_imports import source_of

MODULES = (
    "scmle.governance.loop",
    "scmle.governance.escalation",
    "scmle.governance.never_list",
    "scmle.governance.deployment_ledger",
    "scmle.monitoring.sequential_tests",
    "scmle.monitoring.labels",
)


@pytest.mark.parametrize("module", MODULES)
def test_no_loop_module_imports_custody_directly(module):
    for name in imported_modules(source_of(module)):
        assert not forbidden(name), f"{module} imports {name}"


@pytest.mark.parametrize("module", MODULES)
def test_no_loop_module_reaches_custody_through_its_imports(module):
    for name, trail in closure(module).items():
        assert not forbidden(name), " -> ".join(trail)
