"""Keeps cloud SDKs and infrastructure tooling out of the declared dependencies.

The package runs on a local machine and a local container. Nothing in
`pyproject.toml` (base dependencies, extras or dependency groups) may name a
cloud SDK, a hosted-compute client or a Terraform tool.
"""

import pathlib
import re
import tomllib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
FORBIDDEN = re.compile(
    r"^(boto3|botocore|aiobotocore|s3fs|gcsfs|modal|python-terraform"
    r"|terraform.*|pulumi.*|opentofu.*|google-cloud-.+|azure-.+)$"
)


def requirement_name(requirement: str) -> str:
    """The normalised distribution name of a requirement string."""
    name = re.split(r"[\[<>=!~;@ ]", requirement.strip(), maxsplit=1)[0]
    return name.lower().replace("_", "-")


def declared_requirements() -> list[tuple[str, str]]:
    """Every (where, requirement) pair that pyproject.toml declares."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    found = [("dependencies", d) for d in project["project"]["dependencies"]]
    for extra, requirements in (
        project["project"].get("optional-dependencies", {}).items()
    ):
        found.extend((f"extra {extra}", r) for r in requirements)
    for group, requirements in project.get("dependency-groups", {}).items():
        found.extend(
            (f"group {group}", r) for r in requirements if isinstance(r, str)
        )
    return found


def test_the_declared_requirements_are_found():
    assert len(declared_requirements()) > 10


def test_no_declared_requirement_is_a_cloud_sdk_or_terraform_tool():
    named = [
        f"{where}: {requirement}"
        for where, requirement in declared_requirements()
        if FORBIDDEN.match(requirement_name(requirement))
    ]
    assert not named, named


@pytest.mark.parametrize(
    "name",
    [
        "boto3",
        "botocore",
        "google-cloud-storage",
        "modal",
        "azure-storage-blob",
        "python-terraform",
        "terraform-compliance",
    ],
)
def test_the_guard_recognises_a_forbidden_name(name):
    assert FORBIDDEN.match(requirement_name(f"{name}>=1.0"))
