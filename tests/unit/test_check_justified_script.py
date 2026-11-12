"""The justification check lists definitions that nothing justifies."""

import json
import pathlib
import shutil
import subprocess  # nosec B404
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_justified.py"

PACKAGE = """
import abc


def used():
    return 1


def helper(value):
    return used() + value


def orphan():
    return 2


def only_tested():
    return 3


def forward(a, b):
    return helper(a, b)


class Holder:
    def __init__(self):
        self.size = 1

    def read(self):
        return self.size


class Lone:
    def work(self):
        return used()


class Port(abc.ABC):
    @abc.abstractmethod
    def send(self):
        ...


class Wire(Port):
    def send(self):
        return used()


class Registered:
    @staticmethod
    def kept():
        return 1

    @property
    def flagged(self):
        return 1


def decorated_by_framework():
    return 1


decorated_by_framework = staticmethod(decorated_by_framework)


def run_thing():
    return used() + helper(1) + forward(1, 2) + Holder().read()
"""

TESTS = """
from pkg import only_tested

def test_it():
    assert only_tested() == 3
"""


def build(tmp_path, dynamic="", reviewed=""):
    """Lays out a small repository around a copy of the script."""
    (tmp_path / "scripts").mkdir()
    shutil.copy(SCRIPT, tmp_path / "scripts" / "check_justified.py")
    (tmp_path / "scripts" / "justified_dynamic.txt").write_text(
        dynamic, encoding="utf-8"
    )
    (tmp_path / "scripts" / "justified_reviewed.txt").write_text(
        reviewed, encoding="utf-8"
    )
    (tmp_path / "scmle").mkdir()
    (tmp_path / "scmle" / "pkg.py").write_text(PACKAGE, encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_pkg.py").write_text(TESTS, encoding="utf-8")
    return tmp_path


def run(root, *arguments):
    """Runs the copied script and returns the process and its JSON, if any."""
    output = root / "out.json"
    result = subprocess.run(  # nosec B603
        [
            sys.executable,
            str(root / "scripts" / "check_justified.py"),
            "--json",
            str(output),
            *arguments,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    found = json.loads(output.read_text(encoding="utf-8"))
    return result, {(item["shape"], item["name"]) for item in found}


def test_a_definition_nothing_names_is_unreferenced(tmp_path):
    found = run(build(tmp_path))[1]

    assert ("unreferenced", "orphan") in found


def test_a_definition_only_tests_name_is_reported_as_tested_only(tmp_path):
    found = run(build(tmp_path))[1]

    assert ("tested only", "only_tested") in found
    assert ("unreferenced", "only_tested") not in found


def test_a_function_that_forwards_its_arguments_is_a_passthrough(tmp_path):
    found = run(build(tmp_path))[1]

    assert ("passthrough", "forward") in found
    assert ("passthrough", "helper") not in found


def test_a_method_that_returns_one_attribute_is_a_getter(tmp_path):
    found = run(build(tmp_path))[1]

    assert ("getter", "read") in found


def test_a_stateless_class_with_one_method_is_a_lone_method(tmp_path):
    found = run(build(tmp_path))[1]

    assert ("lone method", "Lone") in found
    assert ("lone method", "Holder") not in found


def test_an_interface_with_one_implementation_is_a_lone_interface(tmp_path):
    found = run(build(tmp_path))[1]

    assert ("lone interface", "Port") in found


def test_a_property_is_not_called_a_getter(tmp_path):
    found = run(build(tmp_path))[1]

    assert ("getter", "flagged") not in found


def test_a_declared_dynamic_use_is_not_reported(tmp_path):
    root = build(
        tmp_path, dynamic="scmle/pkg.py orphan built by name at run time\n"
    )

    found = run(root)[1]

    assert ("unreferenced", "orphan") not in found


def test_a_declaration_without_a_reason_is_refused(tmp_path):
    root = build(tmp_path, dynamic="scmle/pkg.py orphan\n")

    result = subprocess.run(  # nosec B603
        [sys.executable, str(root / "scripts" / "check_justified.py")],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "reason" in result.stderr


def test_a_reviewed_keep_is_not_reported_for_its_shape(tmp_path):
    root = build(
        tmp_path,
        reviewed="scmle/pkg.py flagged getter documented public accessor\n",
    )

    found = run(root)[1]

    assert ("getter", "flagged") not in found


def test_a_reviewed_keep_for_one_shape_does_not_hide_another(tmp_path):
    root = build(
        tmp_path,
        reviewed="scmle/pkg.py orphan getter documented public accessor\n",
    )

    found = run(root)[1]

    assert ("unreferenced", "orphan") in found


def test_a_reviewed_keep_without_a_reason_is_refused(tmp_path):
    root = build(tmp_path, reviewed="scmle/pkg.py flagged getter\n")

    result = subprocess.run(  # nosec B603
        [sys.executable, str(root / "scripts" / "check_justified.py")],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "reason" in result.stderr


def test_strict_mode_fails_while_candidates_remain(tmp_path):
    result, found = run(build(tmp_path), "--strict")

    assert found
    assert result.returncode == 1


def test_the_default_mode_reports_without_failing(tmp_path):
    result = run(build(tmp_path))[0]

    assert result.returncode == 0
    assert "unreferenced" in result.stdout


def test_fail_on_a_shape_exits_nonzero_only_when_that_flag_is_given(tmp_path):
    root = build(tmp_path)

    plain = run(root)[0]
    unreferenced = run(root, "--fail-on", "unreferenced")[0]

    assert plain.returncode == 0
    assert unreferenced.returncode == 1


def test_fail_on_a_shape_with_no_candidates_exits_zero(tmp_path):
    root = build(
        tmp_path, dynamic="scmle/pkg.py * every name is declared used\n"
    )

    result = run(root, "--fail-on", "unreferenced")[0]

    assert result.returncode == 0
