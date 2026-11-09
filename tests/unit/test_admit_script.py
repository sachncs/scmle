"""The admission script runs a command only when the machine has room."""

import importlib.util
import os
import pathlib
import subprocess  # nosec B404
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX load average and slot directories"
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "admit.py"
EASY = ["--load", "1000", "--memory", "0", "--poll", "0.05"]


def load():
    """Imports the script as a module, since `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location("admit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(arguments, root):
    """Runs the script with its slot directory under `root`."""
    environment = dict(os.environ, TMPDIR=str(root))
    environment.pop("OMP_NUM_THREADS", None)
    return subprocess.run(  # nosec B603
        [sys.executable, str(SCRIPT), *arguments],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )


def test_the_command_runs_and_its_exit_status_is_returned(tmp_path):
    command = [sys.executable, "-c", "raise SystemExit(3)"]

    result = run([*EASY, "--", *command], tmp_path)

    assert result.returncode == 3


def test_math_library_threads_default_to_one(tmp_path):
    code = "import os; print(os.environ['OMP_NUM_THREADS'])"

    result = run([*EASY, "--", sys.executable, "-c", code], tmp_path)

    assert result.stdout.strip() == "1"


def test_the_thread_count_can_be_raised(tmp_path):
    code = "import os; print(os.environ['OMP_NUM_THREADS'])"
    arguments = [*EASY, "--threads", "2", "--", sys.executable, "-c", code]

    result = run(arguments, tmp_path)

    assert result.stdout.strip() == "2"


def test_it_gives_up_when_the_machine_stays_busy(tmp_path):
    busy = ["--load", "0.0001", "--poll", "0.05", "--timeout", "0.3"]

    result = run([*busy, "--", sys.executable, "-c", "print('ran')"], tmp_path)

    assert result.returncode == 75
    assert "ran" not in result.stdout


def test_a_missing_command_is_a_usage_error(tmp_path):
    result = run(EASY, tmp_path)

    assert result.returncode == 2


def test_slots_are_exclusive_and_come_back_when_released(tmp_path):
    admit = load()

    first = admit.claim(tmp_path, 2)
    second = admit.claim(tmp_path, 2)
    third = admit.claim(tmp_path, 2)
    admit.release(first)
    again = admit.claim(tmp_path, 2)

    assert {first.name, second.name} == {"slot0", "slot1"}
    assert third is None
    assert again.name == first.name


def test_a_slot_left_by_a_dead_process_is_reclaimed(tmp_path):
    admit = load()
    dead = subprocess.Popen([sys.executable, "-c", "pass"])  # nosec B603
    dead.wait()
    stale = tmp_path / "slot0"
    stale.mkdir()
    (stale / "pid").write_text(str(dead.pid), encoding="utf-8")

    slot = admit.claim(tmp_path, 1)

    assert slot is not None
    assert slot.name == "slot0"
    assert (slot / "pid").read_text(encoding="utf-8") == str(os.getpid())


def test_a_slot_that_is_still_being_claimed_is_left_alone(tmp_path):
    admit = load()
    claiming = tmp_path / "slot0"
    claiming.mkdir()

    slot = admit.claim(tmp_path, 2)

    assert slot is not None
    assert slot.name == "slot1"
    assert claiming.exists()


def test_memory_is_measured_in_gigabytes():
    admit = load()

    available = admit.measure()

    assert available > 0
