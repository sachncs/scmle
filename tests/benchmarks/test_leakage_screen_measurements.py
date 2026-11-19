"""The documented re-measurement of shipped tables runs for every leak type."""

from benchmarks import leakage_screen_measurements
from benchmarks.qualification import f4_certify
from benchmarks.qualification import screen_certify
from scmle.assurance.admissibility import plants
from scmle.assurance.admissibility import screen


def test_every_leak_type_the_certificate_names_can_be_planted():
    frame = f4_certify.breast_cancer()
    names = screen_certify.real_columns(frame, 3, 0)
    for defect in screen_certify.DEFECTS:
        table, column = leakage_screen_measurements.planted_table(
            frame, names, defect, 0
        )
        if defect == plants.DUPLICATES:
            assert column == screen.ROWS and len(table) > len(frame)
        else:
            assert column in table.columns and len(table) == len(frame)


def test_the_planted_measurement_covers_every_leak_type(monkeypatch):
    monkeypatch.setattr(leakage_screen_measurements, "PLANT_SEEDS", (0,))
    found = leakage_screen_measurements.planted()
    assert list(found) == list(screen_certify.DEFECTS)
    assert all(row["tables"] == 1 for row in found.values())
    assert found[plants.DUPLICATES]["detected"] == 1
