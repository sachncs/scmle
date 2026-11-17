"""Each planted leak is what its type says, for classes and for numbers."""

import numpy
import pandas
import pytest

from scmle.assurance.admissibility import plants


def numbers(rows=400, seed=0):
    rng = numpy.random.default_rng(seed)
    return pandas.Series(rng.normal(size=rows) * 3 + 10, name="y")


def classes(rows=400, seed=0, count=3):
    rng = numpy.random.default_rng(seed)
    return pandas.Series(rng.choice(["a", "b", "c"][:count], rows), name="y")


def test_number_plants_are_derived_from_the_target():
    target = numbers()
    plant = plants.plant
    assert plant(target, "leak_copy", 0, True).equals(target.astype(float))
    assert plant(target, "leak_noisy_copy", 0, True).corr(target) > 0.99
    assert plant(target, "leak_threshold", 0, True).corr(target) > 0.9
    gated = plant(target, "leak_missing_for_class", 0, True)
    assert (target[gated.notna()] <= target.median()).all()
    rare = plant(target, "leak_rare_record", 0, True)
    assert (target[rare.notna()] >= target.quantile(0.9)).all()
    assert 0.05 < rare.notna().mean() < 0.15


def test_class_plants_are_derived_from_the_target():
    target = classes()
    codes = pandas.Series(pandas.factorize(target)[0].astype(float))
    plant = plants.plant
    assert plant(target, "leak_copy", 0, False).equals(codes)
    noisy = plant(target, "leak_noisy_copy", 0, False)
    moved = (noisy != codes).mean()
    assert 0 < moved < 0.06
    assert set(noisy.unique()) <= set(codes.unique())
    assert plant(target, "leak_threshold", 0, False).corr(codes) > 0.9
    first = target == target.iloc[0]
    gated = plant(target, "leak_missing_for_class", 0, False)
    assert gated[first].isna().all() and gated[~first].notna().all()
    rare = plant(target, "leak_rare_record", 0, False)
    assert set(target[rare.notna()]) == {target.iloc[0]}
    assert 0.04 < rare.notna().mean() < 0.15


def test_a_two_class_noisy_copy_moves_labels_to_the_other_class():
    target = classes(count=2)
    codes = pandas.Series(pandas.factorize(target)[0].astype(float))
    noisy = plants.plant(target, "leak_noisy_copy", 0, False)
    changed = noisy != codes
    assert changed.any()
    assert ((noisy[changed] + codes[changed]) == 1).all()


def test_plants_are_reproducible_and_seeded():
    target = numbers()
    one = plants.plant(target, "leak_noisy_copy", 3, True)
    assert one.equals(plants.plant(target, "leak_noisy_copy", 3, True))
    assert not one.equals(plants.plant(target, "leak_noisy_copy", 4, True))


def test_the_table_adds_one_column_per_type_and_keeps_the_rows():
    frame = pandas.DataFrame({"x": range(50), "y": numbers(50)})
    planted = plants.table(frame, "y", 0, True)
    assert list(planted.columns) == ["x", "y", *plants.DEFECTS]
    assert planted[["x", "y"]].equals(frame)


def test_unknown_types_and_name_clashes_are_refused():
    with pytest.raises(KeyError):
        plants.plant(numbers(), "leak_unknown", 0, True)
    frame = pandas.DataFrame({"y": numbers(20), "leak_copy": 0.0})
    with pytest.raises(ValueError, match="named as plants"):
        plants.table(frame, "y", 0, True)
