"""Properties of the confirmation carve over random inputs."""

from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies
import numpy
from sklearn import model_selection

from scmle.contracts import errors
from scmle.data.custody import carve


@strategies.composite
def scenarios(draw):
    n = draw(strategies.integers(60, 400))
    fraction = draw(strategies.floats(0.1, 0.5))
    seed = draw(strategies.integers(0, 10_000))
    kind = draw(
        strategies.sampled_from(["plain", "groups", "dups", "time", "both"])
    )
    labels = draw(strategies.booleans())
    embargo = draw(strategies.integers(0, 5))
    return n, fraction, seed, kind, labels, embargo


def inputs(n, kind, seed):
    rng = numpy.random.default_rng(seed)
    groups = (
        rng.integers(0, max(n // 4, 2), n)
        if kind in {"groups", "both"}
        else None
    )
    duplicates = None
    if kind in {"dups", "both"}:
        duplicates = numpy.full(n, -1)
        pairs = rng.choice(n, size=min(40, n // 2), replace=False)
        for a, b in zip(pairs[::2], pairs[1::2], strict=False):
            duplicates[a] = duplicates[b] = a
    times = numpy.arange(n) if kind in {"time", "both"} else None
    return groups, duplicates, times


@settings(max_examples=1000, deadline=None)
@example((60, 0.5, 0, "time", False, 5))
@given(scenarios())
def test_sets_never_overlap_and_units_never_straddle(scenario):
    n, fraction, seed, kind, with_labels, embargo = scenario
    groups, duplicates, times = inputs(n, kind, seed)
    labels = (numpy.arange(n) % 3 == 0).astype(int) if with_labels else None
    dev = numpy.arange(n) + 7
    folds = list(
        model_selection.KFold(3, shuffle=True, random_state=0).split(dev)
    )
    arguments = {
        "fraction": fraction,
        "seed": seed,
        "charter_hash": "a" * 64,
        "data_hash": "b" * 64,
        "labels": labels,
        "groups": groups,
        "times": times,
        "duplicates": duplicates,
        "embargo_rows": embargo,
    }
    refusal = None
    try:
        result = carve.carve_confirmation(dev, folds, **arguments)
    except errors.ConfigError as error:
        refusal = error
    if refusal is not None:
        # An explicit refusal is allowed only for structured input, and only
        # for the documented reason; plain random data must always succeed.
        assert kind != "plain", refusal
        assert "validation fold" in str(refusal)
        return
    x = set(result.exploration_ids.tolist())
    c = set(result.confirmation_ids.tolist())
    e = set(result.embargo_ids.tolist())
    assert x | c | e == set(dev.tolist())
    assert not (x & c) and not (x & e) and not (c & e)
    position = {int(i): p for p, i in enumerate(dev)}
    if groups is not None:
        gx = {groups[position[i]] for i in x}
        gc = {groups[position[i]] for i in c}
        assert gx.isdisjoint(gc)
    if duplicates is not None:
        for label in set(duplicates[duplicates >= 0].tolist()):
            members = {i for i in dev if duplicates[position[int(i)]] == label}
            assert not (members & c and members & x)
    again = carve.carve_confirmation(dev, folds, **arguments)
    assert numpy.array_equal(again.confirmation_ids, result.confirmation_ids)
