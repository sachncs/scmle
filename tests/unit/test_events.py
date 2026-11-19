"""Event streams: incremental store against a naive recompute, and sources."""

import io
import json
import math
import random

import numpy
import pandas
import pytest
from scipy import stats

from scmle.data import events
from scmle.data.events import event
from scmle.data.events import lateness
from scmle.data.events import windows

HOUR = 3_600 * windows.NANOSECONDS
START = pandas.Timestamp("2026-01-01", tz="UTC").value
SPECS = (
    "purchase.amount:total:6h",
    "purchase.amount:mean:12h",
    "purchase.amount:min:6h",
    "purchase.amount:max:24h",
    "purchase.amount:last:24h",
    "purchase:count:3h",
    "login:recency:24h",
    "*:count:2h",
    "*.channel:distinct:24h",
    "login.channel:last:48h",
)


def naive(admitted, entity, cutoff, window):
    """Recomputes one window from scratch over every admitted event."""
    rows = [
        (rank, item)
        for rank, item in enumerate(admitted)
        if item.entity == entity
        and cutoff - window.span < item.time <= cutoff
        and window.kind in {windows.ANY, item.kind}
        and (window.field is None or window.field in item.payload)
    ]
    rows.sort(key=lambda pair: (pair[1].time, pair[0]))
    times = [pair[1].time for pair in rows]
    values = [
        True if window.field is None else pair[1].payload[window.field]
        for pair in rows
    ]
    numeric = [
        float(v)
        for v in values
        if isinstance(v, int | float) and not isinstance(v, bool)
    ]
    name = window.statistic.name
    if name == "count":
        return len(times)
    if name == "total":
        return math.fsum(numeric)
    if name == "mean":
        return math.fsum(numeric) / len(numeric) if numeric else math.nan
    if name == "min":
        return min(numeric) if numeric else math.nan
    if name == "max":
        return max(numeric) if numeric else math.nan
    if name == "last":
        return values[-1] if values else None
    if name == "distinct":
        return len(set(values))
    return (cutoff - times[-1]) / windows.NANOSECONDS if times else math.nan


def recompute(admitted, entity, cutoff):
    """The naive snapshot state: each kind.field's newest value."""
    state = {}
    rows = sorted(
        (
            (item.time, rank, item)
            for rank, item in enumerate(admitted)
            if item.entity == entity and item.time <= cutoff
        ),
        key=lambda triple: (triple[0], triple[1]),
    )
    for triple in rows:
        for field, value in triple[2].payload.items():
            state[f"{triple[2].kind}.{field}"] = value
    return dict(sorted(state.items()))


def same(left, right):
    """Exact equality where NaN equals NaN."""
    if isinstance(left, float) and isinstance(right, float):
        return (math.isnan(left) and math.isnan(right)) or left == right
    return left == right and type(left) is type(right)


def stream(seed, count, disorder):
    """Events in time order, then delivered with delays up to `disorder`."""
    generator = random.Random(seed)
    made = []
    for number in range(count):
        kind = generator.choice(["purchase", "purchase", "login", "refund"])
        payload = {"channel": generator.choice(["web", "app", "store"])}
        if kind != "login":
            payload["amount"] = round(generator.uniform(-50, 500), 2)
        if generator.random() < 0.1:
            payload["amount"] = "n/a"
        made.append(
            events.Event(
                entity=f"e{generator.randrange(5)}",
                time=START + number * HOUR // 4 + generator.randrange(3),
                kind=kind,
                payload=payload,
            )
        )
    return sorted(
        made, key=lambda item: item.time + generator.uniform(0, disorder)
    )


@pytest.mark.parametrize("seed", range(6))
def test_incremental_matches_naive_recompute(seed):
    spans = [windows.Window.parse(s) for s in SPECS]
    store = events.Store(spans, lateness=3 * HOUR)
    admitted = []
    generator = random.Random(seed + 100)
    arrival = stream(seed, 400, disorder=3 * HOUR)
    for position, item in enumerate(arrival, start=1):
        assert store.add(item) is not events.Arrival.DROPPED
        admitted.append(item)
        if position % 50:
            continue
        for ignored in range(25):
            entity = f"e{generator.randrange(6)}"
            cutoff = START + generator.randrange(-HOUR, 110 * HOUR)
            snapshot = store.snapshot(entity, cutoff)
            for window in spans:
                expected = naive(admitted, entity, cutoff, window)
                found = snapshot.features[window.name]
                assert same(found, expected), (window.name, found, expected)
            assert dict(snapshot.state) == recompute(admitted, entity, cutoff)
    assert store.tally[events.Arrival.DROPPED] == 0
    assert store.tally[events.Arrival.DISORDERED] > 0


def test_frame_matches_snapshots_and_orders_columns():
    spans = [windows.Window.parse(s) for s in SPECS[:3]]
    store = events.Store(spans, lateness=HOUR)
    store.extend(stream(1, 120, disorder=HOUR))
    queries = pandas.DataFrame(
        {
            "entity": ["e1", "e2", 3, "nobody"],
            "cutoff": [
                "2026-01-01T05:00:00Z",
                "2026-01-01 10:30",
                pandas.Timestamp("2026-01-03", tz="UTC"),
                "2026-01-01T05:00:00Z",
            ],
            "label": [1, 0, 1, 0],
        }
    )
    found = store.frame(queries)
    names = [w.name for w in spans]
    assert list(found.columns[:3]) == ["entity", "cutoff", "label"]
    assert list(found.columns[3:6]) == names
    assert found.columns[-1] == "final"
    for row, (who, when) in enumerate(
        zip(queries["entity"], queries["cutoff"], strict=True)
    ):
        snapshot = store.snapshot(str(who), event.instant(when))
        for name in names:
            assert same(found[name].iloc[row], snapshot.features[name])
    assert found["purchase.amount:total:6h"].iloc[3] == 0.0
    assert found["final"].tolist() == [True, True, False, True]


def test_window_bounds_are_open_below_and_closed_at_the_cutoff():
    store = events.Store([windows.Window.parse("buy.amount:total:1h")])
    for offset, amount in ((0, 1), (HOUR // 2, 10), (HOUR, 100)):
        store.add(events.Event("a", START + offset, "buy", {"amount": amount}))
    total = "buy.amount:total:1h"
    assert store.snapshot("a", START + HOUR).features[total] == 110.0
    assert store.snapshot("a", START + HOUR - 1).features[total] == 11.0
    assert store.snapshot("a", START - 1).features[total] == 0.0
    assert store.snapshot("a", START - 1).state == {}
    assert store.snapshot("a", START + HOUR // 2).state == {"buy.amount": 10}


def test_equal_times_order_by_arrival():
    store = events.Store([windows.Window.parse("buy.sku:last:1h")])
    store.add(events.Event("a", START, "buy", {"sku": "first"}))
    store.add(events.Event("a", START, "buy", {"sku": "second"}))
    snapshot = store.snapshot("a", START)
    assert snapshot.features["buy.sku:last:1h"] == "second"
    assert snapshot.state == {"buy.sku": "second"}


def primed(policy):
    store = events.Store(
        [windows.Window.parse("buy:count:10h")], lateness=HOUR, policy=policy
    )
    store.add(events.Event("a", START + 5 * HOUR, "buy"))
    return store


def test_lateness_classifies_each_arrival():
    store = primed(events.Lateness.DROP)
    assert store.watermark == START + 4 * HOUR
    assert store.add(events.Event("a", START + 6 * HOUR, "buy")) is (
        events.Arrival.TIMELY
    )
    assert store.watermark == START + 5 * HOUR
    assert store.add(events.Event("a", START + 5 * HOUR, "buy")) is (
        events.Arrival.DISORDERED
    )
    assert store.add(events.Event("a", START + 5 * HOUR - 1, "buy")) is (
        events.Arrival.DROPPED
    )
    assert store.snapshot("a", START + 6 * HOUR).features["buy:count:10h"] == 3
    assert dict(store.tally) == {
        events.Arrival.TIMELY: 2,
        events.Arrival.DISORDERED: 1,
        events.Arrival.DROPPED: 1,
    }


def test_reject_policy_raises_and_keeps_the_store_unchanged():
    store = primed(events.Lateness.REJECT)
    with pytest.raises(events.LatenessError) as caught:
        store.add(events.Event("a", START + 3 * HOUR, "buy"))
    assert caught.value.details == {"entity": "a", "time": START + 3 * HOUR}
    assert store.snapshot("a", START + 6 * HOUR).features["buy:count:10h"] == 1
    assert sum(store.tally.values()) == 1


def test_accept_policy_keeps_late_events_and_counts_them():
    store = primed(events.Lateness.ACCEPT)
    assert store.add(events.Event("a", START, "buy")) is events.Arrival.LATE
    assert store.snapshot("a", START + 6 * HOUR).features["buy:count:10h"] == 2
    assert store.tally[events.Arrival.LATE] == 1


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(
    "policy",
    [events.Lateness.DROP, events.Lateness.REJECT, events.Lateness.ACCEPT],
)
def test_final_snapshots_never_change(seed, policy):
    bound = 2 * HOUR
    spans = [windows.Window.parse(s) for s in SPECS]
    store = events.Store(spans, lateness=bound, policy=policy)
    arrival = stream(seed, 300, disorder=4 * bound)
    frozen = {}
    asked = 0
    late = 0
    generator = random.Random(seed)
    for position, item in enumerate(arrival):
        try:
            store.add(item)
        except events.LatenessError:
            late += 1
        if position % 20 == 0:
            cutoff = START + generator.randrange(80 * HOUR)
            entity = f"e{generator.randrange(5)}"
            snapshot = store.snapshot(entity, cutoff)
            asked += 1
            if snapshot.final:
                frozen[entity, cutoff] = snapshot
    late += (
        store.tally[events.Arrival.LATE] + store.tally[events.Arrival.DROPPED]
    )
    assert late >= 10
    if policy is events.Lateness.ACCEPT:
        assert asked == 15
        assert frozen == {}
        assert not store.final(START)
        return
    assert len(frozen) >= 3
    for (entity, cutoff), snapshot in frozen.items():
        later = store.snapshot(entity, cutoff)
        assert later.state == snapshot.state
        for name, value in snapshot.features.items():
            assert same(later.features[name], value)


def test_accept_policy_never_marks_a_snapshot_final():
    store = primed(events.Lateness.ACCEPT)
    before = store.snapshot("a", START + 2 * HOUR)
    assert before.final is False
    assert before.features["buy:count:10h"] == 0
    store.add(events.Event("a", START + HOUR, "buy"))
    after = store.snapshot("a", START + 2 * HOUR)
    assert after.final is False
    assert after.features["buy:count:10h"] == 1
    assert primed(events.Lateness.DROP).final(START + 2 * HOUR) is True


def test_store_refuses_bad_configuration():
    with pytest.raises(events.EventError, match="negative"):
        events.Store(lateness=-1)
    window = windows.Window.parse("a:count:1h")
    with pytest.raises(
        events.EventError, match=r"named twice: \['a:count:1h'\]"
    ):
        events.Store([window, window])


def test_wildcard_series_are_kept_only_when_a_window_reads_them():
    narrow = events.Store([windows.Window.parse("buy:count:1h")])
    narrow.add(events.Event("a", START, "buy", {"x": 1}))
    assert set(narrow.entities["a"]) == {("buy", None), ("buy", "x")}
    wide = events.Store([windows.Window.parse("*:count:1h")])
    wide.add(events.Event("a", START, "buy", {"x": 1}))
    assert set(wide.entities["a"]) == {
        ("buy", None),
        ("buy", "x"),
        ("*", None),
        ("*", "x"),
    }
    assert wide.snapshot("a", START).state == {"buy.x": 1}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("buy:count", "KIND"),
        ("buy:count:1h:x", "KIND"),
        ("buy::1h", "KIND"),
        ("buy.amount:median:1h", "unknown statistic 'median'"),
        ("buy.amount:total:0s", "positive"),
        ("buy.amount:total:-1h", "non-negative"),
        ("buy.amount:total:soon", "non-negative"),
        ("buy:total:1h", "needs a field"),
        ("buy.:count:1h", "KIND"),
        (".x:count:1h", "KIND"),
        (".:count:1h", "KIND"),
    ],
)
def test_window_parse_rejects(text, message):
    with pytest.raises(events.EventError, match=message):
        windows.Window.parse(text)


def test_window_parse_reads_dotted_fields_and_names():
    window = windows.Window.parse(" card.issuer.country : distinct : 30d ")
    assert window.kind == "card"
    assert window.field == "issuer.country"
    assert window.span == 30 * 24 * HOUR
    assert window.name == "card.issuer.country:distinct:30d"
    assert windows.Window.parse("*:count:90min").name == "*:count:90min"


def test_new_statistic_subclass_is_found_by_name():
    class Spread(windows.Statistic):
        name = "spread"

        def measure(self, times, values, cutoff):
            found = windows.numbers(values)
            return max(found) - min(found) if found else math.nan

    store = events.Store([windows.Window.parse("buy.amount:spread:1d")])
    for amount in (4, 9, 1):
        store.add(events.Event("a", START, "buy", {"amount": amount}))
    assert store.snapshot("a", START).features["buy.amount:spread:1d"] == 8.0


def test_numbers_drop_bools_text_and_nan():
    assert windows.numbers([1, 2.5, True, "3", math.nan, None]) == [1.0, 2.5]


def test_parse_reads_payload_objects_text_and_columns():
    columns = events.Columns()
    nested = event.parse(
        {
            "entity": 7.0,
            "time": "2026-01-01T01:00:00+01:00",
            "type": "buy",
            "payload": {"card": {"type": "visa"}, "amount": numpy.int64(3)},
        },
        columns,
    )
    assert nested == events.Event(
        "7", START, "buy", {"card.type": "visa", "amount": 3}
    )
    assert type(nested.payload["amount"]) is int
    text = event.parse(
        {
            "entity": "a",
            "time": "2026-01-01",
            "type": "x",
            "payload": '{"k":1}',
        },
        columns,
    )
    assert text.payload == {"k": 1}
    wide = event.parse(
        {"who": "a", "at": "2026-01-01", "what": "x", "k": 2, "gone": None},
        events.Columns(entity="who", time="at", kind="what"),
    )
    assert wide.payload == {"k": 2}


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"time": "2026-01-01", "type": "x"}, "needs the field 'entity'"),
        ({"entity": "a", "time": None, "type": "x"}, "needs the field 'time'"),
        ({"entity": "a", "time": 5, "type": "x"}, "the number 5"),
        (
            {"entity": "a", "time": numpy.int64(5), "type": "x"},
            "the number 5",
        ),
        (
            {"entity": "a", "time": numpy.float64(1.5), "type": "x"},
            "the number 1.5",
        ),
        ({"entity": "a", "time": "soon", "type": "x"}, "got 'soon'"),
        (
            {"entity": "a", "time": "2026-01-01", "type": "x", "payload": "{"},
            "not JSON",
        ),
        (
            {
                "entity": "a",
                "time": "2026-01-01",
                "type": "x",
                "payload": "[1]",
            },
            "must be an object",
        ),
        (
            {
                "entity": "a",
                "time": "2026-01-01",
                "type": "x",
                "payload": {"l": [1]},
            },
            "'l' must be a scalar",
        ),
    ],
)
def test_parse_rejects(record, message):
    with pytest.raises(events.EventError, match=message):
        event.parse(record, events.Columns())


def test_instant_reads_naive_times_as_utc():
    assert event.instant("2026-01-01 00:00") == START
    assert event.instant(pandas.Timestamp("2026-01-01")) == START
    assert event.instant(pandas.Timestamp("2026-01-01").to_pydatetime()) == (
        START
    )


def test_needed_is_the_smallest_sufficient_history():
    least = lateness.needed()
    assert least == 299
    assert 1 - 0.99**least >= 0.95 > 1 - 0.99 ** (least - 1)


def test_delays_measure_lag_behind_the_frontier():
    assert lateness.delays([5, 3, 9, 9, 1, 10]) == [0, 2, 0, 0, 8, 0]


def test_bound_is_the_binomial_order_statistic():
    generator = random.Random(3)
    times = list(range(1000))
    generator.shuffle(times)
    ordered = sorted(lateness.delays(times))
    rank = next(
        k for k in range(1, 1001) if stats.binom.cdf(k - 1, 1000, 0.99) >= 0.95
    )
    assert lateness.bound(times) == ordered[rank - 1]
    assert stats.binom.cdf(rank - 2, 1000, 0.99) < 0.95


def test_bound_covers_the_quantile_at_the_stated_rate():
    # Pairs of an in-order event and one `d` behind it give delays that are
    # half zeros and half exponential draws, so the 0.99 quantile of the
    # delays is the exponential's 0.98 quantile.
    generator = numpy.random.default_rng(11)
    gap = 10**9
    true = stats.expon.ppf(0.98, scale=1_000)
    runs = 300
    misses = 0
    for run in range(runs):
        drawn = generator.exponential(1_000, size=200).astype(int)
        times = [
            time
            for pair, delay in enumerate(drawn.tolist())
            for time in ((pair + 1) * gap, (pair + 1) * gap - delay)
        ]
        assert sorted(lateness.delays(times))[200:] == sorted(drawn), run
        misses += lateness.bound(times) < true
    # At most 5% of runs may miss: 30 of 300 lies 3.9 standard deviations
    # above the 15 a 5% rate gives.
    assert misses <= 30


def test_bound_refuses_short_history_and_bad_levels():
    with pytest.raises(events.EventError, match="at least 299 events, got 10"):
        lateness.bound(range(10))
    with pytest.raises(events.EventError, match="strictly between"):
        lateness.bound(range(400), coverage=1.0)


def test_file_source_reads_csv_and_jsonl(tmp_path):
    csv = tmp_path / "events.csv"
    csv.write_text(
        "entity,time,type,amount\n"
        "1,2026-01-01T00:00:00Z,buy,3.5\n"
        "2,2026-01-01 01:00,buy,\n",
        encoding="utf-8",
    )
    found = list(events.File(csv).events())
    assert found == [
        events.Event("1", START, "buy", {"amount": 3.5}),
        events.Event("2", START + HOUR, "buy", {}),
    ]
    lines = tmp_path / "events.jsonl"
    lines.write_text(
        json.dumps(
            {
                "entity": "a",
                "time": "2026-01-01T00:00:00Z",
                "type": "buy",
                "payload": {"card": {"type": "visa"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert list(events.File(lines).events()) == [
        events.Event("a", START, "buy", {"card.type": "visa"})
    ]


def test_strict_source_names_the_bad_record_and_lenient_skips_it():
    text = (
        '{"entity":"a","time":"2026-01-01","type":"x"}\n'
        "\n"
        "not json\n"
        "[1]\n"
        '{"entity":"a","type":"x"}\n'
        '{"entity":"b","time":"2026-01-01","type":"x"}\n'
    )
    with pytest.raises(events.EventError, match="record 3: not JSON") as caught:
        list(events.Lines(io.StringIO(text)).events())
    assert caught.value.details == {"position": 3}
    lenient = events.Lines(io.StringIO(text), strict=False)
    assert [e.entity for e in lenient.events()] == ["a", "b"]
    assert [pair[0] for pair in lenient.skipped] == [3, 4, 5]
    assert lenient.skipped[1][1] == "not a JSON object"


def test_tail_reads_complete_lines_as_the_file_grows(tmp_path):
    path = tmp_path / "live.jsonl"
    tail = events.Tail(path)
    assert list(tail.events()) == []
    line = '{"entity":"a","time":"2026-01-01","type":"x"}\n'
    path.write_text(line + line[:10], encoding="utf-8")
    assert len(list(tail.events())) == 1
    assert tail.offset == len(line)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line[10:])
    assert [e.entity for e in tail.events()] == ["a"]
    assert tail.offset == 2 * len(line)
    resumed = events.Tail(path, offset=len(line))
    assert len(list(resumed.events())) == 1
    path.write_text(line[:5], encoding="utf-8")
    with pytest.raises(events.EventError, match="truncated or replaced"):
        list(tail.events())
    with pytest.raises(events.EventError, match="negative"):
        events.Tail(path, offset=-1)


def test_strict_tail_raises_on_a_malformed_line(tmp_path):
    path = tmp_path / "live.jsonl"
    path.write_text(
        'not json\n{"entity":"a","time":"2026-01-01","type":"x"}\n',
        encoding="utf-8",
    )
    with pytest.raises(events.EventError, match="record 1: not JSON") as caught:
        list(events.Tail(path).events())
    assert caught.value.details == {"position": 1}


def test_tail_reports_invalid_utf8(tmp_path):
    path = tmp_path / "live.jsonl"
    path.write_bytes(b"\xff\xfe\n")
    tail = events.Tail(path, strict=False)
    assert list(tail.events()) == []
    assert tail.skipped == [(1, "not UTF-8 text")]


@pytest.mark.parametrize(
    ("seed", "policy"),
    [
        (seed, policy)
        for seed in range(3)
        for policy in (events.Lateness.DROP, events.Lateness.ACCEPT)
    ],
)
def test_late_events_match_naive_recompute_over_what_was_kept(seed, policy):
    spans = [windows.Window.parse(s) for s in SPECS]
    store = events.Store(spans, lateness=HOUR, policy=policy)
    kept = []
    for item in stream(seed, 300, disorder=4 * HOUR):
        frontier = store.frontier
        arrival = store.add(item)
        late = frontier is not None and item.time < frontier - HOUR
        expected = {
            events.Lateness.DROP: events.Arrival.DROPPED,
            events.Lateness.ACCEPT: events.Arrival.LATE,
        }[policy]
        assert (arrival is expected) == late
        if arrival is not events.Arrival.DROPPED:
            kept.append(item)
    assert store.tally[expected] > 0
    generator = random.Random(seed)
    for ignored in range(150):
        entity = f"e{generator.randrange(5)}"
        cutoff = START + generator.randrange(80 * HOUR)
        snapshot = store.snapshot(entity, cutoff)
        for window in spans:
            found = snapshot.features[window.name]
            assert same(found, naive(kept, entity, cutoff, window))
        assert dict(snapshot.state) == recompute(kept, entity, cutoff)


def test_frame_refuses_answers_that_collide_with_query_columns():
    store = events.Store([windows.Window.parse("buy:count:1h")])
    store.add(events.Event("a", START, "label", {"x": 1}))
    queries = pandas.DataFrame(
        {"entity": ["a"], "cutoff": ["2026-01-01"], "label.x": [0]}
    )
    with pytest.raises(events.EventError, match=r"\['label.x'\]"):
        store.frame(queries)
