"""The event-stream command line: features, follow and lateness."""

import io
import json
import threading

import pandas
import pytest

from scmle.cli import command
from scmle.data.events import __main__


def line(entity, time, kind, **payload):
    return (
        json.dumps(
            {
                "entity": entity,
                "time": time,
                "type": kind,
                "payload": payload,
            }
        )
        + "\n"
    )


@pytest.fixture
def log(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        line("a", "2026-01-01T00:00:00Z", "buy", amount=10)
        + line("a", "2026-01-01T02:00:00Z", "buy", amount=5)
        + line("b", "2026-01-01T01:00:00Z", "buy", amount=7)
        + line("a", "2026-01-01T00:30:00Z", "buy", amount=1000)
        + line("a", "2026-01-01T03:00:00Z", "login", channel="app"),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def queries(tmp_path):
    path = tmp_path / "queries.csv"
    path.write_text(
        "entity,cutoff\na,2026-01-01T02:00:00Z\nb,2026-01-01T02:00:00Z\n"
        "a,2026-01-01T09:00:00Z\n",
        encoding="utf-8",
    )
    return path


def test_features_drops_late_events_and_writes_the_table(
    log, queries, tmp_path, capsys
):
    output = tmp_path / "features.csv"
    code = __main__.main(
        [
            "features",
            str(log),
            "--queries",
            str(queries),
            "--window",
            "buy.amount:total:3h",
            "--window",
            "*:count:1d",
            "--lateness",
            "1h",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    found = pandas.read_csv(output)
    assert found["buy.amount:total:3h"].tolist() == [15.0, 7.0, 0.0]
    assert found["*:count:1d"].tolist() == [2, 1, 3]
    assert found["buy.amount"].tolist() == [5, 7, 5]
    assert found["login.channel"].fillna("").tolist() == ["", "", "app"]
    assert found["final"].tolist() == [False, False, False]
    summary = json.loads(capsys.readouterr().err)
    assert summary == {
        "arrivals": {"timely": 3, "disordered": 1, "late": 0, "dropped": 1},
        "lateness_seconds": 3600.0,
        "watermark": "2026-01-01T02:00:00+00:00",
        "skipped": 0,
    }


def test_features_accept_policy_keeps_late_events(log, queries, capsys):
    code = __main__.main(
        [
            "features",
            str(log),
            "--queries",
            str(queries),
            "--window",
            "buy.amount:total:3h",
            "--lateness",
            "1h",
            "--policy",
            "accept",
        ]
    )
    assert code == 0
    printed = capsys.readouterr()
    rows = printed.out.splitlines()
    assert rows[0] == (
        "entity,cutoff,buy.amount:total:3h,buy.amount,login.channel,final"
    )
    assert rows[1].split(",")[2] == "1015.0"
    assert json.loads(printed.err)["arrivals"]["late"] == 1


def test_features_reports_input_errors(log, queries, capsys):
    code = __main__.main(
        [
            "features",
            str(log),
            "--queries",
            str(queries),
            "--window",
            "buy.amount:median:3h",
        ]
    )
    assert code == 2
    assert (
        "unknown statistic 'median'"
        in json.loads(capsys.readouterr().err)["error"]
    )


def test_follow_answers_queries_as_the_watermark_passes(log, queries, capsys):
    def append():
        with log.open("a", encoding="utf-8") as handle:
            handle.write(line("c", "2026-01-01T12:00:00Z", "buy", amount=1))

    timer = threading.Timer(0.3, append)
    timer.start()
    try:
        code = __main__.main(
            [
                "follow",
                str(log),
                "--queries",
                str(queries),
                "--window",
                "buy.amount:total:3h",
                "--lateness",
                "1h",
                "--poll",
                "0.05",
                "--idle",
                "5",
            ]
        )
    finally:
        timer.join()
    assert code == 0
    printed = capsys.readouterr()
    answers = [json.loads(row) for row in printed.out.splitlines()]
    assert [(a["entity"], a["final"]) for a in answers] == [
        ("a", True),
        ("b", True),
        ("a", True),
    ]
    assert [a["buy.amount:total:3h"] for a in answers] == [15.0, 7.0, 0.0]
    assert answers[0]["cutoff"] == "2026-01-01T02:00:00+00:00"
    assert json.loads(printed.err)["arrivals"]["timely"] == 4


def test_follow_flushes_open_queries_when_idle(log, queries, capsys):
    code = __main__.main(
        [
            "follow",
            str(log),
            "--queries",
            str(queries),
            "--window",
            "buy:count:1d",
            "--lateness",
            "30min",
            "--poll",
            "0.01",
            "--idle",
            "0.05",
        ]
    )
    assert code == 0
    answers = [json.loads(r) for r in capsys.readouterr().out.splitlines()]
    assert [(a["entity"], a["final"]) for a in answers] == [
        ("a", True),
        ("b", True),
        ("a", False),
    ]
    assert answers[2]["buy:count:1d"] == 2


def test_lateness_estimates_a_bound(tmp_path, capsys):
    path = tmp_path / "events.csv"
    rows = ["entity,time,type"]
    for number in range(400):
        minutes = number * 10 - (15 if number % 2 else 0)
        stamp = pandas.Timestamp("2026-01-01", tz="UTC") + pandas.Timedelta(
            minutes=minutes
        )
        rows.append(f"a,{stamp.isoformat()},x")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert __main__.main(["lateness", str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "lateness": "0 days 00:05:00",
        "lateness_seconds": 300.0,
        "coverage": 0.99,
        "confidence": 0.95,
        "events": 400,
        "skipped": 0,
    }
    assert __main__.main(["lateness", str(path), "--coverage", "0.999"]) == 2
    assert "needs at least 2995 events" in capsys.readouterr().err


def test_auto_lateness_uses_the_backlog(log, queries, capsys):
    code = __main__.main(
        [
            "features",
            str(log),
            "--queries",
            str(queries),
            "--window",
            "buy:count:1d",
            "--lateness",
            "auto",
        ]
    )
    assert code == 2
    assert "needs at least 299 events, got 5" in capsys.readouterr().err


def test_features_reads_events_piped_on_stdin(
    log, queries, monkeypatch, capsys
):
    piped = log.read_text(encoding="utf-8") + "not json\n"
    monkeypatch.setattr("sys.stdin", io.StringIO(piped))
    code = __main__.main(
        [
            "features",
            "-",
            "--queries",
            str(queries),
            "--window",
            "buy.amount:total:3h",
            "--lateness",
            "1h",
            "--lenient",
        ]
    )
    assert code == 0
    printed = capsys.readouterr()
    rows = [row.split(",") for row in printed.out.splitlines()[1:]]
    assert [row[2] for row in rows] == ["15.0", "7.0", "0.0"]
    assert json.loads(printed.err)["skipped"] == 1


def features(log, queries, *extra):
    return __main__.main(
        [
            "features",
            str(log),
            "--queries",
            str(queries),
            "--window",
            "buy.amount:total:3h",
            "--lateness",
            "1h",
            *extra,
        ]
    )


@pytest.mark.parametrize("suffix", [".parquet", ".jsonl"])
def test_features_writes_each_output_format(
    log, queries, tmp_path, capsys, suffix
):
    table = tmp_path / "features.csv"
    assert features(log, queries, "--output", str(table)) == 0
    output = tmp_path / f"features{suffix}"
    assert features(log, queries, "--output", str(output)) == 0
    capsys.readouterr()
    expected = pandas.read_csv(table)
    if suffix == ".parquet":
        found = pandas.read_parquet(output)
    else:
        found = pandas.read_json(output, lines=True)
    assert (
        found.columns.tolist()
        == expected.columns.tolist()
        == [
            "entity",
            "cutoff",
            "buy.amount:total:3h",
            "buy.amount",
            "login.channel",
            "final",
        ]
    )
    assert found["entity"].tolist() == expected["entity"].tolist()
    assert pandas.to_datetime(found["cutoff"], utc=True).tolist() == (
        pandas.to_datetime(expected["cutoff"], utc=True).tolist()
    )
    assert found["buy.amount:total:3h"].tolist() == [15.0, 7.0, 0.0]
    assert found["buy.amount"].tolist() == expected["buy.amount"].tolist()
    assert found["login.channel"].fillna("").tolist() == ["", "", "app"]
    assert found["final"].tolist() == expected["final"].tolist()


def test_features_refuses_an_unknown_output_suffix(
    log, queries, tmp_path, capsys
):
    output = tmp_path / "features.txt"
    assert features(log, queries, "--output", str(output)) == 2
    assert json.loads(capsys.readouterr().err)["error"].startswith(
        "--output must end in .csv, .parquet or .jsonl"
    )
    assert not output.exists()


@pytest.mark.parametrize("action", ["features", "follow"])
def test_queries_without_a_cutoff_column_exit_2(log, tmp_path, capsys, action):
    queries = tmp_path / "queries.csv"
    queries.write_text(
        "entity,when\na,2026-01-01T02:00:00Z\n", encoding="utf-8"
    )
    code = __main__.main(
        [
            action,
            str(log),
            "--queries",
            str(queries),
            "--window",
            "buy:count:1d",
            *(["--idle", "0"] if action == "follow" else []),
        ]
    )
    assert code == 2
    assert json.loads(capsys.readouterr().err) == {
        "error": "the queries need a 'cutoff' column"
    }


def test_features_reject_policy_exits_2_on_a_late_event(log, queries, capsys):
    assert features(log, queries, "--policy", "reject") == 2
    printed = capsys.readouterr()
    assert printed.out == ""
    assert "older than the watermark" in json.loads(printed.err)["error"]


def test_follow_accept_policy_answers_only_at_idle(log, queries, capsys):
    code = __main__.main(
        [
            "follow",
            str(log),
            "--queries",
            str(queries),
            "--window",
            "buy.amount:total:3h",
            "--lateness",
            "1h",
            "--policy",
            "accept",
            "--poll",
            "0.01",
            "--idle",
            "0.05",
        ]
    )
    assert code == 0
    answers = [json.loads(r) for r in capsys.readouterr().out.splitlines()]
    assert [(a["entity"], a["final"]) for a in answers] == [
        ("a", False),
        ("b", False),
        ("a", False),
    ]
    assert [a["buy.amount:total:3h"] for a in answers] == [1015.0, 7.0, 0.0]


def test_exit_codes_match_the_command_line():
    assert __main__.EXIT_OK == command.EXIT_OK
    assert __main__.EXIT_INPUT == command.EXIT_BLOCKED
