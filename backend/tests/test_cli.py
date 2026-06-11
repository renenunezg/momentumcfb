import pytest

from backend.cli import parse_args


def test_parse_ingest_defaults():
    args = parse_args(["ingest"])
    assert args.command == "ingest"
    assert 2019 in args.seasons


def test_parse_backtest():
    args = parse_args(["backtest", "--variant", "all", "--method", "nuts"])
    assert args.variant == "all"
    assert args.method == "nuts"


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        parse_args(["nope"])
