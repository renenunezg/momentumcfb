"""Release gates for the roster-to-draft boundary and source completeness."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.draft.board_source import parse_board
from backend.draft.positions import POSITION_NAMES, draft_role


def test_every_depth_role_has_an_explicit_draft_mapping():
    for code in POSITION_NAMES:
        for package in ("Base 3-4 D", "Base 4-3 D"):
            role, group = draft_role(code, package)
            assert group
            assert role is not None or code in ("H", "KR", "LS", "P", "PK", "PR")
    assert draft_role("RCB", "Base 3-4 D") == ("CB", "Secondary")
    assert draft_role("NB", "Base 4-3 D") == ("CB", "Nickel option")
    assert draft_role("FB", "3WR 1TE") == ("RB", "Fullback option")
    for side in ("LDE", "RDE"):
        assert draft_role(side, "Base 3-4 D")[0] == "DT"
        assert draft_role(side, "Base 4-3 D")[0] == "EDGE"
    for side in ("WLB", "SLB"):
        assert draft_role(side, "Base 3-4 D")[0] == "EDGE"
        assert draft_role(side, "Base 4-3 D")[0] == "LB"
    with pytest.raises(ValueError, match="Unreviewed depth-chart position"):
        draft_role("NEW", "Base 4-3 D")
    with pytest.raises(ValueError, match="Unreviewed defensive package"):
        draft_role("LDE", "New package")


def test_source_changes_cannot_silently_republish_the_seed():
    seed = json.loads(Path("backend/draft/board_2027.json").read_text())
    now = datetime(2026, 10, 7, tzinfo=timezone.utc)
    pages = {key: "<h1>2027 NFL Draft</h1>" for key in seed["sources"]}
    with pytest.raises(ValueError, match="Missing source review date"):
        parse_board(pages, seed, now)
    pages.update(order="<h1>2027 NFL Draft</h1>Updated September 1, 2026")
    with pytest.raises(ValueError, match="stale or in the future"):
        parse_board(pages, seed, now)
    pages.update(order="<h1>2028 NFL Draft</h1>Updated October 7, 2026")
    with pytest.raises(ValueError, match="Draft year changed"):
        parse_board(pages, seed, now)


def test_captured_sources_preserve_owners_and_deduplicate_real_player():
    seed = json.loads(Path("backend/draft/board_2027.json").read_text())
    pages = json.loads(Path("tests/fixtures/draft/source_tables.json").read_text())
    now = datetime(2026, 10, 7, tzinfo=timezone.utc)
    board = parse_board(pages, seed, now)
    assert [p["pick"] for p in board["picks"]] == list(range(1, 33))
    assert [p["pick"] for p in board["picks"] if p["owner"] == "NYJ"] == [6, 11, 12]
    assert board["picks"][13]["original"] == "GB"
    assert board["picks"][13]["owner"] == "DAL"
    assert board["picks"][20]["owner"] == "CLE"
    assert len(board["prospects"]) == 40
    assert len({p["id"] for p in board["prospects"]}) == 40
    assert [
        p["name"]
        for p in board["prospects"]
        if p["school"] == "Notre Dame" and p["position"] == "S"
    ] == ["Tae Johnson"]
    assert board["teams"]["HOU"]["needs"] == ["OT", "IOL", "WR"]
    pages["order"] = pages["order"].replace(
        'pfm-draft-pick-no">1<', 'pfm-draft-pick-no">2<', 1
    )
    with pytest.raises(ValueError, match="32 unique"):
        parse_board(pages, seed, now)
