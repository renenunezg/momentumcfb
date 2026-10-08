"""Read the published scouting board, provisional pick ownership, and team priorities."""

import copy
import json
import re
from datetime import datetime

from bs4 import BeautifulSoup

POSITIONS = {"QB", "RB", "WR", "TE", "OT", "IOL", "EDGE", "DT", "LB", "CB", "S"}


def reviewed_date(soup, prefix, now):
    match = re.search(
        prefix + r" ([A-Z][a-z]+ \d{1,2}, \d{4})", soup.get_text(" ", strip=True)
    )
    if not match:
        raise ValueError(f"Missing source review date: {prefix}")
    value = datetime.strptime(match[1], "%B %d, %Y").date()
    if not 0 <= (now.date() - value).days <= 35:
        raise ValueError(f"Source date is stale or in the future: {value}")
    return value.isoformat()


def parse_board(pages, seed, now):
    board = copy.deepcopy(seed)
    soups = {key: BeautifulSoup(value, "html.parser") for key, value in pages.items()}
    for source, soup in soups.items():
        heading = soup.title if source == "rank" and soup.title else soup.find("h1")
        if not heading or str(board["season"]) not in heading.get_text():
            raise ValueError("Draft year changed; a reviewed new edition is required")
    names = {v["name"]: k for k, v in board["teams"].items()}
    board["order_date"] = reviewed_date(soups["order"], "Updated", now)
    board["rank_date"] = reviewed_date(soups["rank"], "Board updated", now)
    board["ownership_date"] = reviewed_date(
        soups["capital"], "Pick ownership reviewed", now
    )
    board["rank_source"] = "Scouting Grade"
    board["needs_date"] = reviewed_date(soups["needs"], "Needs reviewed", now)
    picks = []
    for row in soups["order"].select(
        ".pfm-draft-table tbody tr:has(.pfm-draft-pick-no)"
    ):
        original = names[row.select_one(".pfm-draft-team strong").get_text(strip=True)]
        owner_node = row.select_one(".pfm-draft-owner strong")
        owner = names[owner_node.get_text(strip=True)] if owner_node else original
        pick = int(row.select_one(".pfm-draft-pick-no").get_text(strip=True))
        picks.append(
            dict(
                pick=pick,
                round=1,
                compensatory=False,
                original=original,
                owner=owner,
                ownership_note="Own first-round pick."
                if owner == original
                else f"Acquired from {board['teams'][original]['name']}; ownership follows the current source assessment.",
                ownership_source=board["sources"]["order"],
                playoff_projection=pick >= 19,
            )
        )
    if [p["pick"] for p in picks] != list(range(1, 33)) or len(
        {p["original"] for p in picks}
    ) != 32:
        raise ValueError("Expected exactly 32 unique original first-round picks")
    board["picks"] = picks
    seen = set()
    for card in soups["needs"].select(".pfm-draft-team-needs-card"):
        team = names[card.select_one("h3").get_text(strip=True)]
        needs = card["data-pfm-draft-positions"].split()
        if (
            team in seen
            or len(needs) != 3
            or len(set(needs)) != 3
            or not set(needs) <= POSITIONS
        ):
            raise ValueError("Invalid or duplicate team priorities")
        board["teams"][team]["needs"] = needs
        seen.add(team)
    if seen != set(board["teams"]):
        raise ValueError("Missing team priorities")
    projected_originals = {p["original"] for p in picks if p["playoff_projection"]}
    full_picks = []
    seen_teams = set()
    aliases = {"LAR": "LA", "WSH": "WAS"}
    for card in soups["capital"].select(".pfm-draft-capital-team-card"):
        owner = names[
            card.select_one(".pfm-draft-team-name-line > strong").get_text(strip=True)
        ]
        if owner in seen_teams:
            raise ValueError("Duplicate pick ownership team")
        seen_teams.add(owner)
        for chip in card.select(".pfm-draft-pick-chip"):
            match = re.fullmatch(
                r"R([1-7]) · No\. (\d+)", chip.strong.get_text(strip=True)
            )
            if not match:
                raise ValueError("Unrecognized pick number or round")
            round_number, number = map(int, match.groups())
            note = chip.small.get_text(strip=True)
            compensatory = note == "Projected comp"
            if note in ("Own pick", "Projected comp"):
                original = owner
            elif note.startswith("via "):
                original = aliases.get(note[4:], note[4:])
            else:
                raise ValueError("Unrecognized pick ownership note")
            if original not in board["teams"]:
                raise ValueError("Unknown original pick owner")
            full_picks.append(
                dict(
                    pick=number,
                    round=round_number,
                    owner=owner,
                    original=original,
                    compensatory=compensatory,
                    ownership_note="Projected compensatory pick; not yet awarded."
                    if compensatory
                    else note,
                    ownership_source=board["sources"]["capital"],
                    playoff_projection=original in projected_originals
                    and not compensatory,
                )
            )
    full_picks.sort(key=lambda p: p["pick"])
    if seen_teams != set(board["teams"]) or not 224 <= len(full_picks) <= 272:
        raise ValueError("Incomplete seven-round pick ownership")
    if [p["pick"] for p in full_picks] != list(range(1, len(full_picks) + 1)):
        raise ValueError("Missing or duplicate overall pick numbers")
    if [p["round"] for p in full_picks] != sorted(p["round"] for p in full_picks):
        raise ValueError("Draft rounds are out of order")
    for round_number in range(1, 8):
        native = [
            p["original"]
            for p in full_picks
            if p["round"] == round_number and not p["compensatory"]
        ]
        if len(native) != 32 or set(native) != set(board["teams"]):
            raise ValueError("Expected 32 unique native picks in each round")
    firsts = [p for p in full_picks if p["round"] == 1]
    if [(p["pick"], p["original"], p["owner"]) for p in firsts] != [
        (p["pick"], p["original"], p["owner"]) for p in picks
    ]:
        raise ValueError("First-round order and full pick ownership disagree")
    board["picks"] = full_picks
    data = soups["rank"].select_one("#boardData")
    if data is None:
        raise ValueError("Missing complete scouting board")
    prospects = []
    for row in json.loads(data.get_text()):
        if not isinstance(row, list) or len(row) < 6:
            raise ValueError("Changed scouting board format")
        identifier, name, position, school, _, rank = row[:6]
        if position not in POSITIONS or not isinstance(rank, int) or rank < 1:
            raise ValueError("Unreviewed position or invalid scouting rank")
        if school == "Notre Dame" and name == "Brauntae Johnson":
            name = "Tae Johnson"
        prospects.append(
            dict(id=identifier, name=name, school=school, position=position, rank=rank)
        )
    prospects.sort(key=lambda p: p["rank"])
    identities = {
        (re.sub(r"[^a-z0-9]", "", p["name"].casefold()), p["school"]) for p in prospects
    }
    if (
        len(prospects) < len(full_picks)
        or len({p["id"] for p in prospects}) != len(prospects)
        or len(identities) != len(prospects)
        or [p["rank"] for p in prospects] != list(range(1, len(prospects) + 1))
    ):
        raise ValueError("Incomplete or ambiguous scouting board")
    board["prospects"] = prospects
    board["edition"] = (
        f"{board['season']}-seven-round-{'-'.join(board[k] for k in ('order_date', 'rank_date', 'needs_date', 'ownership_date'))}"
    )
    return board
