"""Read the published scouting board, provisional pick ownership, and team priorities."""

import copy
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
    for soup in soups.values():
        heading = soup.find("h1")
        if not heading or str(board["season"]) not in heading.get_text():
            raise ValueError("Draft year changed; a reviewed new edition is required")
    names = {v["name"]: k for k, v in board["teams"].items()}
    board["order_date"] = reviewed_date(soups["order"], "Updated", now)
    board["rank_date"] = reviewed_date(soups["rank"], "Big Board reviewed", now)
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
    prospects = []
    seen = set()
    ranks = set()
    for row in soups["rank"].select("[data-pfm-big-board-row]"):
        rank = int(row.select_one('[data-label="Rank"] strong').get_text(strip=True))
        if rank > 40:
            continue
        name = row.select_one('[data-label="Prospect"] strong').get_text(strip=True)
        school = row.select_one('[data-label="School"]').get_text(strip=True)
        position = row["data-position"]
        # Notre Dame confirms Brauntae and Tae Johnson are the same player.
        if school == "Notre Dame" and name == "Brauntae Johnson":
            name = "Tae Johnson"
        key = (name.casefold(), school.casefold())
        if key in seen:
            continue
        if position not in POSITIONS or rank in ranks:
            raise ValueError("Unreviewed prospect position or duplicate scouting rank")
        seen.add(key)
        ranks.add(rank)
        prospects.append(
            dict(
                id=re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-"),
                name=name,
                school=school,
                position=position,
                rank=rank,
            )
        )
    if len(prospects) < 39 or len({p["id"] for p in prospects}) != len(prospects):
        raise ValueError("Incomplete or ambiguous scouting board")
    for player in seed["prospects"]:
        if player["rank"] is None and player["id"] not in {p["id"] for p in prospects}:
            prospects.append(player)
    board["prospects"] = prospects
    board["edition"] = (
        f"{board['season']}-{'-'.join(board[k] for k in ('order_date', 'rank_date', 'needs_date'))}"
    )
    return board
