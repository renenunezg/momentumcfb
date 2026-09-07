"""One-off fetch of Heisman voting tables into the versioned seed CSV.

Wikipedia's season pages carry the top-ten ballot table in a stable
wikitable layout (Player, School, Position, 1st, 2nd, 3rd, Total). The CSV
is source data checked into the repo, refreshed once a year after the vote.
"""

import csv
import re
from pathlib import Path

import requests

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "momentumcfb/1.0 (https://github.com/renenunezg/momentumcfb)"
SEED_PATH = Path(__file__).resolve().parent / "data" / "heisman_votes.csv"
SEED_COLUMNS = [
    "season",
    "player",
    "school",
    "position",
    "first",
    "second",
    "third",
    "points",
    "source",
]

_LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]")
_REF = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.DOTALL)
_MARKUP = re.compile(r"'''|''|\{\{[^}]*\}\}")


def _clean(cell: str) -> str:
    cell = _REF.sub("", cell)
    cell = _LINK.sub(r"\1", cell)
    cell = _MARKUP.sub("", cell)
    return cell.strip()


def _page_title(season: int) -> str:
    return f"{season} NCAA Division I FBS football season"


def _wikitext(title: str) -> str:
    response = requests.get(
        WIKIPEDIA_API,
        params={
            "action": "parse",
            "page": title,
            "prop": "wikitext",
            "format": "json",
            "formatversion": 2,
        },
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"{title}: {payload['error'].get('info')}")
    return payload["parse"]["wikitext"]


def _voting_table(text: str) -> str:
    for match in re.finditer(r"\{\|[\s\S]*?\n\|\}", text):
        block = match.group(0)
        header = block.split("\n|-", 1)[0]
        if "Player" in header and "1st" in header and "Total" in header:
            return block
    raise RuntimeError("no Heisman voting table found")


def parse_voting_table(block: str, season: int, source: str) -> list[dict]:
    rows = []
    for raw_row in block.split("\n|-")[1:]:
        lines = [line for line in raw_row.strip().split("\n") if line.startswith("|")]
        cells = []
        for line in lines:
            if line.startswith("|}"):
                continue
            cells.extend(part for part in line[1:].split("||"))
        cells = [_clean(cell) for cell in cells]
        if len(cells) < 7:
            continue
        player, school, position, first, second, third, total = cells[:7]
        if not player or not total:
            continue
        rows.append(
            {
                "season": season,
                "player": player,
                "school": school,
                "position": position,
                "first": int(first.replace(",", "") or 0),
                "second": int(second.replace(",", "") or 0),
                "third": int(third.replace(",", "") or 0),
                "points": int(total.replace(",", "")),
                "source": source,
            }
        )
    if not rows:
        raise RuntimeError(f"{season}: voting table parsed to zero rows")
    return rows


def fetch_heisman_votes(seasons: list[int]) -> list[dict]:
    rows = []
    for season in seasons:
        title = _page_title(season)
        source = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
        rows.extend(parse_voting_table(_voting_table(_wikitext(title)), season, source))
    return rows


def write_seed(rows: list[dict], path: Path = SEED_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
